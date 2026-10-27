#!/usr/bin/env python3
"""Static scanner for host-device sync candidates in PyTorch code.

This is a STATIC scanner. It finds call sites that commonly cause an
implicit host-device synchronization (blocking D2H copy, scalar read that
forces the GPU queue to drain) or that are preconditions for one (boolean
mask indexing, ``nonzero``, ...). It does NOT prove that any of these sites
actually execute, or that they actually synchronize at runtime -- some may
be on code paths that never run, operate on tensors that are already on
CPU, or run when the GPU queue happens to be idle anyway (no observable
cost). Treat every finding as a candidate to verify with a profiler trace
(see references/profiling.md and scripts/analyze_trace_sync.py), not as a
confirmed bug.

Usage:
    python3 scan_sync_points.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_sync_points.py --json <file-or-dir> ...

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Iterable


# (pattern name, human note, severity: "high" sync-causing | "medium" likely precondition | "info" verify-only)
_METHOD_PATTERNS = {
    "item": ("high", "Reads a tensor value to Python; a common blocking D2H sync."),
    "cpu": ("high", "Moves a tensor to host; blocking unless the source is pinned and the caller uses non_blocking=True with a proper wait."),
    "tolist": ("high", "Reads tensor values to a Python list; implies a D2H sync if the tensor is on a CUDA/HIP device."),
    "numpy": ("high", "Converts a tensor to a NumPy array; requires the tensor to already be on CPU, otherwise this raises -- but if called right after .cpu() it is part of the same sync."),
    "nonzero": ("medium", "Output shape depends on tensor contents; may force the allocator to know the count, a common sync precondition."),
    "masked_select": ("medium", "Output shape depends on the mask; same precondition as nonzero."),
    "unique": ("medium", "Output size depends on tensor contents."),
    "bincount": ("medium", "Output size can depend on tensor contents unless `minlength` fixes it."),
    "repeat_interleave": ("medium", "Without `output_size`, PyTorch may need to read `repeats` to size the output."),
    "synchronize": ("high", "Explicit synchronization call (cuda.synchronize / stream.synchronize / event.synchronize)."),
    "all": ("info", "Often precedes a Python-side branch on a boolean tensor; check the call site, not this call alone."),
    "any": ("info", "Often precedes a Python-side branch on a boolean tensor; check the call site, not this call alone."),
}

_TORCH_FUNCTION_PATTERNS = {
    "torch.cuda.synchronize": ("high", "Explicit device-wide synchronization."),
    "torch.cuda.Event": ("info", "Event objects are fine; verify .synchronize() calls elsewhere and that record()/wait() pairs are correct."),
    "torch.nonzero": ("medium", "Same as tensor.nonzero(); output shape depends on contents."),
    "torch.masked_select": ("medium", "Same as tensor.masked_select()."),
    "torch.where": ("info", "Single-argument torch.where(cond) behaves like nonzero (shape depends on data); torch.where(cond, a, b) does not sync. Check the arg count at the call site."),
}

_BUILTIN_SCALAR_CASTS = {"int", "float", "bool"}


@dataclass
class Finding:
    file: str
    line: int
    col: int
    pattern: str
    severity: str
    snippet: str
    note: str


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str, source_lines: list[str]):
        self.filename = filename
        self.source_lines = source_lines
        self.findings: list[Finding] = []

    def _snippet(self, node: ast.AST) -> str:
        line_no = getattr(node, "lineno", None)
        if line_no is None or line_no > len(self.source_lines):
            return ""
        return self.source_lines[line_no - 1].strip()

    def _emit(self, node: ast.AST, pattern: str, severity: str, note: str) -> None:
        self.findings.append(
            Finding(
                file=self.filename,
                line=getattr(node, "lineno", -1),
                col=getattr(node, "col_offset", -1),
                pattern=pattern,
                severity=severity,
                snippet=self._snippet(node),
                note=note,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Method call: tensor.item(), tensor.nonzero(), ...
        if isinstance(func, ast.Attribute):
            name = func.attr
            if name in _METHOD_PATTERNS:
                severity, note = _METHOD_PATTERNS[name]
                self._emit(node, f".{name}()", severity, note)
            dotted = _dotted_name(func)
            if dotted in _TORCH_FUNCTION_PATTERNS:
                severity, note = _TORCH_FUNCTION_PATTERNS[dotted]
                self._emit(node, dotted, severity, note)
        elif isinstance(func, ast.Name) and func.id in _BUILTIN_SCALAR_CASTS:
            # int(x) / float(x) / bool(x) -- only interesting if the single
            # argument looks like it could be a tensor (we cannot know the
            # type statically, so this is always a low-confidence hint).
            if len(node.args) == 1 and not isinstance(
                node.args[0], (ast.Constant, ast.Num)
            ):
                self._emit(
                    node,
                    f"{func.id}(...)",
                    "info",
                    f"{func.id}() on a non-literal argument; if the argument is a tensor this reads its value to Python (D2H sync). Verify the argument type at this call site.",
                )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # `if tensor:` or `if tensor.any():` etc. -- flag boolean tests that
        # are not an obvious Python-native expression (name/constant/compare
        # of names), since those are the ones most likely hiding an implicit
        # `.item()`/`__bool__()` call on a tensor.
        test = node.test
        if isinstance(test, (ast.Call, ast.Attribute)):
            self._emit(
                node,
                "if <call-or-attr>",
                "info",
                "Branch condition is a call or attribute access; if it evaluates a tensor's truthiness (e.g. a 0-dim bool tensor), this is an implicit D2H sync. Verify the type of the test expression.",
            )
        self.generic_visit(node)


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _iter_python_files(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            yield path
        elif os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for name in files:
                    if name.endswith(".py"):
                        yield os.path.join(root, name)


def scan_file(path: str) -> list[Finding]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        print(f"warning: could not parse {path}: {exc}", file=sys.stderr)
        return []
    visitor = _Visitor(path, source.splitlines())
    visitor.visit(tree)
    return visitor.findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Files or directories to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    parser.add_argument(
        "--min-severity",
        choices=["info", "medium", "high"],
        default="info",
        help="Only show findings at or above this severity (info < medium < high).",
    )
    args = parser.parse_args(argv)

    order = {"info": 0, "medium": 1, "high": 2}
    threshold = order[args.min_severity]

    all_findings: list[Finding] = []
    for path in _iter_python_files(args.paths):
        all_findings.extend(scan_file(path))

    all_findings = [f for f in all_findings if order[f.severity] >= threshold]
    all_findings.sort(key=lambda f: (f.file, f.line))

    if args.json:
        print(json.dumps([asdict(f) for f in all_findings], ensure_ascii=False, indent=2))
        return 0

    if not all_findings:
        print("No candidates found (or all filtered by --min-severity).")
    else:
        by_severity: dict[str, int] = {}
        for finding in all_findings:
            by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
        print(f"{len(all_findings)} static candidate(s): " + ", ".join(
            f"{count} {sev}" for sev, count in sorted(by_severity.items(), key=lambda kv: -order[kv[0]])
        ))
        print()
        for finding in all_findings:
            print(f"{finding.file}:{finding.line}:{finding.col}  [{finding.severity}] {finding.pattern}")
            if finding.snippet:
                print(f"    {finding.snippet}")
            print(f"    note: {finding.note}")
            print()

    print(
        "These are STATIC candidates only. They do not prove the code path runs, "
        "or that it actually synchronizes at runtime. Confirm with a profiler "
        "trace before changing anything (see references/profiling.md).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
