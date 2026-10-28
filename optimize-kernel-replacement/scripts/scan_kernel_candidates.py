#!/usr/bin/env python3
"""Static scanner for kernel-replacement CANDIDATES in PyTorch code.

It lists places where a faster or fused implementation might be worth
evaluating. It does NOT prove the code is slow, that a replacement is
available on your platform, or that a replacement would be numerically
equivalent. Every finding is a candidate to confirm with a profiler trace.

Detected patterns (all name-based, so false positives are expected):
  adamw_not_fused     torch.optim.AdamW(...) / AdamW(...) without fused=/foreach=
  sdpa_call           F.scaled_dot_product_attention(...) (check backend choice)
  manual_attention    softmax(q @ k.T ...) style attention written by hand
  complex_rope        view_as_complex / view_as_real (complex RoPE)
  layernorm_module    nn.LayerNorm / nn.RMSNorm construction (vendor kernel candidates)
  norm_then_add       a norm call whose result is added to another value in one expression
  custom_autograd     torch.autograd.Function subclass (check torch.compile traceability)
  custom_op_missing_fake   torch.library.custom_op without register_fake in the same file

Usage:
    python3 scan_kernel_candidates.py <file-or-dir> [...]
    python3 scan_kernel_candidates.py --json <file-or-dir> [...]

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable

_NOTES = {
    "adamw_not_fused": "AdamW constructed without fused= or foreach=; evaluate fused=True (check rounding and checkpoint format).",
    "sdpa_call": "scaled_dot_product_attention call; confirm which backend is selected on the target platform and that gradients are finite.",
    "manual_attention": "Attention written by hand (softmax of a matmul); a fused attention backend may apply if masks and dtype allow.",
    "complex_rope": "Complex-number RoPE; a real-valued formulation or fused kernel may be evaluated. Keep the original precision contract.",
    "layernorm_module": "Norm module construction; a vendor fused kernel may exist for the target platform. Confirm a fallback path before replacing.",
    "norm_then_add": "A norm call is combined with an addition in one expression; a fused add+norm kernel is a candidate only if the profiler shows this region is hot.",
    "custom_autograd": "Custom autograd.Function; check torch.compile traceability and whether a torch.library.custom_op registration is needed.",
    "custom_op_missing_fake": "torch.library.custom_op used but no register_fake found in this file; torch.compile may need a fake implementation.",
}


@dataclass
class Finding:
    file: str
    line: int
    pattern: str
    snippet: str
    note: str


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: list[str]):
        self.path = path
        self.lines = lines
        self.findings: list[Finding] = []
        self.uses_custom_op_line: int | None = None
        self.has_register_fake = False

    def _emit(self, node: ast.AST, pattern: str) -> None:
        line = getattr(node, "lineno", 0)
        snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(self.path, line, pattern, snippet, _NOTES[pattern]))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            if _dotted(base).endswith("autograd.Function") or _dotted(base) == "Function":
                self._emit(node, "custom_autograd")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        tail = name.rsplit(".", 1)[-1]
        keywords = {kw.arg for kw in node.keywords if kw.arg}
        if tail == "AdamW" and not ({"fused", "foreach"} & keywords):
            self._emit(node, "adamw_not_fused")
        elif tail == "scaled_dot_product_attention":
            self._emit(node, "sdpa_call")
        elif tail in ("view_as_complex", "view_as_real"):
            self._emit(node, "complex_rope")
        elif tail in ("LayerNorm", "RMSNorm") and ("nn" in name or name == tail):
            self._emit(node, "layernorm_module")
        elif name.endswith("library.custom_op") or name.endswith("custom_op"):
            self.uses_custom_op_line = self.uses_custom_op_line or node.lineno
        elif tail == "register_fake" or name.endswith("register_fake"):
            self.has_register_fake = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "register_fake":
            self.has_register_fake = True
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            for side in (node.left, node.right):
                if isinstance(side, ast.Call) and "norm" in _dotted(side.func).lower():
                    self._emit(node, "norm_then_add")
                    break
        self.generic_visit(node)


def _scan_manual_attention(tree: ast.AST, path: str, lines: list[str]) -> list[Finding]:
    found: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted(node.func).rsplit(".", 1)[-1] == "softmax":
            if any(isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.MatMult) for sub in ast.walk(node)):
                line = node.lineno
                snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
                found.append(Finding(path, line, "manual_attention", snippet, _NOTES["manual_attention"]))
    return found


def scan_file(path: str) -> list[Finding]:
    try:
        source = open(path, "r", encoding="utf-8", errors="replace").read()
        tree = ast.parse(source, filename=path)
    except (OSError, SyntaxError) as exc:
        print(f"warning: skipped {path}: {exc}", file=sys.stderr)
        return []
    lines = source.splitlines()
    visitor = _Visitor(path, lines)
    visitor.visit(tree)
    findings = visitor.findings + _scan_manual_attention(tree, path, lines)
    if visitor.uses_custom_op_line and not visitor.has_register_fake:
        line = visitor.uses_custom_op_line
        snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        findings.append(Finding(path, line, "custom_op_missing_fake", snippet, _NOTES["custom_op_missing_fake"]))
    return findings


def _iter_files(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            yield path
        elif os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for name in names:
                    if name.endswith(".py"):
                        yield os.path.join(root, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    for path in _iter_files(args.paths):
        findings.extend(scan_file(path))
    findings.sort(key=lambda f: (f.file, f.line, f.pattern))

    if args.json:
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2))
        return 0
    if not findings:
        print("No candidates found.")
    else:
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.pattern] = counts.get(f.pattern, 0) + 1
        print(f"{len(findings)} static candidate(s): " + ", ".join(f"{n} {p}" for p, n in sorted(counts.items())))
        print()
        for f in findings:
            print(f"{f.file}:{f.line}  [{f.pattern}]")
            print(f"    {f.snippet}")
            print(f"    note: {f.note}\n")
    print(
        "STATIC candidates only: they do not show the code is slow, that a replacement "
        "exists for your platform, or that it is numerically equivalent. Confirm with a profiler first.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
