#!/usr/bin/env python3
"""Find host-device synchronization evidence in an already-captured trace.

This script only PARSES an existing Chrome-trace JSON file produced by
``torch.profiler`` (``prof.export_chrome_trace(path)``). It does not run
your model, does not attach a profiler, and does not capture anything. If
you don't have a trace yet, capture one on the target device first; this
tool only helps you read it.

It matches event names against known CUDA synchronization/copy patterns and
their likely ROCm/HIP counterparts. The HIP/DTK event naming has NOT been
verified against a real DTK trace in this skill -- use --list-names on your
own trace first to see what names actually appear, and adjust the pattern
list if needed.

Usage:
    python3 analyze_trace_sync.py trace.json
    python3 analyze_trace_sync.py trace.json.gz --list-names
    python3 analyze_trace_sync.py trace.json --json

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field


# (label, compiled regex). Matched against the raw event "name" field.
# CUDA and HIP/ROCm variants are listed side by side; HIP names are a
# best-effort guess based on ROCm's historical naming and are UNVERIFIED
# against a real DTK trace.
_SYNC_PATTERNS = [
    ("explicit_synchronize", re.compile(r"(cuda|hip)(Device|Stream|Event)?Synchronize", re.IGNORECASE)),
    ("memcpy_d2h_or_h2d", re.compile(r"Memcpy\s*(DtoH|HtoD|D2H|H2D)", re.IGNORECASE)),
    ("memcpy_async_api", re.compile(r"(cuda|hip)MemcpyAsync", re.IGNORECASE)),
    ("item_or_local_scalar", re.compile(r"aten::(item|_local_scalar_dense)")),
    ("nonzero_or_masked_select", re.compile(r"aten::(nonzero|masked_select)")),
    ("index_ops", re.compile(r"aten::(index|index_select|index_put)")),
]


@dataclass
class Aggregate:
    label: str
    count: int = 0
    total_dur_us: float = 0.0
    sample_names: set[str] = field(default_factory=set)
    sample_stacks: list[str] = field(default_factory=list)


def _load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    elif isinstance(data, list):
        events = data
    else:
        events = []
    return [e for e in events if isinstance(e, dict)]


def _extract_stack(event: dict) -> str | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    for key in ("Call stack", "stack", "Python stack", "python_stack"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def analyze(events: list[dict]) -> dict[str, Aggregate]:
    aggregates: dict[str, Aggregate] = {}
    for event in events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        if event.get("ph") not in (None, "X"):
            # "X" = complete event with duration; some traces omit ph.
            continue
        for label, pattern in _SYNC_PATTERNS:
            if pattern.search(name):
                agg = aggregates.setdefault(label, Aggregate(label=label))
                agg.count += 1
                dur = event.get("dur")
                if isinstance(dur, (int, float)):
                    agg.total_dur_us += float(dur)
                agg.sample_names.add(name)
                stack = _extract_stack(event)
                if stack and len(agg.sample_stacks) < 5:
                    agg.sample_stacks.append(stack)
                break
    return aggregates


def list_all_names(events: list[dict]) -> list[tuple[str, int, float]]:
    totals: dict[str, list[float]] = defaultdict(list)
    for event in events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        dur = event.get("dur")
        totals[name].append(float(dur) if isinstance(dur, (int, float)) else 0.0)
    rows = [(name, len(durs), sum(durs)) for name, durs in totals.items()]
    rows.sort(key=lambda row: -row[2])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", help="Path to a Chrome-trace JSON file (optionally .gz).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report.")
    parser.add_argument(
        "--list-names",
        action="store_true",
        help="Dump every distinct event name with count and total duration, sorted by duration. "
        "Use this first on a real DTK/HIP trace to see actual event names before trusting the pattern matches below.",
    )
    args = parser.parse_args(argv)

    try:
        events = _load_events(args.trace)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read trace: {exc}", file=sys.stderr)
        return 1

    if args.list_names:
        rows = list_all_names(events)
        if args.json:
            print(json.dumps(
                [{"name": n, "count": c, "total_dur_us": d} for n, c, d in rows],
                ensure_ascii=False, indent=2,
            ))
        else:
            print(f"{len(rows)} distinct event name(s), sorted by total duration (us):")
            for name, count, dur in rows[:200]:
                print(f"  {dur:>12.1f} us  x{count:<6}  {name}")
        return 0

    aggregates = analyze(events)
    if not aggregates:
        print(
            "No events matched the known CUDA/HIP sync patterns. This may mean there are "
            "none in this trace, or that this trace uses different event names (common on "
            "HIP/DTK). Run with --list-names to see what names actually appear.",
            file=sys.stderr,
        )
        return 0

    ordered = sorted(aggregates.values(), key=lambda a: -a.total_dur_us)
    if args.json:
        print(json.dumps(
            [
                {
                    "label": a.label,
                    "count": a.count,
                    "total_dur_us": a.total_dur_us,
                    "sample_names": sorted(a.sample_names),
                    "sample_stacks": a.sample_stacks,
                }
                for a in ordered
            ],
            ensure_ascii=False, indent=2,
        ))
    else:
        print("Sync-related events found (sorted by total duration):\n")
        for agg in ordered:
            print(f"[{agg.label}] count={agg.count} total_dur={agg.total_dur_us:.1f}us")
            print(f"    matched names: {', '.join(sorted(agg.sample_names))}")
            for stack in agg.sample_stacks:
                first_line = stack.splitlines()[0] if stack else ""
                print(f"    stack (first line): {first_line}")
            print()
        print(
            "This only matches event NAMES against a pattern list that is verified for "
            "CUDA and UNVERIFIED for HIP/DTK. Run --list-names on this trace to confirm the "
            "patterns actually apply; if nothing matched above but you expect sync, the "
            "event names on this backend likely differ.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
