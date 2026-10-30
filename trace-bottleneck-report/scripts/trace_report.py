#!/usr/bin/env python3
"""从已采集的 PyTorch Profiler trace 生成瓶颈统计表与排行(只统计,不实施优化)。

只用 Python 标准库。输入是已有的 trace_rank_*.json(可 .gz);可选旁证数据:运行记录、
编译日志、key_averages 文本、已有 HTA 输出。输出 report.md 与 summary.json。

用法:
    python3 trace_report.py --trace-dir <trace 目录> -o <新输出目录> [选项]
指标口径见 references/metrics.md。时间单位:trace 的 ts/dur 为 µs,内部用 µs,输出用 ms。
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA_VERSION = 1
HERE = Path(__file__).resolve().parent
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATS = {"cpu_op", "cuda_runtime", "user_annotation"}
STEP_RE = re.compile(r"^ProfilerStep#(\d+)")
FRAME_RE = re.compile(r"^(.*?)\((\d+)\): (.*)$")
IDLE_CLASSES = ["头部", "尾部", "搬运等待", "同步等待", "通信等待", "launch 小间隙", "未归因"]


# --------------------------------------------------------------------------- 区间工具
def merge(intervals):
    ordered = sorted((s, e) for s, e in intervals if e > s)
    out = []
    for s, e in ordered:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


class Cover:
    """合并后的区间集合,支持快速的重叠长度查询。"""

    def __init__(self, intervals):
        self.iv = merge(intervals)
        self.starts = [s for s, _ in self.iv]
        self.ends = [e for _, e in self.iv]
        self.cum = [0.0]
        for s, e in self.iv:
            self.cum.append(self.cum[-1] + e - s)
        self.total = self.cum[-1]

    def overlap(self, s, e):
        if e <= s or not self.iv:
            return 0.0
        i = bisect.bisect_right(self.ends, s)
        j = bisect.bisect_left(self.starts, e)
        if i >= j:
            return 0.0
        total = self.cum[j] - self.cum[i]
        total -= max(0.0, s - self.starts[i])
        total -= max(0.0, self.ends[j - 1] - e)
        return total

    def contains(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and self.ends[i] > t


def intersect_len(a: Cover, b: Cover):
    i = j = 0
    total = 0.0
    while i < len(a.iv) and j < len(b.iv):
        lo = max(a.iv[i][0], b.iv[j][0])
        hi = min(a.iv[i][1], b.iv[j][1])
        if hi > lo:
            total += hi - lo
        if a.iv[i][1] < b.iv[j][1]:
            i += 1
        else:
            j += 1
    return total


# --------------------------------------------------------------------------- 规则
def load_rules(path):
    rules = json.loads((HERE / "default_rules.json").read_text(encoding="utf-8"))
    if path:
        extra = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in extra.items():
            if key == "kernel_categories":
                rules[key] = list(value) + rules[key]
            elif not key.startswith("_"):
                rules[key] = value
    for cat in rules["kernel_categories"]:
        cat["_re"] = [re.compile(p, re.IGNORECASE) for p in cat["patterns"]]
    rules["_sync_group"] = {n: g for g, names in rules["sync_events"].items() for n in names}
    rules["_coll_re"] = re.compile(rules["collective_kernel_regex"])
    return rules


class Classifier:
    def __init__(self, rules):
        self.rules = rules
        self.cache = {}

    def kernel(self, name):
        hit = self.cache.get(name)
        if hit is None:
            hit = (self.rules["unclassified_name"], "compute")
            for cat in self.rules["kernel_categories"]:
                if any(r.search(name) for r in cat["_re"]):
                    hit = (cat["name"], cat["kind"])
                    break
            self.cache[name] = hit
        return hit


def collective_of_kernel(name, rules):
    m = rules["_coll_re"].search(name)
    return m.group(1) if m else "其他"


def collective_of_annotation(args, rules):
    text = str(args.get("Collective name", "")).lower().replace("-", "_")
    for kind, keys in rules["collective_annotation_names"].items():
        if any(k in text for k in keys):
            return kind
    return None


# --------------------------------------------------------------------------- 读取与窗口
def read_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def pick_step(events, want):
    steps = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "user_annotation":
            m = STEP_RE.match(e.get("name", ""))
            if m:
                n = int(m.group(1))
                if n not in steps or e["dur"] > steps[n]["dur"]:
                    steps[n] = e
    if not steps:
        return None, None, []
    nums = sorted(steps)
    if want is not None:
        if want not in steps:
            raise ValueError(f"trace 中没有 ProfilerStep#{want}(可用:{nums})")
        chosen = want
    else:
        chosen = nums[len(nums) // 2]
    e = steps[chosen]
    return chosen, (e["ts"], e["ts"] + e["dur"]), nums


def enclosing_in_list(events_sorted, query_pos):
    """events_sorted 按 (ts, -dur) 排序;返回 {pos: [外层事件(由外向内)]}。"""
    wanted = set(query_pos)
    result = {}
    stack = []
    for pos, ev in enumerate(events_sorted):
        t = ev["ts"]
        while stack and stack[-1][0] <= t:
            stack.pop()
        if pos in wanted:
            end = ev["ts"] + ev["dur"]
            result[pos] = [events_sorted[p] for (_, p) in stack if events_sorted[p]["ts"] + events_sorted[p]["dur"] >= end]
        stack.append((ev["ts"] + ev["dur"], pos))
    return result


def python_frames_for(py_sorted, queries):
    """queries: [(key, ts, end)] 按 ts 升序;返回 {key: [外层 python 帧(由外向内)]}。"""
    result = {}
    stack = []
    i = 0
    for key, qs, qe in queries:
        while i < len(py_sorted) and py_sorted[i]["ts"] <= qs:
            ev = py_sorted[i]
            i += 1
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= ev["ts"]:
                stack.pop()
            stack.append(ev)
        while stack and stack[-1]["ts"] + stack[-1]["dur"] < qs:
            stack.pop()
        result[key] = [f for f in stack if f["ts"] + f["dur"] >= qe]
    return result


def is_library_frame(file_name, prefixes, hints):
    if not file_name or file_name.startswith("<"):
        return True
    if hints:
        return not any(h in file_name for h in hints)
    return any(p in file_name for p in prefixes)


def project_site(frames_names, rules, hints):
    """从由外向内的帧名里取最内层非库帧;返回 'file:line (func)' 或 None。"""
    for name in reversed(frames_names):
        m = FRAME_RE.match(name)
        if m and not is_library_frame(m.group(1), rules["library_prefixes"], hints):
            return f"{m.group(1)}:{m.group(2)} ({m.group(3)})"
    return None


def site_from_call_stack_arg(text, rules, hints):
    """旧格式:args['Call stack'] 为 ';' 分隔,按"最内层在前"解析(尽力而为)。"""
    for part in str(text).split(";"):
        m = FRAME_RE.match(part.strip())
        if m and not is_library_frame(m.group(1), rules["library_prefixes"], hints):
            return f"{m.group(1)}:{m.group(2)} ({m.group(3)})"
    return None


# --------------------------------------------------------------------------- 单 rank 分析
def analyze_rank(path, rank, rules, P):
    data = read_trace(path)
    events = data["traceEvents"]
    step_no, window, step_nums = pick_step(events, P["step"])
    warnings = []
    gpu_all = [e for e in events if e.get("ph") == "X" and e.get("cat") in GPU_CATS]
    if not gpu_all:
        raise ValueError(f"{path}: 没有 kernel/memcpy/memset 设备事件")
    if window is None:
        a = min(e["ts"] for e in gpu_all)
        b = max(e["ts"] + e["dur"] for e in gpu_all)
        warnings.append(f"rank {rank}: 没有 ProfilerStep 注解,窗口取全部设备事件范围")
        step_label = "整个 trace"
    else:
        a, b = window
        step_label = f"ProfilerStep#{step_no}"
    gpu, cpu, py = [], [], []
    for e in events:
        if e.get("ph") != "X":
            continue
        ts, dur = e["ts"], e.get("dur", 0)
        if ts + dur <= a or ts >= b:
            continue
        cat = e.get("cat")
        if cat in GPU_CATS:
            gpu.append(e)
        elif cat in CPU_CATS:
            cpu.append(e)
        elif cat == "python_function":
            py.append(e)
    del data, events, gpu_all
    clf = Classifier(rules)
    step_len = b - a

    # ---- 设备事件、类别、忙时
    G = []
    for e in gpu:
        s, t = max(e["ts"], a), min(e["ts"] + e["dur"], b)
        if t <= s:
            continue
        if e["cat"] == "kernel":
            cname, kind = clf.kernel(e["name"])
        else:
            cname, kind = rules["transfer_category"], "transfer"
        G.append((s, t, cname, kind, e))
    U = Cover([(s, t) for s, t, *_ in G])
    cumulative = sum(t - s for s, t, *_ in G)
    cats = collections.OrderedDict()
    for s, t, cname, kind, e in G:
        c = cats.setdefault(cname, {"kind": kind, "iv": [], "ms_us": 0.0, "n": 0, "k": collections.Counter()})
        c["iv"].append((s, t))
        c["ms_us"] += t - s
        c["n"] += 1
        c["k"][e["name"][:100]] += t - s
    compute_total = sum(c["ms_us"] for c in cats.values() if c["kind"] == "compute")
    t1 = []
    for cname, c in sorted(cats.items(), key=lambda kv: -kv[1]["ms_us"]):
        cov = Cover(c["iv"])
        t1.append({
            "category": cname, "kind": c["kind"], "ms": c["ms_us"] / 1000, "n": c["n"],
            "avg_us": c["ms_us"] / c["n"],
            "pct_cumulative": 100 * c["ms_us"] / cumulative if cumulative else None,
            "pct_compute": (100 * c["ms_us"] / compute_total if c["kind"] == "compute" and compute_total else None),
            "pct_step": 100 * cov.total / step_len,
            "pct_busy": 100 * cov.total / U.total if U.total else None,
            "top_kernels": [{"name": n, "ms": v / 1000} for n, v in c["k"].most_common(3)],
        })
    unclassified = next((r for r in t1 if r["category"] == rules["unclassified_name"]), None)
    unclassified_pct = unclassified["pct_cumulative"] if unclassified else 0.0

    # ---- 间隙
    gaps = []
    t = a
    for s, e in U.iv:
        if s > t:
            gaps.append([t, s])
        t = max(t, e)
    if t < b:
        gaps.append([t, b])
    gap_starts = [g[0] for g in gaps]
    gap_ends = [g[1] for g in gaps]

    # ---- CPU 事件排序与索引
    by_tid = collections.defaultdict(list)
    for e in cpu:
        if not e["name"].startswith("ProfilerStep"):
            by_tid[e["tid"]].append(e)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda e: (e["ts"], -e["dur"]))
    runtime_by_corr = {}
    for e in cpu:
        if e.get("cat") == "cuda_runtime":
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                runtime_by_corr[corr] = e

    # ---- 同步事件(只计最外层)
    sync_groups = rules["_sync_group"]
    candidates = sorted((e for e in cpu if e["name"] in sync_groups), key=lambda e: (e["tid"], e["ts"], -e["dur"]))
    syncs = []
    cur_end = {}
    for e in candidates:
        end = e["ts"] + e["dur"]
        if e["tid"] in cur_end and e["ts"] < cur_end[e["tid"]] and end <= cur_end[e["tid"]]:
            continue
        cur_end[e["tid"]] = max(cur_end.get(e["tid"], 0), end)
        syncs.append(e)
    sync_items = []
    for e in syncs:
        s, t = max(e["ts"], a), min(e["ts"] + e["dur"], b)
        if t <= s:
            continue
        busy_in = U.overlap(s, t)
        sync_items.append({"ev": e, "s": s, "e": t, "group": sync_groups[e["name"]], "busy": busy_in,
                           "idle_in": (t - s) - busy_in, "residual": 0.0})
    # 返回后到下一个设备事件之间的残余间隙,只记到最后一个在该间隙内结束的调用
    best = {}
    for idx, it in enumerate(sync_items):
        k = bisect.bisect_right(gap_starts, it["e"]) - 1
        if k >= 0 and gap_ends[k] > it["e"]:
            if k not in best or it["e"] > sync_items[best[k]]["e"]:
                best[k] = idx
    for k, idx in best.items():
        sync_items[idx]["residual"] = gap_ends[k] - sync_items[idx]["e"]

    # ---- 调用点
    has_stack = bool(py) or any((e.get("args") or {}).get("Call stack") for e in syncs)
    py_by_tid = collections.defaultdict(list)
    for e in py:
        py_by_tid[e["tid"]].append(e)
    for tid in py_by_tid:
        py_by_tid[tid].sort(key=lambda e: (e["ts"], -e["dur"]))
    queries_by_tid = collections.defaultdict(list)
    for idx, it in enumerate(sync_items):
        queries_by_tid[it["ev"]["tid"]].append((idx, it["ev"]["ts"], it["ev"]["ts"] + it["ev"]["dur"]))
    for tid, qs in queries_by_tid.items():
        qs.sort(key=lambda q: q[1])
        frames = python_frames_for(py_by_tid.get(tid, []), qs) if py_by_tid.get(tid) else {}
        pos_of = {}
        lst = by_tid.get(tid, [])
        id_to_pos = {id(e): i for i, e in enumerate(lst)}
        for idx, _, _ in qs:
            ev = sync_items[idx]["ev"]
            if id(ev) in id_to_pos:
                pos_of[idx] = id_to_pos[id(ev)]
        chains = enclosing_in_list(lst, pos_of.values()) if pos_of else {}
        for idx, _, _ in qs:
            it = sync_items[idx]
            site, source = None, "operator"
            names = [f["name"] for f in frames.get(idx, [])]
            if names:
                site = project_site(names, rules, P["project_hint"])
                if site:
                    source = "python"
            if not site:
                arg = (it["ev"].get("args") or {}).get("Call stack")
                if arg:
                    site = site_from_call_stack_arg(arg, rules, P["project_hint"])
                    source = "python" if site else source
            if not site:
                chain = chains.get(pos_of.get(idx), [])
                inner_first = [c["name"] for c in reversed(chain)][:3]
                site = " < ".join(inner_first) if inner_first else "(顶层调用)"
                site += " [未定位到代码行]"
            it["site"], it["site_source"] = site, source

    # ---- T3 聚合与判定
    blocking = set(rules["blocking_sync_groups"])
    agg = collections.OrderedDict()
    for it in sync_items:
        key = (it["site"], it["ev"]["name"])
        r = agg.setdefault(key, {"site": it["site"], "source": it["site_source"], "event": it["ev"]["name"],
                                 "group": it["group"], "n": 0, "cpu_us": 0.0, "idle_us": 0.0, "busy_us": 0.0})
        r["n"] += 1
        r["cpu_us"] += it["e"] - it["s"]
        r["idle_us"] += it["idle_in"] + it["residual"]
        r["busy_us"] += it["busy"]
    t3 = []
    for r in agg.values():
        ratio = 100 * r["idle_us"] / r["cpu_us"] if r["cpu_us"] else 0.0
        busy_frac = 100 * r["busy_us"] / r["cpu_us"] if r["cpu_us"] else 0.0
        pct_step_idle = 100 * r["idle_us"] / step_len
        if r["group"] in blocking and busy_frac >= 90:
            verdict = "在等真实工作(不可消除)"
        elif pct_step_idle < P["min_impact_pct"]:
            verdict = f"影响小(<{P['min_impact_pct']:g}% step)"
        elif ratio >= P["starve_pct"]:
            verdict = "GPU 饥饿(值得改)"
        elif ratio <= P["mask_pct"]:
            verdict = "被掩盖"
        else:
            verdict = "待观察"
        t3.append({"site": r["site"], "site_source": r["source"], "event": r["event"], "group": r["group"],
                   "n": r["n"], "cpu_ms": r["cpu_us"] / 1000, "gpu_idle_ms": r["idle_us"] / 1000,
                   "idle_over_cpu_pct": ratio, "gpu_busy_pct_in_call": busy_frac,
                   "pct_step_idle": pct_step_idle, "verdict": verdict})
    t3.sort(key=lambda r: -r["gpu_idle_ms"])
    devsync_us = sum(it["e"] - it["s"] for it in sync_items if it["group"] in blocking)
    sync_event_count = sum(1 for it in sync_items if it["group"] not in blocking)

    # ---- 大拷贝(搬运)
    big_bytes = P["big_copy_mib"] * 1048576
    transfer_windows, big_copies = [], []
    small = collections.OrderedDict()
    for s, t, cname, kind, e in G:
        if e["cat"] != "gpu_memcpy":
            continue
        args = e.get("args") or {}
        direction = args.get("kind") or ("HtoD" if "HtoD" in e["name"] else "DtoH" if "DtoH" in e["name"] else "DtoD")
        nbytes = int(args.get("bytes") or 0)
        if direction in ("HtoD", "DtoH") and nbytes >= big_bytes:
            big_copies.append((s, t, e, direction, nbytes))
        else:
            r = small.setdefault(direction, {"n": 0, "bytes": 0, "us": 0.0})
            r["n"] += 1
            r["bytes"] += nbytes
            r["us"] += t - s
    t4_big = []
    for s, t, e, direction, nbytes in big_copies:
        args = e.get("args") or {}
        runtime = runtime_by_corr.get(args.get("correlation"))
        window_start = s
        pre_copy_us = 0.0
        outer_ts = None
        if runtime is not None:
            lst = by_tid.get(runtime["tid"], [])
            pos = next((i for i, x in enumerate(lst) if x is runtime), None)
            outer = None
            if pos is not None:
                chain = enclosing_in_list(lst, [pos]).get(pos, [])
                outer = next((c for c in chain if c["name"] == "aten::to"), None)
            window_start = runtime["ts"]
            if outer is not None:
                outer_ts = outer["ts"]
                window_start = outer["ts"]
                hips = [x for x in lst if x.get("cat") == "cuda_runtime"]
                hip_starts = [x["ts"] for x in hips]
                for x in lst:
                    if x["name"] != "aten::copy_" or x["ts"] < outer["ts"] or x["ts"] + x["dur"] > runtime["ts"] + 1:
                        continue
                    lo = bisect.bisect_left(hip_starts, x["ts"])
                    inside = lo < len(hips) and hips[lo]["ts"] < x["ts"] + x["dur"]
                    if not inside:
                        pre_copy_us = max(pre_copy_us, x["dur"])
        transfer_windows.append((max(a, window_start), t))
        k = bisect.bisect_left(gap_ends, s - 1e-6)
        gap_before = (gap_ends[k] - gap_starts[k]) if k < len(gaps) and abs(gap_ends[k] - s) < 1e-6 else 0.0
        t4_big.append({"direction": direction, "bytes": nbytes, "gpu_ms": (t - s) / 1000,
                       "pct_step": 100 * (t - s) / step_len, "gbps": nbytes / ((t - s) / 1e6) / 1e9 if t > s else None,
                       "gap_before_ms": gap_before / 1000, "gap_before_pct_step": 100 * gap_before / step_len,
                       "pre_cpu_copy_ms": pre_copy_us / 1000, "pre_cpu_copy_pct_step": 100 * pre_copy_us / step_len,
                       "start_ms": (s - a) / 1000})
    for direction in ("HtoD", "DtoH"):
        peers = [r["gbps"] for r in t4_big if r["direction"] == direction and r["gbps"]]
        top = max(peers) if len(peers) > 1 else None
        for r in t4_big:
            if r["direction"] != direction:
                continue
            hints = []
            if r["pre_cpu_copy_ms"] >= 1.0:
                hints.append(f"拷贝前有 {r['pre_cpu_copy_ms']:.1f} ms 无 hip 调用的 CPU copy_,疑似 CPU 侧 dtype 转换")
            if top and r["gbps"] and r["gbps"] < 0.5 * top:
                hints.append("带宽低于同类最大值一半,疑似 pageable 内存")
            r["hint"] = ";".join(hints) + (" [推断]" if hints else "")
    t4_small = [{"direction": d, "n": r["n"], "bytes": r["bytes"], "ms": r["us"] / 1000,
                 "pct_step": 100 * r["us"] / step_len} for d, r in small.items()]

    # ---- 通信
    comm_events = [(s, t, e) for s, t, cname, kind, e in G if kind == "comm" and e["cat"] == "kernel"]
    noncomm = Cover([(s, t) for s, t, cname, kind, e in G if kind != "comm" and e["cat"] == "kernel"])
    comm_all = Cover([(s, t) for s, t, e in comm_events])
    per_type = collections.OrderedDict()
    for s, t, e in comm_events:
        r = per_type.setdefault(collective_of_kernel(e["name"], rules),
                                {"iv": [], "us": 0.0, "n": 0, "starts": [], "bytes": []})
        r["iv"].append((s, t))
        r["us"] += t - s
        r["n"] += 1
        if e["ts"] >= a:
            r["starts"].append(e["ts"])
        kargs = e.get("args") or {}
        size = rules["dtype_bytes"].get(str(kargs.get("dtype", "")))
        if kargs.get("In msg nelems") is not None and size:
            r["bytes"].append(float(kargs["In msg nelems"]) * size)
    ann_bytes = collections.defaultdict(list)
    for e in cpu:
        if e.get("cat") == "user_annotation" and e["name"].startswith("nccl:"):
            args = e.get("args") or {}
            kind = collective_of_annotation(args, rules)
            nel = args.get("In msg nelems")
            size = rules["dtype_bytes"].get(str(args.get("dtype", "")))
            if kind and nel is not None and size:
                ann_bytes[kind].append(float(nel) * size)
    t5 = []
    for ctype, r in sorted(per_type.items(), key=lambda kv: -kv[1]["us"]):
        cov = Cover(r["iv"])
        ov = intersect_len(cov, noncomm)
        avg_bytes = (statistics.mean(r["bytes"]) if r["bytes"]
                     else statistics.mean(ann_bytes[ctype]) if ann_bytes.get(ctype) else None)
        t5.append({"collective": ctype, "n": r["n"], "total_ms": r["us"] / 1000, "union_ms": cov.total / 1000,
                   "pct_step": 100 * cov.total / step_len, "overlap_ms": ov / 1000,
                   "overlap_rate_pct": 100 * ov / cov.total if cov.total else None,
                   "exposed_ms": (cov.total - ov) / 1000, "exposed_pct_step": 100 * (cov.total - ov) / step_len,
                   "exposed_pct_comm": 100 * (cov.total - ov) / cov.total if cov.total else None,
                   "avg_bytes": avg_bytes,
                   "algo_gbps": (avg_bytes / ((r["us"] / r["n"]) / 1e6) / 1e9) if avg_bytes else None})
    comm_overlap = intersect_len(comm_all, noncomm)
    comm_summary = {"union_ms": comm_all.total / 1000, "overlap_ms": comm_overlap / 1000,
                    "exposed_ms": (comm_all.total - comm_overlap) / 1000}
    comm_starts = {k: sorted(v["starts"]) for k, v in per_type.items()}

    # ---- 空闲归因
    transfer_cover = Cover(transfer_windows)
    sync_cover = Cover([(it["s"], it["e"]) for it in sync_items])
    comm_cover = Cover([(max(a, e["ts"]), min(b, e["ts"] + e["dur"])) for e in cpu
                        if e.get("cat") == "user_annotation" and e["name"].startswith("nccl:")])
    classes = {k: {"us": 0.0, "n": 0} for k in IDLE_CLASSES}
    head_transfer_us = 0.0
    big_gaps = []
    for gs, ge in gaps:
        length = ge - gs
        if length / 1000 >= P["big_gap_ms"]:
            big_gaps.append({"at_ms": (gs - a) / 1000, "ms": length / 1000})
        if gs == a:
            cls = "头部"
            head_transfer_us += transfer_cover.overlap(gs, ge)
        elif ge == b:
            cls = "尾部"
        else:
            cls = None
            for name, cover in (("搬运等待", transfer_cover), ("同步等待", sync_cover), ("通信等待", comm_cover)):
                if cover.contains(gs) or cover.overlap(gs, ge) >= 0.5 * length:
                    cls = name
                    break
            if cls is None:
                cls = "launch 小间隙" if length < P["small_gap_us"] else "未归因"
        classes[cls]["us"] += length
        classes[cls]["n"] += 1
    idle_us = step_len - U.total
    t2 = [{"class": k, "ms": classes[k]["us"] / 1000, "pct_step": 100 * classes[k]["us"] / step_len,
           "pct_idle": 100 * classes[k]["us"] / idle_us if idle_us > 0 else None, "n": classes[k]["n"]}
          for k in IDLE_CLASSES]
    return {
        "rank": rank, "file": str(path), "step_label": step_label, "step_number": step_no, "step_numbers": step_nums,
        "window_start_us": a, "step_ms": step_len / 1000, "busy_ms": U.total / 1000, "idle_ms": idle_us / 1000,
        "cumulative_ms": cumulative / 1000, "has_stack": has_stack, "warnings": warnings,
        "nccl_ms": comm_summary["union_ms"], "exposed_comm_ms": comm_summary["exposed_ms"],
        "devsync_ms": devsync_us / 1000, "sync_event_count": sync_event_count, "big_gaps": big_gaps,
        "T1": t1, "unclassified_pct": unclassified_pct, "T2": t2, "head_transfer_ms": head_transfer_us / 1000,
        "T3": t3, "T4_big": t4_big, "T4_small": t4_small, "T5": t5, "comm_summary": comm_summary,
        "comm_starts_us": comm_starts,
    }


# --------------------------------------------------------------------------- 排行
def build_ranking(ranks, detail_rank):
    d = ranks[detail_rank]
    step = d["step_ms"]
    t2 = {r["class"]: r for r in d["T2"]}
    starve = sum(r["gpu_idle_ms"] for r in d["T3"] if r["verdict"].startswith("GPU 饥饿"))
    idles = [r["idle_ms"] for r in ranks.values()]
    slow = max(ranks.values(), key=lambda r: r["idle_ms"])
    cand = [
        ("头部搬运空档", t2["头部"]["ms"] + t2["搬运等待"]["ms"], "§3 §5", "[实测]", "数据搬运"),
        ("同步导致 GPU 饥饿", starve, "§3 §4", "[实测]", "同步消除"),
        ("暴露通信", d["exposed_comm_ms"], "§6", "[实测]", "通信重叠"),
        ("launch 小间隙", t2["launch 小间隙"]["ms"], "§3", "[上限]", "launch 开销/融合/编译"),
        (f"慢 rank 等待(rank {slow['rank']})", max(0.0, max(idles) - statistics.median(idles)), "§1 §3", "[实测]", "慢 rank 排查"),
    ]
    cand = [c for c in cand if c[1] > 0]
    cand.sort(key=lambda c: -c[1])
    rows = [{"rank_no": i + 1, "name": n, "ms": ms, "pct_step": 100 * ms / step, "evidence": ev, "label": lab,
             "direction": dr} for i, (n, ms, ev, lab, dr) in enumerate(cand[:5])]
    compute = [r for r in d["T1"] if r["kind"] == "compute"]
    hot = None
    if compute:
        top = max(compute, key=lambda r: r["ms"])
        hot = {"category": top["category"], "pct_compute": top["pct_compute"], "pct_step": top["pct_step"],
               "ms": top["ms"], "top_kernel": top["top_kernels"][0] if top["top_kernels"] else None}
    return rows, hot


# --------------------------------------------------------------------------- 旁证
def read_run_record(run_dir, trace_steps_ms):
    out = {"dir": str(run_dir)}
    cmd = Path(run_dir) / "command.json"
    if cmd.exists():
        try:
            data = json.loads(cmd.read_text(encoding="utf-8"))
            command = data.get("command") or []
            env = data.get("environment") or {}
            joined = [str(x) for x in command]

            def arg(*names):
                for i, tok in enumerate(joined):
                    for n in names:
                        if tok == n and i + 1 < len(joined):
                            return joined[i + 1]
                        if tok.startswith(n + "="):
                            return tok.split("=", 1)[1]
                return None

            out["global_batch_size"] = arg("--global_batch_size", "--global-batch-size") or env.get("GLOBAL_BATCH_SIZE")
            out["num_gpus"] = arg("--num_gpus", "--num-gpus", "--nproc_per_node") or env.get("NUM_GPUS")
            out["with_stack_flag"] = ("--profiler-with-stack" in joined) and ("--no-profiler-with-stack" not in joined)
            out["profiler_flags"] = [t for t in joined if t.startswith("--profiler") or t == "--enable-profiling"]
            keep = ("GR00T_", "HIP_", "NCCL_", "RCCL_", "OMP_", "TORCH_", "PYTORCH_", "CUDA_", "GPU_")
            out["env"] = {k: v for k, v in env.items() if k.startswith(keep) or k in
                          ("GLOBAL_BATCH_SIZE", "NUM_GPUS", "DATALOADER_NUM_WORKERS", "MAX_STEPS")}
        except Exception as exc:  # 旁证读取失败不影响主流程
            out["command_error"] = str(exc)
    sm_dir = Path(run_dir) / "step_metrics"
    if sm_dir.is_dir():
        medians, tails, peaks = [], [], []
        for f in sorted(sm_dir.glob("rank-*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                secs = [float(x) for x in d.get("step_seconds", [])]
                if len(secs) > 10:
                    medians.append(statistics.median(secs[10:]))
                if len(secs) >= 3:
                    tails.append(statistics.mean(secs[-3:]))
                if d.get("max_memory_allocated_bytes") is not None:
                    peaks.append(float(d["max_memory_allocated_bytes"]))
            except Exception:
                continue
        if medians:
            out["step_ms_median_after_10"] = statistics.median(medians) * 1000
        if tails:
            out["step_ms_last3_mean"] = statistics.median(tails) * 1000
        if peaks:
            out["peak_alloc_gib"] = max(peaks) / 2 ** 30
        out["trace_step_ms"] = trace_steps_ms
    return out


def read_compile_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    breaks = [l for l in text if re.search(r"graph break", l, re.IGNORECASE)]
    recompiles = [l for l in text if re.search(r"recompil", l, re.IGNORECASE)]
    reasons = collections.Counter()
    for i, l in enumerate(text):
        if re.search(r"guard failure|triggered by", l, re.IGNORECASE):
            for nxt in text[i + 1:i + 4]:
                nxt = nxt.strip()
                if nxt.startswith("-") or nxt.startswith("*"):
                    reasons[re.sub(r"\d+", "N", nxt)[:140]] += 1
    break_reasons = collections.Counter(re.sub(r"\d+", "N", l.strip())[:140] for l in breaks)
    return {"file": str(path), "graph_break_lines": len(breaks), "recompile_lines": len(recompiles),
            "top_break_reasons": break_reasons.most_common(3), "top_recompile_reasons": reasons.most_common(3),
            "note": "按关键字统计,日志格式随 PyTorch 版本变化" if breaks or recompiles else "未识别到图断/重编译关键字"}


def parse_time_ms(text):
    m = re.match(r"^\s*([\d.,]+)\s*(ns|us|µs|ms|s)\s*$", text)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1000.0}[m.group(2)]


def read_key_averages(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header_i = next((i for i, l in enumerate(lines) if "Name" in l and "Self CUDA" in l), None)
    if header_i is None:
        return {"file": str(path), "error": "未找到包含 Name 与 Self CUDA 的表头"}
    cols = [c.strip() for c in re.split(r"\s{2,}", lines[header_i].strip())]
    try:
        idx = cols.index("Self CUDA")
    except ValueError:
        return {"file": str(path), "error": "表头中没有 'Self CUDA' 列"}
    rows = []
    for l in lines[header_i + 2:]:
        if not l.strip() or set(l.strip()) <= {"-", " "}:
            if rows:
                break
            continue
        parts = [p.strip() for p in re.split(r"\s{2,}", l.strip())]
        if len(parts) <= idx:
            continue
        ms = parse_time_ms(parts[idx])
        if ms is not None:
            rows.append({"name": parts[0][:100], "self_cuda_ms": ms})
    rows.sort(key=lambda r: -r["self_cuda_ms"])
    return {"file": str(path), "top": rows[:5]}


def read_hta(hta_dir):
    out = []
    for pattern in ("temporal-*.csv", "idle-*.csv", "overlap-*.csv"):
        for f in sorted(Path(hta_dir).glob(pattern))[:2]:
            try:
                with open(f, newline="", encoding="utf-8") as fh:
                    rows = list(csv.reader(fh))
                out.append({"file": f.name, "header": rows[0][:8] if rows else [],
                            "rows": [r[:8] for r in rows[1:3]]})
            except Exception as exc:
                out.append({"file": f.name, "error": str(exc)})
    return out


# --------------------------------------------------------------------------- 渲染
def f2(x):
    return "—" if x is None else f"{x:.2f}"


def pct(x):
    if x is None:
        return "—"
    return "<0.1%" if 0 < x < 0.1 else f"{x:.1f}%"


def table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|"
    lines = ["| " + " | ".join(headers) + " |", sep]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def ms_pct(ms, p):
    return f"{ms:.2f} ({pct(p)})"


def render(summary, P):
    ranks = summary["ranks"]
    dr = summary["detail_rank"]
    d = ranks[dr]
    out = []
    name = summary["name"]
    stack = "有调用栈" if d["has_stack"] else "无调用栈"
    out.append(f"# {name} Trace 瓶颈速览\n")
    out.append(f"口径:{len(ranks)} 个 rank;每个 rank 取一个中间 step({d['step_label']});{stack};"
               f"单次采集。时间窗口为 CPU 侧 ProfilerStep,设备事件按窗口裁剪。明细表取 rank {dr}。\n")
    for w in summary["warnings"]:
        out.append(f"> 提示:{w}\n")
    out.append("## 0. 瓶颈排行(前 5,按可回收上限)\n")
    if summary["ranking"]:
        out.append(table(["#", "瓶颈", "上限 ms/step", "占 step%", "证据", "标签", "方向类别"],
                         [[r["rank_no"], r["name"], f2(r["ms"]), pct(r["pct_step"]), r["evidence"], r["label"],
                           r["direction"]] for r in summary["ranking"]], "lrrrlll"))
    else:
        out.append("没有发现可回收的非计算时间。")
    out.append("\n各上限可能重叠同一段空闲,不能相加。\"方向类别\"只指引去向,对应 references/handoff.md。\n")
    hot = summary["compute_hotspot"]
    if hot:
        tk = hot["top_kernel"]
        extra = f";最大 kernel:{tk['name']}({tk['ms']:.2f} ms)" if tk else ""
        out.append(f"计算热点(不进排行):{hot['category']} 占计算 {pct(hot['pct_compute'])}、占 step {pct(hot['pct_step'])}{extra}\n")

    out.append("## 1. 总览(L0)\n")
    rows = []
    slow = summary["slow_rank"]
    for rk in sorted(ranks):
        r = ranks[rk]
        sp = lambda ms: 100 * ms / r["step_ms"]
        bg = ", ".join(f"{g['at_ms']:.0f}:{g['ms']:.1f}" for g in r["big_gaps"]) or "—"
        rows.append([f"{rk}{' (最慢)' if rk == slow else ''}", f2(r["step_ms"]),
                     ms_pct(r["busy_ms"], sp(r["busy_ms"])), ms_pct(r["idle_ms"], sp(r["idle_ms"])),
                     ms_pct(r["nccl_ms"], sp(r["nccl_ms"])), ms_pct(r["exposed_comm_ms"], sp(r["exposed_comm_ms"])),
                     ms_pct(r["devsync_ms"], sp(r["devsync_ms"])), r["sync_event_count"], bg])
    out.append(table(["rank", "step ms", "GPU 忙 ms(占 step%)", "空闲 ms(占 step%)", "NCCL ms(占 step%)",
                      "暴露通信 ms(占 step%)", "设备同步 ms(占 step%)", "同步事件数", f"大间隙(位置ms:ms,≥{P['big_gap_ms']:g}ms)"], rows))
    out.append("\n占 step% 的分母为各 rank 自己的 step 窗口;\"最慢\"指空闲最多的 rank。设备同步为 CPU 调用耗时,其间 GPU 可能仍在忙。\n")

    out.append("## 2. GPU 耗时速览(T1)\n")
    rows = [[r["category"], f2(r["ms"]), pct(r["pct_cumulative"]), pct(r["pct_compute"]), pct(r["pct_step"]),
             pct(r["pct_busy"]), r["n"], f"{r['avg_us']:.1f}"] for r in d["T1"]]
    rows.append(["合计(累计)", f2(d["cumulative_ms"]), "100.0%", "—", "—", "—", sum(r["n"] for r in d["T1"]), "—"])
    out.append(table(["类别", "GPU ms", "占累计%", "占计算%", "占 step%", "占忙时%", "kernel 数", "均值 µs"], rows))
    out.append("\n占累计%:该类 ÷ 全部设备事件累计耗时(合计 100%);占计算%:仅计算类行,分母为计算类合计(不含通信与搬运,含未分类);"
               "占 step%、占忙时%:该类区间并集 ÷ step 窗口 / GPU 忙时(跨 stream 并集)。累计类百分比因多 stream 重叠可能与占 step 不同,不可跨列相加。\n")
    out.append("各类 top-3 kernel:\n")
    for r in d["T1"]:
        tops = ";".join(f"{k['name']}({k['ms']:.2f} ms)" for k in r["top_kernels"])
        out.append(f"- {r['category']}:{tops}")
    if d["unclassified_pct"] and d["unclassified_pct"] > P["unclassified_warn_pct"]:
        out.append(f"\n> 未分类占累计 {pct(d['unclassified_pct'])},超过 {P['unclassified_warn_pct']:g}%,建议用 --rules 补充规则。")
    out.append("")

    def t2_block(r, title):
        rows = []
        for row in r["T2"]:
            rows.append([row["class"], f2(row["ms"]), pct(row["pct_step"]), pct(row["pct_idle"]), row["n"]])
            if row["class"] == "头部":
                rows.append(["　↳ 其中与搬运重叠", f2(r["head_transfer_ms"]), pct(100 * r["head_transfer_ms"] / r["step_ms"]), "—", "—"])
        rows.append(["合计空闲", f2(r["idle_ms"]), pct(100 * r["idle_ms"] / r["step_ms"]), "100.0%" if r["idle_ms"] > 0 else "—",
                     sum(x["n"] for x in r["T2"])])
        out.append(title + "\n")
        out.append(table(["类别", "ms", "占 step%", "占空闲%", "间隙数"], rows))
        out.append("")

    out.append("## 3. 空闲归因(T2)\n")
    t2_block(d, f"rank {dr}:")
    sl = ranks[slow]
    if slow != dr and sl["idle_ms"] - d["idle_ms"] > P["slow_rank_ms"]:
        t2_block(sl, f"最慢 rank {slow}(空闲比 rank {dr} 多 {sl['idle_ms'] - d['idle_ms']:.1f} ms):")
    out.append("每个间隙只归一类,优先级:头部/尾部(按位置)→搬运等待→同步等待→通信等待→launch 小间隙→未归因;"
               "窗口取 CPU 侧对应事件,间隙起点落入窗口或与窗口重叠 ≥50% 即归类。通信等待依据 CPU 侧 nccl:* 标注,属 [推断]。\n")

    out.append("## 4. 同步表(T3)\n")
    shown = d["T3"][:P["top_sites"]]
    if shown:
        rows = [[r["site"], r["event"], r["n"], f2(r["cpu_ms"]), f2(r["gpu_idle_ms"]), pct(r["idle_over_cpu_pct"]),
                 pct(r["pct_step_idle"]), r["verdict"]] for r in shown]
        rest = d["T3"][P["top_sites"]:]
        if rest:
            rows.append([f"其余 {len(rest)} 项", "—", sum(r["n"] for r in rest), f2(sum(r["cpu_ms"] for r in rest)),
                         f2(sum(r["gpu_idle_ms"] for r in rest)), "—", pct(sum(r["pct_step_idle"] for r in rest)), "—"])
        out.append(table(["调用点", "事件", "次数", "CPU ms", "调用期间 GPU 空闲 ms", "空闲/CPU%", "GPU 空闲占 step%", "判定"], rows, "llrrrrrl"))
    else:
        out.append("窗口内没有同步事件。")
    out.append(f"\n调用点:有 Python 栈时取最内层项目帧(file:line),无栈时退化为外层 CPU 算子链并标\"未定位到代码行\"。"
               f"调用期间 GPU 空闲 = 调用区间内空闲 + 返回后到下一个设备事件的残余间隙,故空闲/CPU% 可超过 100%。"
               f"判定:调用点 GPU 空闲占 step <{P['min_impact_pct']:g}% 判\"影响小\";否则 ≥{P['starve_pct']:g}% GPU 饥饿;≤{P['mask_pct']:g}% 被掩盖;"
               f"设备/流/事件同步在调用期间 GPU 忙 ≥90% 判\"在等真实工作\"。\n")

    out.append("## 5. 搬运表(T4)\n")
    if d["T4_big"]:
        peak = P["peak_h2d_gbs"]
        headers = ["方向", "MiB", "GPU ms", "占 step%", "带宽 GB/s"] + (["占峰值%"] if peak else []) + [
            "拷贝前 GPU 空档 ms(占 step%)", "前置 CPU copy_ ms(占 step%)", "提示"]
        rows = []
        for r in d["T4_big"]:
            row = [r["direction"], f"{r['bytes'] / 1048576:.1f}", f2(r["gpu_ms"]), pct(r["pct_step"]), f2(r["gbps"])]
            if peak:
                row.append(pct(100 * r["gbps"] / peak) if r["gbps"] else "—")
            row += [ms_pct(r["gap_before_ms"], r["gap_before_pct_step"]), ms_pct(r["pre_cpu_copy_ms"], r["pre_cpu_copy_pct_step"]),
                    r.get("hint") or "—"]
            rows.append(row)
        out.append(f"≥{P['big_copy_mib']:g} MiB 的 H2D/D2H 逐条:\n")
        out.append(table(headers, rows, "lrrrr" + ("r" if peak else "") + "rrl"))
    else:
        out.append(f"窗口内没有 ≥{P['big_copy_mib']:g} MiB 的 H2D/D2H 拷贝。")
    if d["T4_small"]:
        rows = [[r["direction"] + (" (设备内,不计入搬运)" if r["direction"] == "DtoD" else ""), r["n"],
                 f"{r['bytes'] / 1048576:.2f}", f2(r["ms"]), pct(r["pct_step"])] for r in d["T4_small"]]
        out.append("\n其余拷贝按方向汇总:\n")
        out.append(table(["方向", "次数", "MiB", "GPU ms", "占 step%"], rows))
    out.append("\n拷贝前 GPU 空档 = 该拷贝开始前紧邻的设备空闲;前置 CPU copy_ = 拷贝前、期间没有 hip/cuda 调用的嵌套 aten::copy_。"
               "提示列由结构规则给出,均为 [推断]。带宽占峰值需用 --peak-h2d-gbs 提供硬件峰值,否则不显示。\n")

    out.append("## 6. 通信表(T5)\n")
    if d["T5"]:
        peak_c = P["peak_comm_gbs"]
        spreads = summary["comm_spread"]
        headers = ["集合", "次数", "总 ms", "占 step%", "重叠 ms", "重叠率%", "暴露 ms", "暴露占 step%", "暴露占通信%",
                   "起始极差 ms(中位/最大)"] + (["算法带宽 GB/s", "占峰值%"] if peak_c else []) + ["平均字节"]
        rows = []
        for r in d["T5"]:
            sp = spreads.get(r["collective"])
            spread = f"{sp['median_ms']:.2f}/{sp['max_ms']:.2f}" if sp else "—"
            row = [r["collective"], r["n"], f2(r["total_ms"]), pct(r["pct_step"]), f2(r["overlap_ms"]), pct(r["overlap_rate_pct"]),
                   f2(r["exposed_ms"]), pct(r["exposed_pct_step"]), pct(r["exposed_pct_comm"]), spread]
            if peak_c:
                row += [f2(r["algo_gbps"]), pct(100 * r["algo_gbps"] / peak_c) if r["algo_gbps"] else "—"]
            row.append(f"{r['avg_bytes'] / 1048576:.2f} MiB" if r["avg_bytes"] else "—")
            rows.append(row)
        cs = d["comm_summary"]
        out.append(table(headers, rows))
        out.append(f"\n通信区间并集 {cs['union_ms']:.2f} ms,其中与计算 kernel 重叠 {cs['overlap_ms']:.2f} ms,暴露 {cs['exposed_ms']:.2f} ms。")
    else:
        out.append("窗口内没有通信 kernel。")
    out.append(f"\n暴露 = 通信区间并集中未被非通信 kernel 覆盖的部分。起始极差 = 各 rank 第 k 次同类集合起始时间的最大差({summary['clock_note']})。"
               "算法带宽 = 平均字节 ÷ 平均 kernel 耗时,未含 NCCL 总线系数。\n")

    out.append("## 7. 口径与旁证\n")
    ev = summary["evidence"]
    run = ev.get("run")
    if run:
        out.append("**运行记录**:")
        out.append(f"- 来源:{run['dir']}")
        if run.get("global_batch_size"):
            out.append(f"- 全局 batch:{run['global_batch_size']};卡数:{run.get('num_gpus', '—')}")
        if "with_stack_flag" in run:
            out.append(f"- 采集命令{'带' if run['with_stack_flag'] else '不带'} --profiler-with-stack(带栈会改变 CPU 计时)")
        if run.get("env"):
            out.append("- 关键环境:" + ", ".join(f"{k}={v}" for k, v in sorted(run["env"].items())[:12]))
        if run.get("step_ms_median_after_10") or run.get("step_ms_last3_mean"):
            out.append(f"- step_metrics 真实 step 时间:第 11 步起中位数 {f2(run.get('step_ms_median_after_10'))} ms,"
                       f"末 3 步均值 {f2(run.get('step_ms_last3_mean'))} ms;trace 所选 step {f2(d['step_ms'])} ms(含 profiler 开销,不作测速)")
        if run.get("peak_alloc_gib"):
            out.append(f"- 峰值显存分配:{run['peak_alloc_gib']:.1f} GiB")
        out.append("")
    cl = ev.get("compile_log")
    if cl:
        out.append("**编译日志**:")
        if cl.get("error"):
            out.append(f"- 读取失败:{cl['error']}")
        else:
            out.append(f"- 图断相关行 {cl['graph_break_lines']},重编译相关行 {cl['recompile_lines']}({cl['note']})")
            for text, n in cl["top_break_reasons"]:
                out.append(f"  - 图断 ×{n}:{text}")
            for text, n in cl["top_recompile_reasons"]:
                out.append(f"  - 重编译原因 ×{n}:{text}")
        out.append("")
    ka = ev.get("key_averages")
    if ka:
        out.append("**key_averages 交叉校验**(Self CUDA top5,与 T1 类别排名对照):")
        if ka.get("error"):
            out.append(f"- 无法解析:{ka['error']}")
        else:
            for r in ka["top"]:
                out.append(f"- {r['name']}:{r['self_cuda_ms']:.2f} ms")
        out.append("")
    hta = ev.get("hta")
    if hta:
        out.append("**HTA 输出(只读已有结果)**:窗口与过滤规则与本报告不同,空闲/重叠不能互换。")
        for item in hta:
            if item.get("error"):
                out.append(f"- {item['file']}:读取失败 {item['error']}")
            else:
                out.append(f"- {item['file']}:列 {item['header']};前两行 {item['rows']}")
        out.append("")
    out.append("**限制**:单次采集,只取每个 rank 的一个 step,各 rank 不独立;trace 的 step 时间含 profiler 开销;"
               "上限只是占用上限,不代表优化收益。")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- 入口
def discover(trace_dir):
    files = {}
    for f in sorted(Path(trace_dir).glob("trace_rank_*.json*")):
        m = re.search(r"trace_rank_(\d+)", f.name)
        if m:
            files[int(m.group(1))] = f
    return files


def _job(args):
    return analyze_rank(*args)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Profiler trace 瓶颈统计速览")
    ap.add_argument("--trace-dir", required=True, help="含 trace_rank_*.json 的目录")
    ap.add_argument("-o", "--output-dir", required=True, help="新输出目录(已存在则拒绝覆盖)")
    ap.add_argument("--name", default=None, help="报告标题中的名称,默认用 trace 目录名")
    ap.add_argument("--run-dir", help="含 command.json / step_metrics 的运行目录")
    ap.add_argument("--compile-log", help="TORCH_LOGS=graph_breaks,recompiles 的输出文件")
    ap.add_argument("--key-averages", help="含 key_averages 表的文本(如 train.log)")
    ap.add_argument("--hta-dir", help="已有的 HTA 输出目录")
    ap.add_argument("--step", type=int, default=None, help="指定 ProfilerStep 编号,默认取中间 step")
    ap.add_argument("--rules", help="补充/覆盖分类规则的 JSON 文件")
    ap.add_argument("--project-hint", default="", help="逗号分隔的项目路径片段,用于识别项目帧(如 gr00t/)")
    ap.add_argument("--jobs", type=int, default=1, help="并行解析的 rank 数")
    ap.add_argument("--small-gap-us", type=float, default=50)
    ap.add_argument("--big-copy-mib", type=float, default=1)
    ap.add_argument("--big-gap-ms", type=float, default=5)
    ap.add_argument("--slow-rank-ms", type=float, default=10)
    ap.add_argument("--starve-pct", type=float, default=50)
    ap.add_argument("--mask-pct", type=float, default=20)
    ap.add_argument("--min-impact-pct", type=float, default=0.5,
                    help="同步调用点的 GPU 空闲占 step 低于此值时判为影响小,不报饥饿")
    ap.add_argument("--top-sites", type=int, default=10)
    ap.add_argument("--unclassified-warn-pct", type=float, default=5)
    ap.add_argument("--peak-h2d-gbs", type=float, default=None, help="硬件 H2D 峰值,传入后才显示占峰值%%")
    ap.add_argument("--peak-comm-gbs", type=float, default=None, help="通信峰值带宽,传入后才显示占峰值%%")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        print(f"错误:输出目录已存在,拒绝覆盖:{out_dir}", file=sys.stderr)
        return 2
    files = discover(args.trace_dir)
    if not files:
        print(f"错误:{args.trace_dir} 下没有 trace_rank_*.json", file=sys.stderr)
        return 2
    P = {"step": args.step, "small_gap_us": args.small_gap_us, "big_copy_mib": args.big_copy_mib,
         "big_gap_ms": args.big_gap_ms, "slow_rank_ms": args.slow_rank_ms, "starve_pct": args.starve_pct,
         "mask_pct": args.mask_pct, "min_impact_pct": args.min_impact_pct, "top_sites": args.top_sites,
         "unclassified_warn_pct": args.unclassified_warn_pct,
         "peak_h2d_gbs": args.peak_h2d_gbs, "peak_comm_gbs": args.peak_comm_gbs,
         "project_hint": [h for h in args.project_hint.split(",") if h]}
    rules = load_rules(args.rules)
    jobs = [(path, rk, rules, P) for rk, path in sorted(files.items())]
    results = {}
    warnings = []
    if args.jobs > 1 and len(jobs) > 1:
        import multiprocessing
        with multiprocessing.Pool(min(args.jobs, len(jobs))) as pool:
            for r in pool.imap(_job, jobs):
                results[r["rank"]] = r
    else:
        for j in jobs:
            r = analyze_rank(*j)
            results[r["rank"]] = r
            print(f"rank {r['rank']}: {r['step_label']} step {r['step_ms']:.1f} ms 已分析", file=sys.stderr)
    for r in results.values():
        warnings += r["warnings"]
    detail = min(results)
    slow = max(results, key=lambda k: results[k]["idle_ms"])

    # 跨 rank:时钟校验与集合起始极差
    starts = [r["window_start_us"] for r in results.values()]
    comm_spread = {}
    if len(results) > 1:
        if max(starts) - min(starts) < 50_000:
            clock_note = "同机时钟假设,各 rank step 起点差 <50 ms"
            types = set.intersection(*[set(r["comm_starts_us"]) for r in results.values()]) if results else set()
            for ctype in types:
                lists = [r["comm_starts_us"][ctype] for r in results.values()]
                if len({len(x) for x in lists}) == 1 and lists[0]:
                    gaps = [(max(col) - min(col)) / 1000 for col in zip(*lists)]
                    comm_spread[ctype] = {"median_ms": statistics.median(gaps), "max_ms": max(gaps)}
                else:
                    warnings.append(f"{ctype}:各 rank 窗口内集合次数不一致,未计算起始极差")
        else:
            clock_note = "各 rank step 起点差 ≥50 ms,时钟不可比,未计算"
            warnings.append("各 rank step 起点差 ≥50 ms,可能不同机或时钟不可比,起始极差未计算")
    else:
        clock_note = "仅一个 rank,无法计算"

    ranking, hot = build_ranking(results, detail)
    evidence = {}
    run_dir = args.run_dir
    if not run_dir and (Path(args.trace_dir).parent / "command.json").exists():
        run_dir = str(Path(args.trace_dir).parent)
    if run_dir:
        evidence["run"] = read_run_record(run_dir, results[detail]["step_ms"])
    for key, fn, value in (("compile_log", read_compile_log, args.compile_log),
                           ("key_averages", read_key_averages, args.key_averages), ("hta", read_hta, args.hta_dir)):
        if value:
            try:
                evidence[key] = fn(value)
            except Exception as exc:
                evidence[key] = {"error": str(exc)} if key != "hta" else [{"file": value, "error": str(exc)}]
    d = results[detail]
    if d["unclassified_pct"] and d["unclassified_pct"] > args.unclassified_warn_pct:
        warnings.append(f"未分类 kernel 占累计 {d['unclassified_pct']:.1f}%,建议补充 --rules")
    summary = {
        "schema_version": SCHEMA_VERSION, "name": args.name or Path(args.trace_dir).resolve().name,
        "inputs": {"trace_dir": str(args.trace_dir), "ranks": sorted(results), "run_dir": run_dir},
        "params": {k: v for k, v in P.items()}, "detail_rank": detail, "slow_rank": slow,
        "ranks": {str(k): v for k, v in results.items()} if False else results,
        "ranking": ranking, "compute_hotspot": hot, "comm_spread": comm_spread, "clock_note": clock_note,
        "evidence": evidence, "warnings": warnings,
    }
    out_dir.mkdir(parents=True)
    report = render(summary, P)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    dump = dict(summary)
    dump["ranks"] = {str(k): v for k, v in results.items()}
    (out_dir / "summary.json").write_text(json.dumps(dump, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已生成:{out_dir / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
