#!/usr/bin/env python3
"""自测:构造可手算的合成 trace,运行 trace_report 并断言各表数字。

    python3 selftest.py

合成场景(rank 0,窗口 100 ms,单位 µs,T0 为基准):
  GPU 忙 74.95 ms,空闲 25.05 ms = 头部 10 + 尾部 2 + 同步 9.5 + 通信等待 1.5 + 小间隙 0.05 + 未归因 2
  rank 1 比 rank 0 多 2.47 ms 空闲(最慢 rank),其内部间隙含一个搬运等待。
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_report as tr  # noqa: E402

T0 = 1_000_000_000


def ev(name, cat, ts, dur, tid=1, args=None):
    e = {"ph": "X", "name": name, "cat": cat, "ts": T0 + ts, "dur": dur, "pid": 0, "tid": tid}
    if args is not None:
        e["args"] = args
    return e


def kernel(name, ts, end, stream=7, extra=None):
    return ev(name, "kernel", ts, end - ts, tid=stream, args={"stream": stream, **(extra or {})})


def build_rank(rank, with_stack=False):
    ar_start = 50000 if rank == 0 else 52000
    E = [
        ev("ProfilerStep#4", "user_annotation", -200000, 100000),
        ev("ProfilerStep#5", "user_annotation", 0, 100000),
        ev("ProfilerStep#5", "gpu_user_annotation", 0, 100000, tid=0),
        ev("ProfilerStep#6", "user_annotation", 200000, 100000),
        # 头部搬运:CPU 转换 7.5 ms(无 hip 调用)+ 一次 hip 拷贝
        ev("aten::to", "cpu_op", 500, 8500),
        ev("aten::copy_", "cpu_op", 500, 7500),
        ev("aten::copy_", "cpu_op", 8000, 1000),
        ev("hipMemcpyWithStream", "cuda_runtime", 8100, 800, args={"correlation": 1}),
        ev("Memcpy HtoD (Host -> Device)", "gpu_memcpy", 10000, 2000, tid=7,
           args={"bytes": 4194304, "kind": "HtoD", "correlation": 1, "stream": 7}),
        kernel("Cijk_Alik_Bljk_test", 12000, 32000),
        kernel("void at::native::vectorized_elementwise_kernel<4>", 34000, 42000),
        # .item() 让 GPU 饥饿
        ev("MyFunc", "cpu_op", 40000, 7000),
        ev("aten::item", "cpu_op", 41000, 5000),
        ev("aten::_local_scalar_dense", "cpu_op", 41100, 4800),
        kernel("Cijk_Alik_Bljk_test2", 47000, 57000),
        kernel("mystery_kernel_xyz", 57000, 62000),
        kernel("ncclDevKernel_AllReduce_Ring_Simple_Sum_b16(ncclDevKernelArgsStorage<1024ul>)", ar_start, 70000, stream=9,
               extra={"Collective name": "allreduce", "In msg nelems": 1000000, "dtype": "BFloat16"}),
        kernel("void at::native::multi_tensor_apply_kernel<foo>", 70000, 76000),
        ev("Optimizer.step", "cpu_op", 69000, 12000),
        ev("hipDeviceSynchronize", "cuda_runtime", 70000, 10000),
        ev("Trainer.sync", "cpu_op", 80500, 10500),
        ev("hipDeviceSynchronize", "cuda_runtime", 80500, 9500),
        kernel("ncclDevKernel_AllGather_Ring_LL_Sum_i8(ncclDevKernelArgsStorage<1024ul>)", 80500, 90000, stream=9,
               extra={"Collective name": "_allgather_base", "In msg nelems": 1000000, "dtype": "Float"}),
        kernel("vectorized_elementwise_kernel", 90000, 92000),
        kernel("vectorized_elementwise_kernel", 92020, 94000),
        ev("nccl:all_reduce", "user_annotation", 49000, 1000,
           args={"Collective name": "all_reduce", "In msg nelems": 1000000, "dtype": "BFloat16"}),
        ev("nccl:_all_gather_base", "user_annotation", 79000, 1500,
           args={"Collective name": "_allgather_base", "In msg nelems": 500000, "dtype": "Float"}),
        ev("nccl:all_reduce", "user_annotation", 95000, 3000,
           args={"Collective name": "all_reduce", "In msg nelems": 1000000, "dtype": "BFloat16"}),
    ]
    if rank == 0:
        E += [kernel("vectorized_elementwise_kernel", 94030, 96000),
              kernel("vectorized_elementwise_kernel", 97500, 98000)]
    else:  # rank 1:少两个 kernel(尾部更长),并在 32000-34000 间隙内有一次搬运
        E += [ev("aten::to", "cpu_op", 31000, 4000),
              ev("hipMemcpyWithStream", "cuda_runtime", 31500, 3000, args={"correlation": 3}),
              ev("Memcpy HtoD (Host -> Device)", "gpu_memcpy", 34000, 200, tid=8,
                 args={"bytes": 2097152, "kind": "HtoD", "correlation": 3, "stream": 8})]
    if with_stack:
        E += [ev("proj/train.py(10): train", "python_function", 0, 100000),
              ev("/usr/lib/python3/dist-packages/transformers/trainer.py(5): _prepare", "python_function", 38000, 10000),
              ev("proj/model.py(77): forward", "python_function", 40000, 7000),
              ev("<built-in method item of Tensor object at 0x1>", "python_function", 41000, 5000)]
    return {"traceEvents": E}


def write_traces(root, with_stack=False):
    root.mkdir(parents=True)
    for rank in (0, 1):
        (root / f"trace_rank_{rank}_iter_5.json").write_text(json.dumps(build_rank(rank, with_stack)))


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


def row(rows, key, value):
    return next(r for r in rows if r[key] == value)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="trace_report_selftest_"))
    plain, stack = tmp / "plain", tmp / "stack"
    write_traces(plain)
    write_traces(stack, with_stack=True)

    # ---- 主场景
    out = tmp / "out"
    rc = tr.main(["--trace-dir", str(plain), "-o", str(out), "--slow-rank-ms", "1"])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    report = (out / "report.md").read_text()
    r0, r1 = s["ranks"]["0"], s["ranks"]["1"]
    assert s["schema_version"] == 1 and s["detail_rank"] == 0 and s["slow_rank"] == 1
    assert r0["step_number"] == 5 and r0["step_numbers"] == [4, 5, 6], "应取中间 step 且忽略 GPU 注解副本"
    close(r0["step_ms"], 100.0)
    close(r0["busy_ms"], 74.95)
    close(r0["idle_ms"], 25.05)
    close(r0["cumulative_ms"], 86.95)

    # T2:各类之和等于空闲总和
    t2 = {r["class"]: r for r in r0["T2"]}
    expect = {"头部": 10.0, "尾部": 2.0, "搬运等待": 0.0, "同步等待": 9.5, "通信等待": 1.5, "launch 小间隙": 0.05, "未归因": 2.0}
    for k, v in expect.items():
        close(t2[k]["ms"], v)
    close(sum(r["ms"] for r in r0["T2"]), r0["idle_ms"])
    close(sum(r["pct_idle"] for r in r0["T2"]), 100.0, 1e-6)
    close(r0["head_transfer_ms"], 9.5)
    t2b = {r["class"]: r for r in r1["T2"]}
    close(t2b["搬运等待"]["ms"], 2.0)
    close(t2b["未归因"]["ms"], 0.0)
    close(sum(r["ms"] for r in r1["T2"]), r1["idle_ms"])

    # T1:百分比定义
    t1 = {r["category"]: r for r in r0["T1"]}
    close(t1["GEMM"]["ms"], 30.0)
    close(t1["elementwise/reduce/copy"]["ms"], 14.45)
    close(t1["未分类"]["ms"], 5.0)
    close(t1["通信"]["ms"], 29.5)
    close(sum(r["pct_cumulative"] for r in r0["T1"]), 100.0, 1e-6)
    compute_total = 30.0 + 14.45 + 5.0 + 6.0
    close(t1["GEMM"]["pct_compute"], 100 * 30.0 / compute_total, 1e-6)
    assert t1["通信"]["pct_compute"] is None and t1["搬运(memcpy/memset)"]["pct_compute"] is None
    close(sum(r["pct_compute"] for r in r0["T1"] if r["pct_compute"] is not None), 100.0, 1e-6)
    close(t1["GEMM"]["pct_step"], 30.0)
    close(t1["GEMM"]["pct_busy"], 100 * 30.0 / 74.95, 1e-6)
    assert r0["unclassified_pct"] > 5 and "未分类" in report and "--rules" in report

    # T3:三种判定与嵌套去重
    sites = {(r["site"].split(" [")[0], r["event"]): r for r in r0["T3"]}
    item = next(r for r in r0["T3"] if r["event"] == "aten::item")
    assert "MyFunc" in item["site"] and "未定位到代码行" in item["site"]
    assert item["n"] == 1 and item["verdict"].startswith("GPU 饥饿")
    close(item["cpu_ms"], 5.0)
    close(item["gpu_idle_ms"], 5.0)
    assert not any(r["event"] == "aten::_local_scalar_dense" for r in r0["T3"])
    dev1 = next(r for r in r0["T3"] if r["site"].startswith("Optimizer.step"))
    close(dev1["gpu_idle_ms"], 4.5)
    assert dev1["verdict"] == "待观察", dev1["verdict"]
    dev2 = next(r for r in r0["T3"] if r["site"].startswith("Trainer.sync"))
    close(dev2["gpu_idle_ms"], 0.0)
    assert dev2["verdict"].startswith("在等真实工作")
    assert r0["sync_event_count"] == 1
    close(r0["devsync_ms"], 19.5)
    del sites

    # T4:大拷贝
    assert len(r0["T4_big"]) == 1
    c = r0["T4_big"][0]
    assert c["direction"] == "HtoD"
    close(c["gpu_ms"], 2.0)
    close(c["gbps"], 4194304 / 0.002 / 1e9, 1e-6)
    close(c["gap_before_ms"], 10.0)
    close(c["pre_cpu_copy_ms"], 7.5)
    assert "CPU copy_" in c["hint"] and "[推断]" in c["hint"]

    # T5:暴露通信与起始极差
    t5 = {r["collective"]: r for r in r0["T5"]}
    close(t5["AllReduce"]["overlap_ms"], 12.0)
    close(t5["AllReduce"]["exposed_ms"], 8.0)
    close(t5["AllReduce"]["overlap_rate_pct"], 60.0)
    close(t5["AllGather"]["exposed_ms"], 9.5)
    close(r0["exposed_comm_ms"], 17.5)
    close(t5["AllReduce"]["avg_bytes"], 2_000_000)
    close(t5["AllGather"]["avg_bytes"], 4_000_000)  # 取自 kernel args(优先于 CPU 标注的 2e6)
    close(s["comm_spread"]["AllReduce"]["max_ms"], 2.0)
    close(s["comm_spread"]["AllGather"]["max_ms"], 0.0)
    assert "同机时钟假设" in s["clock_note"]

    # 排行
    names = [r["name"] for r in s["ranking"]]
    assert names[0] == "暴露通信" and names[1] == "头部搬运空档" and names[2] == "同步导致 GPU 饥饿", names
    assert names[3].startswith("慢 rank 等待") and names[4] == "launch 小间隙", names
    close(s["ranking"][0]["ms"], 17.5)
    close(s["ranking"][3]["ms"], 1.235, 1e-6)
    assert s["compute_hotspot"]["category"] == "GEMM"

    # 报告:结构与百分比规范
    for title in ("## 0. 瓶颈排行", "## 1. 总览", "## 2. GPU 耗时速览", "## 3. 空闲归因", "## 4. 同步表",
                  "## 5. 搬运表", "## 6. 通信表", "## 7. 口径与旁证", "单次采集"):
        assert title in report, title
    assert "占累计%" in report and "占计算%" in report and "占 step%" in report and "占忙时%" in report
    assert "不可跨列相加" in report and "最慢 rank 1" in report
    assert "占峰值%" not in report, "不传 --peak-* 时不应出现占峰值%"

    # ---- 分类规则回归:C++ 名字里的 "lambda" 不能被当成优化器
    rules_default = tr.load_rules(None)
    clf = tr.Classifier(rules_default)
    assert clf.kernel("void at::native::elementwise_kernel<128, 4, {lambda(float)#1}>")[0] == "elementwise/reduce/copy"
    assert clf.kernel("void at::native::multi_tensor_apply_kernel<foo>")[0] == "optimizer"
    assert clf.kernel("void at::native::(anonymous namespace)::cuComputePartGradGammaBeta<float>")[0] == "norm/softmax"

    # ---- 影响下限:GPU 空闲占 step 低于阈值时不报饥饿,也不进排行
    out_f = tmp / "out_floor"
    assert tr.main(["--trace-dir", str(plain), "-o", str(out_f), "--min-impact-pct", "6"]) == 0
    sf = json.loads((out_f / "summary.json").read_text())
    item_f = next(r for r in sf["ranks"]["0"]["T3"] if r["event"] == "aten::item")
    assert item_f["verdict"].startswith("影响小"), item_f["verdict"]
    assert all(r["name"] != "同步导致 GPU 饥饿" for r in sf["ranking"])

    # ---- 带栈场景:取项目帧
    out_s = tmp / "out_stack"
    assert tr.main(["--trace-dir", str(stack), "-o", str(out_s), "--project-hint", "proj/"]) == 0
    ss = json.loads((out_s / "summary.json").read_text())
    assert ss["ranks"]["0"]["has_stack"] is True
    item_s = next(r for r in ss["ranks"]["0"]["T3"] if r["event"] == "aten::item")
    assert item_s["site"] == "proj/model.py:77 (forward)", item_s["site"]
    out_s2 = tmp / "out_stack2"  # 不给 hint:按库前缀过滤 dist-packages,同样落在项目帧
    assert tr.main(["--trace-dir", str(stack), "-o", str(out_s2)]) == 0
    item_s2 = next(r for r in json.loads((out_s2 / "summary.json").read_text())["ranks"]["0"]["T3"] if r["event"] == "aten::item")
    assert item_s2["site"] == "proj/model.py:77 (forward)", item_s2["site"]

    # ---- 峰值参数 / 规则覆盖 / 旁证数据
    rules = tmp / "rules.json"
    rules.write_text(json.dumps({"kernel_categories": [{"name": "我的类", "kind": "compute", "patterns": ["mystery"]}]}))
    run_dir = tmp / "run"
    (run_dir / "step_metrics").mkdir(parents=True)
    (run_dir / "command.json").write_text(json.dumps({
        "command": ["torchrun", "--nproc_per_node=8", "x.py", "--global_batch_size", "768"],
        "environment": {"OMP_NUM_THREADS": "1", "MYPROJ_X": "1", "SECRET": "no"}}))
    (run_dir / "step_metrics" / "rank-00.json").write_text(json.dumps(
        {"step_seconds": [1.0] * 10 + [1.5, 1.4, 1.6, 1.5], "max_memory_allocated_bytes": 2 ** 31}))
    clog = tmp / "compile.log"
    clog.write_text("Graph break in user code at foo.py:3\nRecompiling function f in file.py:1\n"
                    "    triggered by the following guard failure(s):\n    - 0/0: tensor size mismatch 5 vs 6\n")
    ka = tmp / "train.log"
    ka.write_text("noise\n"
                  "Name    Self CPU %    Self CPU    CPU total %    CPU total    CPU time avg    Self CUDA    Self CUDA %    CUDA total    CUDA time avg    # of Calls\n"
                  "-----    ---    ---\n"
                  "gemm_a    1.0%    1.0ms    1.0%    1.0ms    1.0ms    30.5ms    40%    30.5ms    1ms    2\n"
                  "elem_b    1.0%    1.0ms    1.0%    1.0ms    1.0ms    2.5ms    4%    2.5ms    1ms    2\n"
                  "-----    ---    ---\n")
    hta = tmp / "hta"
    hta.mkdir()
    (hta / "temporal-0.csv").write_text("rank,idle_time(us),compute_time(us)\n0,100,900\n")
    out2 = tmp / "out2"
    assert tr.main(["--trace-dir", str(plain), "-o", str(out2), "--rules", str(rules), "--run-dir", str(run_dir),
                    "--compile-log", str(clog), "--key-averages", str(ka), "--hta-dir", str(hta),
                    "--peak-h2d-gbs", "10", "--peak-comm-gbs", "20", "--env-prefix", "MYPROJ_"]) == 0
    s2 = json.loads((out2 / "summary.json").read_text())
    rep2 = (out2 / "report.md").read_text()
    assert any(r["category"] == "我的类" for r in s2["ranks"]["0"]["T1"]) and s2["ranks"]["0"]["unclassified_pct"] == 0
    assert "占峰值%" in rep2
    ev2 = s2["evidence"]
    assert ev2["run"]["global_batch_size"] == "768"
    assert "SECRET" not in ev2["run"]["env"] and ev2["run"]["env"]["OMP_NUM_THREADS"] == "1" and ev2["run"]["env"]["MYPROJ_X"] == "1"
    close(ev2["run"]["step_ms_median_after_warmup"], 1500.0, 1e-6)
    assert ev2["compile_log"]["graph_break_lines"] == 1 and ev2["compile_log"]["recompile_lines"] >= 1
    assert ev2["compile_log"]["top_recompile_reasons"]
    assert ev2["key_averages"]["top"][0]["name"] == "gemm_a"
    close(ev2["key_averages"]["top"][0]["self_cuda_ms"], 30.5)
    assert ev2["hta"][0]["file"] == "temporal-0.csv"

    # ---- 拒绝覆盖、缺 rank、缺目录
    assert tr.main(["--trace-dir", str(plain), "-o", str(out)]) == 2, "输出目录已存在必须拒绝覆盖"
    empty = tmp / "empty"
    empty.mkdir()
    assert tr.main(["--trace-dir", str(empty), "-o", str(tmp / "o3")]) == 2

    # ---- --step 指定
    out4 = tmp / "out4"
    assert tr.main(["--trace-dir", str(plain), "-o", str(out4), "--step", "5"]) == 0
    try:
        tr.main(["--trace-dir", str(plain), "-o", str(tmp / "out5"), "--step", "99"])
    except ValueError as exc:
        assert "ProfilerStep#99" in str(exc)
    else:
        raise AssertionError("不存在的 step 应报错")

    print("selftest: 全部断言通过")
    print(f"临时目录(可删除): {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
