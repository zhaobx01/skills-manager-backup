# 采集清单

本清单只说明**需要什么数据、怎么开**,不执行采集。采集会占用设备和时间,须由用户确认后再做。

## 1. Profiler trace(必需)

| 项 | 建议 |
| --- | --- |
| 活动 | CPU + 设备(CUDA/HIP),所有 rank 都导出 |
| 窗口 | `schedule(skip_first=N, wait=1, warmup≥5, active=3, repeat=1)`;`skip_first` 要越过首次编译/预热,否则 trace 里混入编译时间;active 步数只有 1 时结论是"单步样本" |
| 文件名 | `trace_rank_<rank>_*.json`(脚本按 `trace_rank_(\d+)` 识别 rank,可 `.gz`) |
| 一致性 | 所有 rank 来自同一次运行、同一窗口;不要混入不同 batch 或不同开关的文件 |
| 调用栈 | **计时用的 trace 不开 `with_stack`**。要定位同步/拷贝的代码行,另采一次带栈的(CPU 计时变长、文件约大 3 倍) |
| 形状 | `record_shapes=True` 可选,脚本不依赖 |
| 其他 | 采集期间关闭数值检查、模型保存;保证设备空闲,避免外部负载干扰 |

采集后检查:rank 文件数 = 卡数;每个文件含 3 个左右的 `ProfilerStep#N`;有 `kernel` 事件;文件大小在各 rank 间接近。

示例(GR00T 微调项目):

```bash
# 计时用:不带栈
--enable-profiling --profiler-skip-first 20 --profiler-wait 1 --profiler-warmup 5 \
--profiler-active 3 --profiler-repeat 1 --no-profiler-with-stack
# 归因用:带栈(另采一次)
--enable-profiling ... --profiler-with-stack
```

输出在 `<output_dir>/profiling/trace_rank_<rank>_iter_<step>.json`,rank 0 同时把 `key_averages()` 表打印到日志。

## 2. 运行记录(推荐)

- `command.json`:启动命令和环境变量(batch、卡数、`OMP_NUM_THREADS`、各开关)。
- `step_metrics/rank-*.json`:每步 `step_seconds`、峰值显存;**不受 profiler 影响**,用来报真实 step 时间。
- 生效配置:确认 profiler 窗口、是否带栈、编译开关。

## 3. 编译日志(可选,单独采集)

`TORCH_LOGS="graph_breaks,recompiles"` 另跑一次短训练并保存日志。日志写入会影响时间线,**不要和 profiler trace 同一次采集**。需要结构化报告时可加 `TORCH_TRACE` 并用 `tlparse` 生成(脚本不解析 tlparse)。

## 4. key_averages 文本(可选)

rank 0 的 `prof.key_averages().table(sort_by="self_cuda_time_total")` 输出,或包含它的训练日志。只作 top 算子交叉校验;它是活跃窗口内的累计值,不含时间线。

## 5. HTA 输出(可选)

已有的 `temporal-*.csv`、`idle-*.csv`、`overlap-*.csv` 所在目录。本 Skill 不运行 HTA。

## 6. 缺数据时怎么办

| 缺什么 | 影响 |
| --- | --- |
| 没有 trace | 无法统计;只交付本清单 |
| 没有设备 kernel 事件 | 脚本报错;确认 profiler 活动包含设备 |
| 没有 `ProfilerStep` | 退化为整个 trace 窗口并提示 |
| 没有调用栈 | 同步表只到外层算子链,标"未定位到代码行" |
| 没有运行记录/编译日志/key_averages/HTA | 只影响 §7 对应项,排行和 §1~§6 不受影响 |
