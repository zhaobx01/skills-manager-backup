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

示例(通用 `torch.profiler`;参数名按项目实际接入方式调整):

```python
from torch.profiler import profile, schedule, ProfilerActivity
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],   # ROCm/HIP 同样用 CUDA 这一项
    schedule=schedule(skip_first=20, wait=1, warmup=5, active=3, repeat=1),
    with_stack=False,                                           # 计时用 False;归因另采一次改 True
    on_trace_ready=lambda p: p.export_chrome_trace(f"{out_dir}/trace_rank_{rank}_iter_{p.step_num}.json"),
)
# 训练循环每步结束调用 prof.step()
```

各 rank 都要导出;文件名保持 `trace_rank_<rank>_*.json`。如需 rank 0 的 key_averages,在导出时另存 `p.key_averages().table(sort_by="self_cuda_time_total")`。

## 2. 运行记录(推荐,格式由用户的训练脚本自行写出)

脚本只认下面两种可选文件,字段缺失即跳过,**不要求任何特定框架**:

- `<run-dir>/command.json`:`{"command": ["启动命令", "参数", ...], "environment": {"NAME": "value"}}`。用于报 batch/卡数(识别 `--global_batch_size`/`--num_gpus`/`--nproc_per_node` 或环境 `GLOBAL_BATCH_SIZE`/`NUM_GPUS`)和环境开关。环境变量默认只保留 `HIP_/NCCL_/RCCL_/OMP_/TORCH_/PYTORCH_/CUDA_/GPU_` 前缀,项目自己的开关前缀用 `--env-prefix` 追加;写文件时请自行剔除密钥。
- `<run-dir>/step_metrics/rank-<n>.json`:`{"step_seconds": [每步墙钟秒数...], "max_memory_allocated_bytes": N}`。在训练循环里用 `time.perf_counter()` 记每步间隔、结束时写出即可;**不受 profiler 影响**,用来报真实 step 时间(`--warmup-steps` 控制跳过前几步)。
- 另建议人工核对:profiler 窗口、是否带栈、编译开关,写进报告口径说明。

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
