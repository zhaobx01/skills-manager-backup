# Trace 中的同步证据

本文只讲怎么从已经采集好的 PyTorch profiler trace（Chrome trace JSON，`torch.profiler` 导出的格式）里找到同步证据，不提供采集代码。采集本身由你在目标设备上用 `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True)` 完成，导出 `trace.export_chrome_trace(path)`。

## 为什么要看 trace 而不是只信静态扫描

静态扫描（[scripts/scan_sync_points.py](../scripts/scan_sync_points.py)）只能说"这行代码调用了 `.item()`"，不能说：
- 这行代码在当前配置下是否真的被执行到（可能在 `if` 分支里，当前没走到）。
- 即使执行到，是否真的触发了设备同步（取决于前面有没有未完成的 kernel 排队；如果 GPU 队列本来就空，`.item()` 几乎不耗时）。
- 同步的实际耗时占整个 step 的比例有多大，值不值得改。

Trace 能回答这三个问题，是消除同步前后对比收益的主要依据。

## 在 trace 里找什么

CUDA 和 ROCm/HIP 的事件命名不同，以下按两套列：

| 要找的东西 | CUDA 典型事件名 | ROCm/HIP 典型事件名（未验证，按名称推断） |
| --- | --- | --- |
| 显式同步 | `cudaDeviceSynchronize`、`cudaStreamSynchronize`、`cudaEventSynchronize` | `hipDeviceSynchronize`、`hipStreamSynchronize`、`hipEventSynchronize` |
| D2H/H2D 拷贝 | `Memcpy DtoH`、`Memcpy HtoD`、`cudaMemcpyAsync` | `Memcpy DtoH`、`Memcpy HtoD`、`hipMemcpyAsync`（HIP 的 Chrome trace 事件名经常沿用 CUDA 风格的分类名，但底层 API 名不同） |
| Python 侧触发点 | `aten::item`、`aten::_local_scalar_dense` | 同名（这是 ATen 层，不区分后端） |
| 条件/布尔索引相关 | `aten::nonzero`、`aten::masked_select`、`aten::index` | 同名 |

**DTK（海光/Hygon 的 ROCm 分发版）的具体事件命名我们没有验证过**；如果在目标环境的 trace 里看不到预期的 HIP 事件名，先用 `grep -o '"name": *"[^"]*"' trace.json | sort -u` 把所有出现过的事件名列出来，再调整匹配规则，不要假设命名和上游 ROCm 完全一致。

## 判断某次 `.item()` 是否真的造成了可见的 GPU 空闲

1. 找到 CPU 线程上 `aten::item`（或对应调用）这个事件的时间区间。
2. 看同一时刻 GPU 线程（trace 里通常是单独的 "stream" 轨道）是否有事件在跑。如果 `.item()` 期间 GPU 轨道完全空白，且空白持续到下一个 kernel 提交，这段空白就是可疑的同步代价。
3. 看 `.item()` 之后下一个 CPU 侧事件（通常是下一次 kernel launch）的时间差——如果几乎紧贴，说明确实是"CPU 等 GPU 排空"，而不是"CPU 本来就在做别的事"。
4. 对比关闭某个同步消除开关前后的两份 trace，同一时间窗口内，GPU 轨道的空白总时长和 CPU 侧同步事件的调用次数应该同时下降；只有调用次数下降但空白时长不变，说明那次同步原本就不是瓶颈。

## 用 scripts/analyze_trace_sync.py 做什么

这个脚本只做机械匹配和汇总，不替你判断"值不值得改"：
- 按事件名匹配上表里的模式（CUDA 和 HIP 都试），统计每种事件出现的次数、总耗时。
- 如果 trace 里带 `with_stack=True` 采的 Python 调用栈，按调用栈聚合，告诉你"这个 `.item()` 来自哪一行代码"。
- 输出按总耗时排序的列表，方便你决定先改哪个。

脚本不会告诉你某次同步是不是必要的，也不会告诉你改了之后会快多少——这两件事都需要你在改动前后分别采集 trace 做对比。

## 常见误读

- **静态扫描器分不清张量是否已经在 CPU 上**：这是最常见的误报来源。例如 GR00T `gr00t/model/modules/qwen3_backbone.py:139` 的 `_grid_thw_as_int_rows` 先显式 `grid_thw.detach().to(device="cpu", ...)` 做一次 D2H，再对结果 `.tolist()`——`scripts/scan_sync_points.py` 会把这个 `.tolist()` 标成候选，但它作用的对象已经是 CPU 张量，不会再触发一次同步。同文件 `_cpu_fast_pos_embed_interpolate`（约 186-187 行）里的 `indices[index].tolist()`、`weights[index].tolist()` 同理：这两个列表本来就是在 CPU 上构造出来的。看到扫描结果后，先去读调用点前面几行，确认操作对象的来源，再决定要不要改。
- **事件次数多不代表耗时占比高**：一个调用了几万次但每次耗时几微秒的 `.item()`，总耗时可能远小于一次耗时几十毫秒的大张量 D2H 拷贝。优先看总耗时，不是次数。
- **CPU 侧耗时不等于 GPU 空闲时长**：`.item()` 本身在 CPU 线程上花的时间包含了等待 GPU 完成的时间，但如果 GPU 本来就还有别的工作在排队（多 stream），CPU 等待不一定意味着 GPU 空闲。要对照 GPU 轨道确认。
- **trace 窗口太短**：只采了几步的 trace 可能包含编译/预热开销，把这部分和稳态混在一起会高估或低估同步的影响；采集时跳过前几步。
