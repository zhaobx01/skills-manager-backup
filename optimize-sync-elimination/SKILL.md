---
name: optimize-sync-elimination
description: Analyze or implement removal of implicit host-device synchronization and D2H copies in PyTorch training/inference (`.item()`, `.cpu()`, `.tolist()`, boolean-mask indexing, `nonzero`, blocking small transfers) to raise GPU utilization. Use when profiling shows idle GPU gaps around scalar reads, data-dependent control flow, or per-step logging/metrics. Not for general perf tuning unrelated to sync.
---

# Optimize sync elimination

以目标项目的实际调用路径和 Trace 证据为依据，找出并消除不必要的 host-device 同步（阻塞 D2H 拷贝、隐式 `cudaStreamSynchronize`/`cudaDeviceSynchronize`、数据相关的 CPU 分支），让 GPU 队列保持繁忙。配置存在、静态扫描命中、运行时确实发生同步、以及性能提升，是四个不同结论。静态检查只给候选，不证明运行时发生。只询问方法或分析时不修改项目；用户要求实施时按授权范围修改并验证。

## 确定范围和基线

1. 确认调用链里哪些位置读取张量的值来驱动 Python 控制流：loss/grad_norm 日志、动态长度/mask、数据相关的分支与循环边界、优化器状态标量、混合精度的 `found_inf`、checkpoint 元数据、调度器的 timestep/sigma。记录 PyTorch、后端（CUDA/ROCm-HIP）、分布式框架（DDP/FSDP/DeepSpeed）版本，优先用目标版本的文档确认某个调用是否仍会同步——行为在不同版本间会变。
2. 建立原始基线：有 profiler trace 时，先用 [Trace 中的同步证据](references/profiling.md) 确认真实发生同步的位置、频率和耗时占比，而不是只凭静态扫描下结论。没有 trace 时，用 [scripts/scan_sync_points.py](scripts/scan_sync_points.py) 产出候选清单，明确标注"静态候选，未证明运行时发生"。
3. 明确验收阈值：数值结果（loss、梯度、输出、checkpoint 恢复）必须与消除前逐位或在项目容差内一致；性能看 step 时间的中位数与分布、GPU idle gap、以及是否真的减少了 trace 里的同步事件数，而不是单看一次墙钟时间。不要把 compile、算子替换或精度改变的收益归到同步消除上。

## 实施与诊断

- **把标量判断留在设备上**：对张量做 `if`/`.item()`/`int()`/`float()`/`bool()` 来驱动控制流，通常意味着一次阻塞 D2H。能否改成设备端的 `torch.where`、`clamp`、掩码乘法，或者用 `.reshape(())` 保留零维张量而不转 Python 数字（MapTRv2 `sync_elimination.py` 把 loss 归一化因子的 `.item()` 换成 `.reshape(())`，见 [TurboPhysAI 证据](references/turbophysai-sync-evidence.md)）。改写前确认下游确实只需要设备端结果，不需要 Python 侧的真实数值做分支。
- **元数据本来就在 CPU 时不要搬上去再搬下来**：图像尺寸、文本/序列长度、mask 的有效长度等如果来自 CPU 侧的 dataloader 或配置，保持在 CPU 上计算，只把真正要参与设备计算的张量搬上 GPU（Cosmos3 `batch_transfer.py` 拦截 `misc.to(...)`，只让 `image_size` 留在 CPU；OpenWAM `text.py`/`ports/encode.py` 在 CPU 上算长度，避免 `.tolist()` 读 GPU 张量）。这类改写必须精确匹配原调用的形状和参数，否则会悄悄改变其他调用点的行为。
- **布尔掩码索引换成固定形状的操作**：`tensor[bool_mask]` 或依赖 `nonzero()` 结果长度的后续分配，输出形状由数据决定，框架需要先知道非零个数才能分配输出，这通常是一次同步。换成 `torch.where`、`masked_fill`、`index_select` 配合提前知道的上界，或者用 `scatter_add_`/`index_copy_` 在固定大小的缓冲区上写（BEVFusion 的 `quick_cumsum` 用 `torch.where(kept)[0]` 与 `index_select` 取代布尔索引）。确认索引张量上的梯度语义不变。
- **自定义 varlen / 长度相关 API 常在前端里悄悄调用 `.max().item()`**：FlashAttention 等库的便捷封装经常从 `seqlens` 算 `max_seqlen` 再传给底层 kernel，这一步是 CPU 标量。如果已知序列组织方式（如单样本打包、固定分桶上界），自己构造 `cu_seqlens`/`max_seqlen` 作为 Python int 或预先分配的张量，绕开封装层（Cosmos3 `mot_attention.py` 自建 `cu_seqlens`，绕开 `generate_varlen_parameters` 里的 `.max().item()`）。这类改写依赖 "batch=1 的打包序列" 或类似强前提，迁移到别的模型前要重新核实前提是否成立。
- **指标和日志合并、延迟读取**：每步对 loss/grad_norm 单独 `.item()` 会导致 GPU 队列排空。改成多个标量 stack 成一个张量只做一次归约和一次 `.cpu()`，或者把结果拷到 pinned 内存配合 `cuda.Event`，下一步开始前才 `event.synchronize()` 读取（上一步的梯度已经不再被用，读取延迟一步在语义上通常是安全的，但要确认训练逻辑不依赖同步的即时值）。参考 OpenWAM `host_async.py` 和 BEVFusion `parse_losses`。
- **关掉库内部默认开启的同步**：分布式框架、profiler、吞吐计时器有时默认在每步调用一次 `cuda.synchronize()`（如 DeepSpeed 的 throughput timer）。这类开关通常和真正的训练逻辑无关，可以直接关闭，但要确认关闭后不影响依赖该同步点的其他功能（比如某些 OOM 检测或计时逻辑）。
- **必须留在 CPU 的求解不可避免一次 D2H，但要合并成一次，并放在编译图/cudagraph 之外**：匈牙利匹配（`scipy.optimize.linear_sum_assignment`）只能在 CPU 跑，常见做法是把代价矩阵一次性 `.detach().cpu()`，结果一次性 `torch.from_numpy(...).to(device)` 传回（MapTRv2 `assigner.py`、BEVFusion `transfusion.py`）。这不是"消除"，是"收敛成一次"；不要在同一个函数里对代价矩阵再做零散的 `.item()`。
- **非阻塞拷贝的安全条件**：`tensor.to(device, non_blocking=True)` 只有源张量在 pinned host memory 时才真正异步；对非 pinned 的 CPU 张量调用它不会报错，但也不会变快，而且如果消费方没有等待对应的 stream/event，可能读到尚未完成的数据。核对 `pin_memory()` 的调用点，以及消费前是否有 `event.synchronize()` 或默认 stream 上的隐式同步。
- **与 `torch.compile`、分布式的交互**：进入编译区域后，`.item()`、依赖数据的分支、`nonzero` 等通常直接导致图中断或报错，而不是悄悄同步——这类问题先交给 compile 相关 skill 处理图中断诊断，本 skill 关注 eager 路径和编译区域之外的同步。DDP/FSDP/DeepSpeed 的集合通信顺序问题（如 RCCL 要求同 communicator 全局顺序一致）和同步消除是两类不同问题，参见下面的"邻近但不同的问题"。

### 邻近但不同的问题（不在本 skill 范围内，只做指引）

- **FlashAttention varlen 的 `max_seqlen` 分桶**：目的是减少 `torch.compile` 重编译（避免 Python int guard 每批次变化），不是消除同步；如果同时也绕开了 `.max().item()`，才同时算同步消除。
- **分布式集合通信的有序调度**（如 OpenWAM `deferred_gather`、FSDP 的 ordered all-gather）：解决的是通信重叠与顺序正确性，不是 host-device sync；归到分布式/通信相关的优化去处理。
- **channels-last、AOT 编译、算子融合**：这些改变的是张量布局和计算图结构，即使文档里提到"避免了某次拷贝"，也要先确认那次拷贝是 D2H/H2D 同步还是设备内的布局转换，二者不同。

## 验收与回退

按 [验证方法](references/validation.md) 比较相同代码、相同 compile/环境设置下改写前后的数值与性能。覆盖代表性的动态长度、空输入、batch=1 等边界；训练场景额外验证梯度、优化器状态、checkpoint 保存与恢复后数值一致。性能验收优先看 trace 里同步事件的数量和耗时是否真的下降，而不是单次墙钟差异；多次测量报告中位数和离散程度，注明是否包含首次编译/预热。

若消除后数值不一致、功能回归、或收益不明显，关闭改动恢复原路径，报告触发条件与实际执行路径，不要静默吞掉错误。非阻塞拷贝相关的改动尤其要验证："改了但没变快"通常意味着拷贝源不是 pinned memory，或者消费点仍在等待；"变快了但结果错了"通常意味着提前读取了未完成的数据。没有目标设备或依赖时完成静态工作，标明未运行的部分及原因，不宣称已验证。

按需阅读 [TurboPhysAI 跨模型证据](references/turbophysai-sync-evidence.md)（六个模型加 GR00T 的真实案例、未核实候选、以及与现有 compile 相关 skill 的对照）。这些案例说明可考虑的手法，不证明当前目标项目也存在同样的同步点或会获得同样的收益。

需要系统过一遍可消除同步的模式和适用前提时，阅读 [同步消除模式](references/patterns.md)；需要用已采集的 Trace 确认候选是否真实发生同步时，阅读 [Trace 中的同步证据](references/profiling.md) 并配合 [scripts/analyze_trace_sync.py](scripts/analyze_trace_sync.py)；没有 trace 只想先扫一遍代码找候选时，用 [scripts/scan_sync_points.py](scripts/scan_sync_points.py)。两个脚本都只做静态文本/trace 分析，不运行目标模型。
