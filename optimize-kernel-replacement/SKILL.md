---
name: optimize-kernel-replacement
description: Analyze or implement replacement of PyTorch operators with faster fused or vendor implementations (fused AdamW, vendor LayerNorm/RMSNorm kernels, FlashAttention backend selection, real-valued RoPE, fused conv+bias graphs, custom_op-registered kernels), including availability checks, fallback paths, numerical equivalence and measured performance. Use when evaluating or adding such replacements on CUDA or ROCm/HIP. Not for writing new Triton/CUDA kernels from scratch or for host-device sync removal.
---

# Optimize kernel replacement

以目标项目的实际调用路径和测量为依据，评估或接入"用更快的实现换掉原实现"。保持模型数学、输入输出协议和用户要求的数值精度。配置存在、替换真正命中、数值等价和性能提升是四个不同结论；静态检查只给候选，不证明运行或收益。只询问方法或分析时不修改项目；用户要求实施时按授权范围修改并验证。

## 确定范围和基线

1. 确认真实热点：用 profiler 或 trace 找到占时的算子和 kernel（归一化、逐元素、注意力、优化器 step、卷积）。没有 trace 时，热点只是假设，可先用 [scripts/scan_kernel_candidates.py](scripts/scan_kernel_candidates.py) 产出"未替换候选"清单。记录 PyTorch、后端（CUDA 或 ROCm/HIP）、厂商库及其版本，替换是否可用取决于这些。
2. 逐个确认候选的前提：目标库是否已安装并提供所需函数、dtype 与 shape 是否在支持范围内、是否需要连续内存、是否处于 `torch.compile` 区域内、是否带 hook 或自定义 forward。前提不满足时必须有明确的回退，不能让替换悄悄改变行为。
3. 建立基线和验收阈值：输出、loss、梯度、优化器状态、checkpoint 保存恢复要在项目容差内一致；性能看目标热点和端到端 step 时间的中位数与分布。不要把 compile、布局、同步消除等其他改动的收益归因于算子替换。

## 实施与诊断

- **替换要有开关、可用性检查和回退三件套**：开关默认值按项目约定，关闭后恢复原路径；可用性检查在调用时判断（库缺失、函数缺失、dtype、形状、设备类型），不满足则走原实现，而不是抛错或悄悄走另一条路径。OpenWAM 的 `lightop_norm` 只在 HIP、`x.shape[-1] >= 64`、参数与输入同 dtype 且连续时才进入厂商 kernel，其余回退原 `LayerNorm`（见 [证据](references/turbophysai-kernel-evidence.md)）。
- **依赖私有或不稳定接口时要锁版本**：直接调用库的私有函数（如 FlashAttention 的 `_wrapped_flash_attn_varlen_forward`）不属于稳定 API。OpenWAM 用目标函数源码的 SHA256 白名单判断是否认识当前构建，不认识就回退公开入口；Cosmos3 则显式放宽版本范围并在断言里要求私有函数存在。迁移时优先选择这类"认不出就回退"的写法。
- **融合优化器要处理 checkpoint 和数值差异**：`fused=True` 改变的是更新的实现方式，源码里有一处注释自述"same AdamW update, rounding may differ slightly"。第三方融合实现（如 Apex FusedAdam）通常有状态格式和不支持的选项（amsgrad、capturable、master_weights 等），需要校验并在保存时转回标准 AdamW 格式；多参数的 step 不一致时应报错，不能静默取其一。
- **注意力后端选择与数值正确性是两件事**：切换 FlashAttention、SDPA 的后端时，先在目标平台确认数值和梯度没有 NaN；有项目在 HIP 上默认强制 `SDPBackend.MATH`，原因是默认后端曾产生 NaN 梯度，这是为正确性放弃速度，不是优化。只询问或分析时要如实区分"换成更快后端"和"退到更稳后端"。
- **逐元素表达式的融合优先交给编译器**：`x * (1 + scale) + shift`（adaLN 调制）、`x + gate * residual`、RMSNorm 的 `rsqrt` 表达式、实数形式的 RoPE，这些无状态表达式可以单独 `torch.compile`，不必手写 kernel。需要核对：是否已处于外层编译区域内（嵌套分派会给共享函数产生多份特化）、是否带 `dynamic` 与 `fullgraph` 的约定、关闭编译开关后是否仍保持数值一致。
- **自定义算子要让编译器能追踪**：把外部 kernel 包成 `torch.library.custom_op`，补 `register_fake` 和 `register_autograd`，否则进入 `torch.compile` 会断图。fake 实现的形状与 dtype 必须与真实 kernel 一致。图中断本身的诊断交给 compile 相关 skill。
- **融合卷积类算子的输入契约**：此类 kernel 往往只支持 bf16、特定通道数、特定 stride 和布局，不满足时回退普通卷积。记录每一条契约和被排除的层（例如源码里被显式排除的输入通道为 12 的 stem 层），并确认回退路径输出一致。布局本身的问题参见 `optimize-channels-last`。

### 邻近但不同的问题（只做指引）

- **host-device 同步**：用 `.item()`、`.cpu()` 读取标量等，见 `optimize-sync-elimination`。
- **内存布局**：`channels_last` 与 NHWC kernel 的格式要求，见 `optimize-channels-last`。
- **编译边界与重编译**：见 `torch-compile-integration`。
- **集合通信的重叠与顺序**：不在本 skill 范围。

## 验收与回退

按 [验证方法](references/validation.md) 比较相同代码、相同环境下开关关闭与开启：覆盖代表性 shape、dtype、空输入和边界值，训练场景还要比较梯度、优化器状态和 checkpoint 往返。性能先用 profiler 确认目标 kernel 真的被调用，再比较 step 时间的中位数与离散；分别记录首次编译或图构建与稳态。没有目标设备或依赖时完成静态工作，标明未运行的项目，不宣称通过或提速。

若可用性检查失败、数值不达标、实际回归或收益不抵成本，关闭开关并验证恢复原路径。切换开关后用新进程重新构建模型和优化器，避免残留包装、缓存的图或已注册的优化器类。

按需阅读 [通用模式](references/patterns.md)（替换的前提与不适用条件）和 [TurboPhysAI 跨模型证据](references/turbophysai-kernel-evidence.md)（OpenWAM、Cosmos3、BEVFormer、MapTRv2、OpenVLA、GR00T 的真实案例与未找到项）。这些案例说明可考虑的手法，不证明当前项目会获得同样收益。没有 trace 只想先扫一遍代码时，运行 `scripts/scan_kernel_candidates.py`，结果只是静态候选。
