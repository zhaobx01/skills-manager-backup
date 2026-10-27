---
name: optimize-channels-last
description: Analyze or implement PyTorch channels-last memory-format optimization for convolutional 4D or 5D workloads, including layout boundaries, correctness, and measured performance. Use when evaluating or adding channels_last or channels_last_3d, not for arbitrary NHWC API conversion.
---

# Optimize channels-last

以目标项目的实际调用路径和测量为依据，评估或接入 channels-last；保持模型数学、逻辑维度和精度语义。配置存在、布局实际生效、正确性通过和性能提升是四个不同结论。静态检查不能证明运行或收益。只询问方法或分析时不修改项目；用户要求实施时按授权范围修改并验证。

## 确定范围和基线

1. 确认调用链、卷积热点、真实 shape、dtype、device、训练或推理、输入预处理、模型构建/加载、compile、分布式及自定义算子。记录 PyTorch、后端和硬件版本，优先使用目标版本的文档。没有 profiler 时把热点作为假设；对没有卷积热点或转换成本过高的路径可以不启用。
2. 建立原始基线和明确验收阈值：输出/任务指标，训练时还包括 loss、梯度、更新及状态；性能包括端到端和热点、峰值显存、冷启动和稳态。保持数据、精度、seed、批次、配置和设备一致。不要将 compile、环境变量或算法变化的收益归因于布局。
3. 按卷积子图选择转换边界，先小范围测试。4D 的逻辑形状仍是 N,C,H,W，使用 `torch.channels_last`；5D 仍是 N,C,D,H,W，使用 `torch.channels_last_3d`。仅对实际为图像/体积的 4D/5D 张量转换，不能按维数盲改 token、相机维、索引或元数据。不要将物理内存格式当成 `permute` 后的 NHWC/NDHWC 逻辑接口。

## 实施与诊断

- 提供显式可关闭配置，尊重项目默认行为；在合适的模型生命周期点转换相关 Conv2d/Conv3d 权重和输入，避免在每层或热循环中重复复制。整体 `module.to(memory_format=...)` 可能触及其他 4D/5D 参数；混合模型可只转换卷积子模块或使用版本支持的卷积权重转换工具。先核对共享参数、优化器状态、checkpoint、DDP/FSDP 初始化及加载顺序，转换前后验证对象与状态，不假定通用顺序。
- 输入使用 `x.to(memory_format=torch.channels_last)` 或 `torch.channels_last_3d`，依据维度选择；输出和索引的逻辑顺序保持原有契约。`to` 对单维退化形状的显式 restride 比 `contiguous(memory_format=...)` 更可靠，但某些完全退化形状仍有歧义。记录 `shape`, `stride()`, `is_contiguous()` 和对应 memory format 检查；两个检查同时为真并不能证明后端采用某种算法。
- 追踪 reshape/view、permute、切片、拼接、padding、插值、缓存、数据增强和自定义算子边界。`view` 有 stride 要求，不能盲改为 reshape 后忽略潜在复制；检查 alias、原地更新、梯度与输出协议。算子可能改变布局或引入复制，使用 profiler 检查转换、copy 和实际瓶颈；只在有必要的边界修复布局，保持不支持该格式的算子接口。
- 分别验证 eager 与实际使用的 `torch.compile`：转换置于合适的编译边界，观察 stride guards、重编译、图断和运行时布局，不假定 eager 中的转换在融合图里一定产生预期布局。OpenWAM 使用了私有 Inductor primitive，但它是版本依赖的案例，不作为通用方案；只有明确需要时才核对版本、兼容性、fallback 和梯度。
- ROCm 上把 `PYTORCH_MIOPEN_SUGGEST_NHWC` / `PYTORCH_MIOPEN_SUGGEST_NDHWC` 视为版本和构建相关的后端提示，不等于输入或权重转换，也不保证算法或加速。`MIOPEN_FIND_MODE` 控制搜索策略和启动成本；独立消融并检查该版本的支持、默认值和数据库状态。启动进程前设置并记录原值；不顺带改变 TF32、GEMM、精度或其他变量。详见 [布局与后端](references/layout-and-backends.md)。

## 验收与回退

按 [验证方法](references/validation.md) 比较相同代码、相同 compile/环境设置下开关关闭与开启。覆盖代表性动态形状及退化维度、输入输出、训练梯度与状态，并测端到端转换成本。测量前同步设备，预热，报告多次测量的中位数及离散情况，分别记录首次编译/搜索和稳态；必要时记录多 rank 最大耗时。显存和任务指标也纳入验收。

若不支持、正确性不达标、实际回归或收益不抵成本，关闭开关并验证恢复原路径、格式及结果；报告触发条件与实际执行路径，不静默吞掉错误。对于已经创建的模型或编译缓存，重新构建/重启进程以获得干净对照，避免切开关后残留权重布局、已捕获图或后端缓存。用户允许局部回退时可隔离不兼容子图并重新测试。没有目标设备或依赖时完成静态工作，标明未运行的项目及原因，不宣称通过或提速。

按需阅读 [源码案例与证据范围](references/source-evidence.md)。这些案例说明可考虑的边界，不证明当前模型启用了优化或获得收益。

