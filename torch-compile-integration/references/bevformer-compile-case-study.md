# BEVFormer：不同粒度编译与 query 长度分桶

本文件是完整随包提供的静态案例，不需要安装 BEVFormer 或访问原项目。源码采用不等于运行捕获或加速。问题入口见 [跨模型问题参考](compile-problem-patterns.md)。

## 来源与范围

固定来源：[iMountTai/TurboPhysAI，bb302a6363a06582e303f052c743469c55af65bf](https://github.com/iMountTai/TurboPhysAI/tree/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevformer)。下列路径相对该目录，目标符号相对原始项目。仅核对该快照的注册、配置和实现，未独立对照完整 BEVFormer 基线，未运行模型。布局和计算改写仅作为理解边界及归因的背景。

## 编译目标与接入

| 编号 | 原项目限定符号 | 接入与配置 Group | 包装的实际对象 | 参数 |
| --- | --- | --- | --- | --- |
| BF01 | `projects.mmdet3d_plugin.bevformer.detectors.bevformer.BEVFormer.extract_img_feat` | `catalog.py` 的 `bevformer.mdc` → `backbone.py:compiled_extract_img_feat` | 适配后的 `extract_img_feat`，不是传入的 original | options.mode，默认 max-autotune-no-cudagraphs |
| BF02 | `projects.mmdet3d_plugin.bevformer.modules.encoder.BEVFormerLayer.forward` | `bevformer.compile.encoder` → `compile.py:compile_wrapper` | 传入的方法；是单层边界，不是 Encoder 外层循环整体 | 同上 |
| BF03 | `projects.mmdet3d_plugin.bevformer.modules.decoder.DetectionTransformerDecoder.forward` | `bevformer.compile.decoder` → `compile.py:compile_wrapper` | 传入的 Decoder forward | 同上 |

三个入口均未显式传 dynamic/fullgraph，不能从别的案例借用参数，也不能认为存在三个实际图或所有实例均已命中。`configs/recipe.yaml` 开启上述三个 Group；`optimization.yaml` 的 Encoder/Decoder options 为空，采用包装器默认 mode。图像特征入口放在 mdc Group 内，不是独立 compile Group。

来源：[编译包装器](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevformer/compile.py#L10)、[图像特征包装](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevformer/backbone.py#L73)、[注册](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevformer/catalog.py#L161)、[recipe](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevformer/configs/recipe.yaml)。

## 开关与真实关闭路径

包装器在准备阶段检查 `TURBO_PHYSAI_DISABLE_TORCH_COMPILE == "1"`。BF02/BF03 关闭时返回 original；BF01 丢弃 original，关闭时仍返回适配后的 `extract_img_feat`，保留其形状整理及布局相关改动。

启用后返回的 callable 不逐次读取该变量；运行中修改变量不会自动撤销已有包装。建议独立进程切换并验证实际入口。源码没有 API 能力探测或编译异常捕获；创建包装或首次调用报错不能称为自动 eager 回退。切换整个 mdc Group 还可能改变其他独立实现，不是纯 compile 对照。

迁移价值是：同一模型可在“特征提取阶段、重复单层、Decoder 整体”选不同边界，且启停需要区分原实现与重构后 eager。默认 mode 属于项目配置，不是其他硬件的建议默认值。

## SCA 长度分桶

`geometry_sca.py:encoder_forward` 先尝试 `_prefetched_sca_inputs`：

- 没有预取输入时，从 mask 算真实 `index_lengths`，仍调用 `.max().item()` 获取 Python 长度，再向上选择 9728、10240、11264、12288 中的桶；桶值取 `min(cam_query_mask.size(-1), bucket_len)`，上界是 mask 最后一维的长度。桶值直接写在 `geometry_sca.py` 的元组里，没有 options 开关。
- 长度大于全部桶时保留实际最大长度，不是固定截断到最大桶。长度为零也可能得到非零桶容量，不能将容量当有效数量。
- 有预取输入时直接使用预取的 indexes 和 lengths，不经过 `geometry_sca.py` 里这段现场分桶。分桶改在数据侧：`data.py` 的 `_prepare_bev_meta_group` 在 numpy 数组上取同一批次内 `index_lengths` 的最大值，用 `bev_meta_cfg['buckets']` 里同样的四个桶值向上取整，上界同样是 mask 最后一维的长度，再按该长度截取 `argsort` 结果。这一步在 CPU 上完成，不产生 GPU 同步。
- 预取路径的桶值在 `data.py` 的 `build_dataloader` 里硬编码，与现场路径的元组互不引用，改桶值需两处同步。`data.py` 里另有一个带同样桶值的 `_build_bev_meta_prefetch_cfg`，但全仓库没有调用点，不是实际生效的配置来源。预取只在 shuffle 的训练 loader 且 `TURBO_PHYSAI_CUDA_PREFETCH` 为 1 时创建，运行时再由 `ENABLE_BEV_META_PREFETCH` 控制是否计算；关闭时 meta 里没有预取字段，退回现场路径。验收时仍要确认实际命中的是哪条路径。
- `spatial_cross_attention_forward` 用真实 lengths 构造有效 mask，并在 gather/attention/scatter 路径使用。任何分桶迁移均须覆盖补齐元素不影响有效输出及梯度的契约。

这改变索引承载长度以尝试减少下游编译形状变化，和 Cosmos3 仅调整 Attention Python launch 参数不是同一机制；现场路径没有消除 `.item()`，预取路径把取最大长度挪到了数据侧 CPU 上；两条路径都不保证无重编译。

来源：[现场分桶](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevformer/geometry_sca.py#L278)、[有效元素处理](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevformer/geometry_sca.py#L409)。

## 最小验证与证据

| 场景 | 目标项目需要验证 |
| --- | --- |
| 首次关闭、首次开启、开启后修改变量 | 准备和调用阶段各自的真实对象；关闭不等于恢复全部基线 |
| 单个及组合边界 | 类实例匹配、避免重复包装、输出/梯度、图捕获和边界融合成本 |
| 分桶的零、边界两侧、超最大桶、容量上限 | 真实 lengths 保留、mask/索引/梯度正确；不虚构固定图数 |
| 预取／非预取路径 | 实际数据契约、命中路径与形状分布分别记录 |
| 性能 | 同一重构与布局下 compile 开关对照；Trace 诊断与无 profiler 正式测速分开 |

本轮仅静态核对与隔离控制检查；未运行模型、真实 Dynamo、GPU 正确性或性能。项目 recipe 的“validated”文字属于来源描述，不是本次验证结果。
