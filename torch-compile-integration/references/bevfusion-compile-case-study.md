# BEVFusion：分模块编译、eager 边界与重构消融

本文件完整随包提供，不要求安装 BEVFusion、访问原电脑或联网。按问题查阅见 [跨模型问题参考](compile-problem-patterns.md)。本例只记录直接编译及必要配套，不展开独立算子、布局、通信和数据加载优化。

## 来源与范围

固定来源：[iMountTai/TurboPhysAI，bb302a6363a06582e303f052c743469c55af65bf](https://github.com/iMountTai/TurboPhysAI/tree/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion)。以下路径相对该目录。核对 catalog、recipe、包装器与调用点，未独立对照完整原始模型仓库，未运行模型；注册数量、包装数量、实际图数分开统计。

## 八个编译目标

| 编号 | 目标与原项目定位 | 实际接入方式 | Group 与参数 |
| --- | --- | --- | --- |
| BU01 | `model.encoders["camera"]["backbone"]` | `training.py:_compile_model` 编译实例并替换槽位；`training_wrapper` 在调用原训练入口前执行 | bevfusion.training，options.compile_mode；显式 fullgraph=False、dynamic=False |
| BU02 | `mmdet3d.models.backbones.sparse_encoder.SparseEncoder` | `compile.py:compile_class_wrapper` 将类对象传给 torch.compile | bevfusion.compile.sparse_encoder，options.mode |
| BU03 | `mmdet3d.models.necks.generalized_lss.GeneralizedLSSFPN` | 同上，camera Group 指 neck，不是 BU01 backbone | bevfusion.compile.camera，options.mode |
| BU04 | `mmdet3d.models.fusers.conv.ConvFuser` | 同上 | bevfusion.compile.fuser，options.mode |
| BU05 | `mmdet3d.models.vtransforms.depth_lss.DepthLSSTransform` | 同上 | bevfusion.compile.depth_transform，options.mode |
| BU06 | `mmdet3d.models.heads.bbox.transfusion.TransFusionHead.forward_single` | 显式选择 `transfusion.py:transfusion_forward_single`，再交给 compile_wrapper | bevfusion.compile.transfusion，options.mode |
| BU07 | `mmdet3d.core.utils.gaussian.draw_heatmap_gaussian` | 编译适配后的同名 helper，忽略 original/options | bevfusion.gaussian；固定 mode、fullgraph=False、dynamic=False |
| BU08 | `mmdet3d.core.utils.gaussian.gaussian_radius` | 同上 | 同上 |

recipe 开启上述 Group，training.compile_mode 与四个类及 TransFusion 的 mode 均设为 `max-autotune-no-cudagraphs`；包装器默认也是该值。BU02–BU06 未显式传 dynamic/fullgraph。BU07/08 固定 mode 为相同字符串，不接受 options.mode 覆盖。Gaussian aliases 是同一目标的引用，不增加编译目标数；gaussian Group 依赖 `mmdet3d.gaussian`。

类、实例和 forward 方法不是可随意互换的包装对象。`compile_class_wrapper` 注释声称其符合来源中的类装饰器行为；这里只确认代码把类传给 torch.compile，实际支持、类身份/状态修改、实例范围及 guards 必须按目标 PyTorch 构建验证，不把注释当跨版本保证。

来源：[Group 注册](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion/catalog.py#L232)、[包装器](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion/compile.py#L10)、[backbone 初始化入口](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion/training.py#L468)、[Gaussian](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion/gaussian.py#L23)、[recipe](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion/configs/recipe.yaml)。

## 三个显式 eager 边界

| 边界 | 注册与实现 | 与编译开关的关系 |
| --- | --- | --- |
| `mmdet3d.models.utils.transformer.PositionEmbeddingLearned.forward` | compile.transfusion Group 调用 `torch._dynamo.disable(original)` | disable 包装器不读取全局关闭变量 |
| `mmdet3d.models.utils.transformer.FFN.forward` | 同上 | 同上 |
| 深度概率 Softmax | depth_factorization Group 替换 `DepthLSSTransform.get_cam_feats`；`depth.py` 将 `_softmax_impl` 包装成 `torch.compiler.disable` 并缓存，由 materialized_softmax 调用 | 不读取该全局关闭变量，属于另一个 Group 的边界 |

这是源码显式选择的边界，不证明位置编码、FFN 或 Softmax 普遍不适合编译，也不证明独立计时收益。私有 disable 与公共入口的版本兼容性需核对。局部 eager 会影响 fullgraph 与融合，不能一边要求完整捕获一边忽略主动图外执行的设计。

来源：[方法 disable](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion/compile.py#L53)、[Softmax 边界](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/bevfusion/depth.py#L124)。

## 包装顺序与关闭路径

TransFusion 包装器忽略 prepare 时捕获的 original，主动导入改写后的 `transfusion_forward_single` 再编译。源码注释说明 engine 先准备所有 Group，再应用；这样避免后续 compile Group 恢复旧函数。此处确认的是包装器选择及注释，未通过真实 engine 运行验证顺序。

| 入口 | 准备时全局变量为 1 | 准备时启用后再修改变量 |
| --- | --- | --- |
| 通用方法／类包装 | 返回传入对象，不创建新包装 | 已返回对象不会逐次检查变量；类若已被修改，返回同一类对象也不保证解除编译 |
| 相机 backbone | 返回当前 model，不新增该次 backbone 包装 | 不自动移除先前包装；重复调用训练入口需检查重复包装 |
| TransFusion | 返回改写后的 forward_single，不是传入 original | 不自动恢复基线 |
| Gaussian helpers | 返回改写后的 helper | 不自动恢复基线；固定参数也不因 options 改变 |
| 显式 eager 边界 | 仍创建 disable 包装 | 不因全局变量切换撤销 |

以上全局变量均为 `TURBO_PHYSAI_DISABLE_TORCH_COMPILE`。它阻止相应准备步骤创建编译，不是保证清除全部编译的总开关；Group 关闭和环境变量关闭也不是同一操作。默认使用独立进程切换，检查调用链、类/实例身份与实际状态。包装创建及执行未捕获编译异常，不能描述为自动 eager fallback。

Gaussian wrapper 的 aliases 与 TransFusion 内部 CPU helper 名称不同；`transfusion.py` 还定义并调用 CPU 版本的 Gaussian 工具，不能仅凭 catalog 注册推断 BU07/08 在当前训练目标生成中实际命中。实际配置、调用链和 Trace 必须另核对。

## 最小验证与证据

- 在同一重构后实现上比较 eager/compiled，再按需比较原实现；冻结布局、精度、数据及其他 Group，不能将组合收益归为 compile。
- 校验类、实例、方法的实际匹配；新增包装、已有包装、重入和独立进程关闭分别检查。别名覆盖不等于实际调用覆盖。
- 覆盖 Gaussian 半径/边缘中心、原地 heatmap 修改、输出及梯度契约；TransFusion 的改写与 eager 区域单独确认。此处只列验证要求，不执行计算重构。
- 分开判断配置开启、实际捕获、正确性通过和性能改善。重新采集 Trace 做诊断，无 profiler 正式测速记录冷启动、稳态吞吐及峰值显存。

本次证据为固定快照静态核对与轻量包装控制检查；没有真实 engine、PyTorch/GPU 编译、模型数值、梯度和性能结果。recipe 的 validated 描述不能转写为本次验证通过。
