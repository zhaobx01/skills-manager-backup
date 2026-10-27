# MapTRv2 TurboPhysAI 静态案例：包装、helper 重建和独立编译

整理日期：2026-10-15，于2026-10-27按新提交重新核对行号与默认值；对应 Skill v1.10.0。本文完整随包提供，无需原始仓库、原电脑、其他 Skill 或联网。路径/行号与摘要是历史定位标签，不是要求读取的外部文件。

## 固定来源与证据范围

- 仓库：zhaobx01/TurboPhysAI；提交：`bc8ced4a4b55e106579502edcb00b47b124f3fe4`（v1.10.0 起；此前固定在 `6fdd1d96cecd5b9a3d3410f7ff2fadbb3a685980`）。不跟随 main 后续漂移。
- 两提交间对本案例有影响的变化：assigner.py 代码里的缺省值未变，但 runtime.yaml 新增了 `TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE: "1"`；catalog.py 与 reference.py 的行号整体移动；`configs/optimization.yaml` 已从仓库删除。compile.py 内容未变。
- 6fdd1d9 的 `configs/optimization.yaml` 曾声明 MapTR 基线 `e03f097abef19e1ba3fed5f471a8d80fbfa0a064`；bc8ced4 已无该文件，recipe 自身 compatibility 仍为空，不能把二者混为一谈，也不能在新提交上再引用该基线声明。
- 以下相对实现文件位于 `turbo_physai/optimizations/models/maptrv2_optimization/`。源码、注册和配置已作静态核对；本轮未运行仓库测试、真实模型、训练或 GPU 编译。
- 仓库 README 关于参考实现/通过测试的描述属于来源自述，不是本次测量。本 Skill 旧 MapTRv2 源码案例是另一快照，其19处装饰器模式不能被新 README 描述覆盖；两份记录保留各自证据。

本案例只研究直接 Compile 的三条路径。布局和静态 GT 契约只解释开关与调用背景；不扩充数据加载、DDP、匹配算法或其他优化的实施流程。

## 三条路径与计数

| 路径 | recipe/生成配置状态 | Compile 入口与控制 | 不应误读为 |
| --- | --- | --- | --- |
| maptrv2.compile | enabled=True，options为空 | catalog 的10个原生 target → compile.compile_wrapper；prepare时创建包装，声明逐次调用条件 | 全部10项均已命中/捕获/加速 |
| maptrv2.reference_boundaries | enabled=True，options为空 | 5个基线调用方 → reference 中重建实现 →8个 helper；prepare按mode缓存包装 | 5+8=13个额外编译边界，或关闭只影响编译而不影响重构 |
| maptrv2.assigner | enabled=True，options为空 | 静态替换路径中的assign，额外由独立环境开关 opt-in 编译 | static assigner开启就代表assign已经编译 |

10+8+1是潜在具名编译边界的分组，不是静态torch.compile调用点数、真实FX图数或当前运行命中数；assign的代码缺省不编译（`os.getenv(..., "0")`），但随包 runtime.yaml 把该开关设为 "1"，所以按 recipe 加载运行时配置时 assign 会编译，单独调用代码则不会。实际代码在三个位置发起 torch.compile：compile.py:44、reference.py:283（循环）、assigner.py:300。

## 10 个基线同名目标

catalog.py:135–149列出以下target；151–161行逐个注册同一compile_wrapper及runtime_condition。目标命名中的类/模块限定名必须保留，不能将其他文件的同名函数视为同一目标。

| ID | 完整 target |
| --- | --- |
| W01 | `projects.mmdet3d_plugin.maptr.modules.transformer.MapTRPerceptionTransformer.format_feats` |
| W02 | `projects.mmdet3d_plugin.maptr.modules.decoder.MapTRDecoder.forward` |
| W03 | `projects.mmdet3d_plugin.maptr.detectors.maptrv2.MapTRv2.extract_img_feat` |
| W04 | `projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.get_cam_feats` |
| W05 | `projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.get_mlp_input` |
| W06 | `projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_3d_pts` |
| W07 | `projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_2d_bbox` |
| W08 | `projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_2d_pts` |
| W09 | `projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.denormalize_2d_bbox` |
| W10 | `projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.denormalize_2d_pts` |

## 8 个 helper 与 5 个调用方

这些helper不是直接注册到基线的新增target；catalog.py:303–360包装已有调用方，reference.py内部重建阶段边界。_HELPERS清单在260–269行。存在同名helper不能证明干净基线也有这个符号。

| ID | helper定义（reference.py行） | 基线调用方 | 重建实现与helper调用 |
| --- | --- | --- | --- |
| H01 | `matmul_1`，45行 | `BaseTransform.get_geometry_v1` | `base_transform_get_geometry_v1`，helper lookup在324行 |
| H02 | `matmul_2`，62行 | `BaseTransform.get_geometry_v1` | `base_transform_get_geometry_v1`，helper lookup在327行 |
| H03 | `matmul_3`，76行 | `BaseTransform.get_geometry_v1` | `base_transform_get_geometry_v1`，helper lookup在335行 |
| H04 | `extract_metas`，86行 | `BaseTransform.forward` | `base_transform_forward`，helper lookup在415行 |
| H05 | `down_sample`，112行 | `LSSTransform.forward` | `ls_transform_forward`，helper lookup在447行 |
| H06 | `initialize_queries_and_bev`，121行 | `MapTRPerceptionTransformer.forward` | `transformer_forward`，helper lookup在495行 |
| H07 | `compute_decoder_predictions`，148行 | `MapTRv2Head.forward` | `head_forward`，helper lookup在570行 |
| H08 | `prepare_transformer_inputs`，218行 | `MapTRv2Head.forward` | `head_forward`，helper lookup在540行 |

BaseTransform.forward 的重建还带有5D BEV布局路径（bc8ced4 起 `_bev_pool_5d` 改为调用 `bev_pool_fix.normalize_bev_pool_layout`，并在已打补丁的 bev_pool 返回 5D 时直接使用）；因此整个reference_boundaries组的开/关不一定是仅改变torch.compile的对照。这里不展开布局优化或推荐其默认适用性。

## 参数、能力与准备/调用阶段

- compile.py:25–50：工厂先检查 `TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1` 和 compat.torch_compile_available，关闭或无API时返回同一original；否则读取options.mode，缺省为 `max-autotune-no-cudagraphs`，调用torch.compile(original, mode=mode)，返回转发参数的wrapped。
- 该wrapped本身不再次读取开关；catalog另声明runtime_condition。按该接口约定，逐次条件为False选择原始callable；真实引擎分派没有在本轮执行。直接调用已返回的wrapped不能代表经过引擎条件分派的行为。
- compat.py:19–28懒加载torch，仅检查torch.compile是否callable，并尊重全局关闭。它不测试设备、后端、编译器或数值兼容性。
- 工厂调用发生在prepare阶段，创建包装不代表第一次实际追踪/编译已经完成。源码没有捕获torch.compile创建或compiled首次调用的异常，不能称作出错自动eager回退。
- reference.py和assigner.py的直接compile也没有显式dynamic/fullgraph参数；不要借用OpenVLA实例的True/False配置，也不假定所有版本默认行为一致。
- 与普通compile_wrapper不同，reference._prepare_helpers只在全局关闭时提前返回；内部未检查torch_compile_available，也未捕获缺少torch.compile的异常。runtime_condition属于逐次调用门控，不能据此保证prepare阶段在无API环境也安全；缺少API时实际准备可能失败，应明确报告，不能承诺这组三路径全部自动降级。

## helper 缓存与活动 mode

reference.py:272–287使用模块级 `_COMPILED` 与 `_ACTIVE_MODE`：

1. _prepare_helpers在全局关闭时直接返回，不重置cache或active mode；否则读取options.mode（默认no-cudagraphs）并设置active mode。
2. 每个helper按 `(name, mode)` 创建/缓存compile包装；重复准备同一mode复用，切换mode创建另一组，切回可复用旧项。缓存的是包装，不能据此声称已存在后端图。
3. _helper按当前active mode查询；无键时返回普通_HELPERS函数。启用后再关闭若绕过引擎直接调_helper，旧包装仍可能被取出。
4. 多个工厂共享active mode；不能承诺每个调用方的配置完全隔离。应检查不同配置安装顺序、多实例和缓存保留，不能把多mode字典误解为所有调用者互不影响。

通过实际runtime_condition回到原始调用方，与让重建实现使用eager helper不同；前者可能同时绕过重构与布局逻辑。应报告实际路径，再决定A/B解释。

## assigner：独立 opt-in，不等于静态化开关

catalog.py:247–279的maptrv2.assigner声明替换assign、loss和_get_target_single，运行条件static_assigner_available还要求compile/dynamo API可用且静态化未关闭。这里只记录编译入口与条件，不展开GT静态化实现。

assigner.py:273–314：代码里 `TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE` 缺省为0；不等于1时直接调用_assign_no_grad。等于1且_COMPILED_ASSIGN为空时，编译_assign_no_grad，mode固定为max-autotune-no-cudagraphs；随后复用单一槽中的包装。该函数未接受可配置mode，也不按mode建立字典。

缺省值分两层看：6fdd1d9 的 runtime.yaml 没有该变量，所以"代码缺省"和"随包运行时配置"一致，assign 不编译；bc8ced4 的 runtime.yaml:43 把它设为 "1"，随包配置下 assign 编译，除非调用方自行覆盖。核对 assign 是否真在编译，要看进程里最终生效的环境变量，不能只读 assigner.py。recipe 注释还要求 maptrv2.ddp_static_graph 与 maptrv2.assigner 配对：静态图假设自动求导图固定，由 pad_to_static_list 提供。

assign函数本身不读取全局关闭；通过catalog条件才有额外门控。直接调用它与完整引擎分派不同。独立开关从1切回0会选择普通_assign_no_grad，但旧cache不会被清空，再启用复用；编译创建/执行异常也未在这里捕获。

## 场景与验收边界

| 场景 | 应检查/记录 |
| --- | --- |
| 初始化全局关闭或无compile API | 普通compile_wrapper返回original；helper准备被关闭时不新增cache；能力条件仅说明可用性 |
| 初始化开启 | mode是否传入、10个target是否安装、helper是否创建包装；安装不等于捕获 |
| 启用后全局关闭 | 工厂wrapped/直接helper lookup可能仍用旧包装；经运行时条件选择original的分派需另行验证 |
| 更换mode并切回 | helper缓存键和active mode变化、旧cache复用；多调用方共享状态的影响 |
| assigner默认/独立开启/再关闭 | 静态Group与编译opt-in分开；一次包装复用、普通调用返回及cache保留 |
| 创建/首次执行抛错 | 传播或失败；不能写成已自动恢复eager |

轻量替身仅能检验这些Python包装控制逻辑，不导入真实torch/模型，也不证明Dynamo图、引擎集成、GPU内存、精度或吞吐。仓库测试报告与本轮检查分别记录，不重复宣称其测试通过。

## 对照方式与迁移经验

先检查基线符号，绘制target→工厂→条件→调用方→helper图。全局关闭如果恢复原实现、同时关闭布局/静态路径，得到的是组合回退对照；若要归因compile本身，应冻结同一重构实现和其他优化，单独控制实际compile入口，并验证路径确实不同。无法隔离时只报告整组结果。

不将本例默认mode、私有Dynamo入口、静态GT或布局契约推广给其他项目；只迁移符号存在性、阶段分离、开关语义和缓存生命周期的检查方法。

## 固定来源（可选追溯，非运行依赖）

- [catalog.py](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/catalog.py)；采集内容SHA256：`133e7c23d034cac3bf693b666ba10886388af75cf145a48bbff11e101db1079a`。
- [compile.py](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/compile.py)；采集内容SHA256：`a0122678314a094b250b89221e621d4fe03f57ebe7b0449fa712bfe87b92735e`。
- [compat.py](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/compat.py)；采集内容SHA256：`5a1b2b8159f83f0fce3b94f992ee30eda9c29bb1ba8966ec30e4efdb06b62328`。
- [reference.py](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/reference.py)；采集内容SHA256：`8ee4767c0fa4ca9e3e62fcd0d6e0aa2cdbd12b701d367112bea1ce84c5a8d3c9`。
- [assigner.py](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/assigner.py)；采集内容SHA256：`c892a90795ca466fe9050b44e46ae531762aa0ed8dd0018fc926dd6f9fdb4ae4`。
- [configs/recipe.yaml](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/configs/recipe.yaml)；采集内容SHA256：`07e026b0b47521da9f138e1dfdc62ec34c1edf4cac1c734850fdee53dadf3a5a`。
- [configs/runtime.yaml](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/configs/runtime.yaml)；采集内容SHA256：`17f76d3093d1c1897a508dd04b0acbf6957a3f3dd3e046d9059642944ca8e080`。
- [README.md](https://github.com/zhaobx01/TurboPhysAI/blob/bc8ced4a4b55e106579502edcb00b47b124f3fe4/turbo_physai/optimizations/models/maptrv2_optimization/README.md)；采集内容SHA256：`53156b6d25300f4f5419139492b13b9cf2896dfbd9eeeb38ff901efe51a3d913`。

摘要仅用于历史标识，不需要原始文件或在线复核即可读取/使用本案例。
