# 来源与维护说明

版本：1.9.0；整理日期：2026-10-20。

## OpenWAM 与第七类问题（v1.9.0）

新增 [OpenWAM 静态案例](openwam-compile-case-study.md)，来源为 [iMountTai/TurboPhysAI，e4528c392096d2ad490e716af6f39ee3603af103](https://github.com/iMountTai/TurboPhysAI/tree/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam)。provenance 声明 OpenWAM 来源提交 bfce20b575a9b34aea385030fcb00a9eaf61fcb5、基线 1204dde1b9e9c232131fc99a3f2c6b78330e7f64；本次未独立对照完整上游，不将移植机制全部归为新增优化。

记录五类编译入口及必要配置、动态标记、eager、checkpoint 状态背景；补入 P1–P6 并新增 [P7：重复层复用与嵌套编译](compile-problem-patterns.md#p7)。源码核对发现 MoT 同时有 checkpoint 与 suppress_inner_attn_ckpt 条件下的非 checkpoint 编译分支，修正“只在 checkpoint 路径”这一过窄概括。

仅以固定快照、配置和源码提取替身验证控制逻辑；未运行模型、真实 PyTorch 编译、checkpoint backward 或 GPU 性能，不把共享包装称为唯一图。案例完整随包提供，公开链接不构成加载依赖；既有案例、HTA、UI 和用户提示词保留。

## 跨模型问题参考与 BEV 案例（v1.8.0）

新增 [按问题查阅](compile-problem-patterns.md)，连接不同模型经验，不合并快照、参数或证据。保留全部历史案例、HTA、UI 和用户提示词。

新增 [BEVFormer](bevformer-compile-case-study.md) 与 [BEVFusion](bevfusion-compile-case-study.md)，固定来源均为 [iMountTai/TurboPhysAI，bb302a6363a06582e303f052c743469c55af65bf](https://github.com/iMountTai/TurboPhysAI/tree/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models)。静态核对 catalog、recipe、包装器及调用点，未独立对照完整模型基线，未运行模型。BEVFormer 记录三个编译目标和现场 SCA 分桶；BEVFusion 记录八个目标与三个显式 eager 边界。别名不增加目标数，目标数不等于实际图数。

包装器、分桶及关闭路径仅以轻量替身检验控制逻辑；不证明真实 engine、目标 PyTorch 构建、GPU 数值、梯度、吞吐或显存表现。源码注释中的类编译行为、validated 描述均属于来源声明。公开链接用于追溯，不构成本地依赖。

## Cosmos3 编译案例（v1.7.0）

固定来源：[iMountTai/TurboPhysAI，bb302a6363a06582e303f052c743469c55af65bf](https://github.com/iMountTai/TurboPhysAI/tree/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3)。仅记录局部 AOT、减少重复编译及必要配置/失败背景，不扩展为其他独立优化。

按 provenance 对照了 [NVIDIA/cosmos-framework 基线 VAE 文件，9726697a83315540c6885baefd2fe353d9c74920](https://github.com/NVIDIA/cosmos-framework/blob/9726697a83315540c6885baefd2fe353d9c74920/cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py)：AOT 框架已存在；适配改变形状解包、缓存目录处理且未保留逐变体正确性检查，不将省略验证列为通用建议。适配来源提交 `291c4e231809273d850910f07e9cdb252e9bd244` 在同一公开仓库的文件入口返回 404，属于来源声明、未独立对照。

完整解释见 [包内案例](cosmos3-compile-case-study.md)。全部链接是追溯资料，不是加载依赖；本次源码对照和轻量替身不构成模型/GPU 数值、吞吐或显存验证。三个历史案例及 HTA 内容保留。

本文件是随包提供的来源记录，不是外部输入清单。用户流程的有效内容已归纳到本包中；使用、迁移、更新或验证本 Skill 不需要外部原件、原作者电脑或历史源码。所有项目名称及相对源码位置仅用于说明案例，不是文件查找指令。

## 用户流程

创作时参考了用户提供的两份流程文档，保留其任务目标、源码候选扫描、A/B/C 分类、最小重构、精度约束，以及数据搬运检查表的思路：

- Torch_Compile_Optimize_Skill.md：编译候选筛选流程。
- Data_Move_Optimize_Skill.md：数据搬运检查流程。

这些名称只保留来源含义，文件不随包分发，也不需要另外寻找或复制；不要求任何特定目录结构。

修正了以下过强或不完整的规则：不统一排除 forward、优化器、容器/循环、动态 shape、自定义算子；不认为 Tensor 数据驱动 Python 分支易于捕获；不认为 any/all、Stream、view 或融合库能无条件消除同步与拷贝。

加入 D 类“证据不足”，避免将静态推断写成实测。保留仅分析的用户意图，同时增加明确请求实施时才生效的模式。

## 官方依据

技术规则整理时核对了 PyTorch 2.14 文档；实际应用优先确认目标项目版本，不能把文档版本作为项目最低版本或自动升级要求。链接供有网络时按版本复核；离线不阻止静态分析，无法核实的版本特性明确标记待核对。包内已提供基本决策与验证规则，不需要先下载这些网页。

- [编译位置](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html)：高层边界、训练步骤、子模块、分布式包装与区域编译。
- [torch.compile API](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)：mode、dynamic、fullgraph 及适用条件。
- [常见图中断](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html)：数据相关控制流和标量提取等问题。
- [端到端教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)：预热、首次调用成本与同步计时。
- [Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)：view 与复制的区别。
- [异步内存传输](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)：pinned memory、non_blocking、方向与正确性条件。
- [禁用编译](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.disable.html)：显式 eager 边界；仍需按目标版本确认。

## v1.1.0 项目实践来源

历史整理时只读检查用户提供的“MapTRv2 优化快照”，并与“MapTRv2 基础快照”的对应文件比较。基础快照记录的 HEAD 为 e03f097abef19e1ba3fed5f471a8d80fbfa0a064；优化快照未发现 Git 元信息，不能假定它们构成完整或唯一的优化历史。该历史标识不用于获取仓库或检查安装环境。

[MapTRv2 案例](maptrv2-case-study.md) 逐项记录 19 个 compile 和 2 个 eager 边界、配置可达性、调用契约、源文件摘要及平台条件。用户报告该项目做过有效优化；本地未发现可关联的性能日志，本次没有执行目标环境或训练验证。

v1.1.0 增加阶段拆分、eager 岛、父子编译入口检查、精度装饰器顺序、padding 契约、配置实际可达性与混合优化归因。海光镜像及私有 options 仅作为案例条件，不需要安装或获取该镜像。

## v1.2.0 自包含迁移

删除原电脑盘符、用户名和绝对路径，将历史来源改为逻辑名称；MapTRv2 案例与历史摘要全部静态随包保存，不要求重新打开原项目或校验历史哈希。增加没有其他 Skill、离线和没有 GPU 时的基础处理流程。

名称、分析/实施模式和自动选择策略保持兼容。外部 Skill 只提供可选增强；运行模型仍使用用户目标项目自己的环境，不捆绑 Python/PyTorch/驱动，也不自动更改系统或环境。

## v1.3.0 OpenVLA 直接 Compile 案例

新增 [OpenVLA 静态案例](openvla-compile-case-study.md)：只提炼 LLM 逐层与 ViT 塔整体两类编译入口、调用时的类型/包装识别、参数及配置开关。checkpoint 的外层位置仅解释边界背景，不增加配套优化流程。

来源固定为 TurboPhysAI 的 e7aaad0202cf2fb7ccf8082765bfba2057e00517，比较基线固定为其配置声明的 OpenVLA c8f03f48af692657d3060c19588038c7220e9af9；不随分支更新漂移。案例内含完整说明与公开来源链接，链接只供追溯，不是运行依赖。

没有导入模型、运行 FSDP、执行仓库测试或测量 GPU 性能；源码关于 checkpoint/Dynamo 限制的解释保留为该项目的版本相关背景，不推广为所有 PyTorch 的限制。不增加其他优化项的实现指导，不承诺可单独抽取这两类 compile 代码即可运行完整 OpenVLA。

## v1.4.0 MapTRv2 包装式 Compile 案例

新增 [MapTRv2 TurboPhysAI 静态案例](maptrv2-turbophysai-compile-case-study.md)，来源固定为 zhaobx01/TurboPhysAI 提交 6fdd1d96cecd5b9a3d3410f7ff2fadbb3a685980。只纳入三条直接编译路径：10 个原生目标包装、5 个调用方承载的8个 helper 边界和独立 opt-in 的 assigner 编译。

本来源的生成配置声明基线 e03f097abef19e1ba3fed5f471a8d80fbfa0a064。它与此前 MapTRv2 本地优化快照是不同证据来源；旧案例的源码模式/行号记录保留不变，不根据新仓库 README 覆写旧案例。

说明完整随包提供，无需源码、原电脑或联网。新规则只提炼基线符号存在性、准备/调用分阶段、开关回退路径与缓存隔离；布局、静态契约和其他优化只作为解释直接编译的必要背景，不扩充为独立优化指导。仓库 README 自述的测试通过/权威参考不等于本次验证；本轮只做静态及轻量替身检查，不执行训练、真实模型测试或硬件性能验证。

## v1.5.0 按需获取协作项

主表新增7个固定提交的完整目录下载入口，并增加 [协作获取流程](companion-skills.md)。来源为 meta-pytorch/skills 的 15ea40020f7ab0ad6645f78dca3b5634383e93c9 和 obra/superpowers 的 bb92a77741419a4ab5f06e711a283343f1ada0c3。已核对入口存在、目录清单及入口直接链接的模板（writing-good-tests.md、code-reviewer.md）；不声称上游指令在所有环境兼容。

策略从旧版“仅复用，缺少直接降级”改为“任务触发时优先复用，缺少且环境允许则主动下载到用户目录、校验、读取并应用”。失败仍降级；基础内容不要求联网/额外 Skill，且不全量安装7项。禁止覆盖已有项或自动安装其他运行环境。来源核验不等于安装，本次不重新安装本机已有协作项。

历史版本关于无依赖的说明指基础加载与包内流程，不阻止v1.5.0的可选按需获取。三个案例保持原样，来源追溯链接不是下载授权清单；不将上游 Skill 的其他引用自动视为无限递归安装授权。

## v1.5.1 诊断专项学习材料

新增 [三个诊断Skill学习说明](diagnostic-skills-primer.md)，将前述profile-model、debug-graph-breaks和analyze-memory-snapshot的介绍精简为输入、产出、适用时机、能力边界及与本Skill的协作顺序。依据当前专项指令与用户已讨论内容整理，不加入新的诊断实现或运行依赖；不改变v1.5.0的按需主动获取策略。

## v1.6.0 HTA工具关联

新增 [HTA简明操作参考](hta-analysis.md)，在入口、学习文档、协作获取边界与验证规范中关联profile-model、已有HTA Python和专项输入。HTA不是第8个协作Skill，不新增自动依赖安装；同一批Trace优先复用有效结果，逐项记录API状态、警告和覆盖。

接口与单位核对基于官方仓库 [c023a08a24ed44b4dd9f691d0e3d04464f1e76e9](https://github.com/facebookresearch/HolisticTraceAnalysis/tree/c023a08a24ed44b4dd9f691d0e3d04464f1e76e9)，参见参考中的具体实现链接。该版本某些返回字段使用us而接口文字出现ns描述；以实际安装版本和Trace核对，不将该观察泛化到所有版本。分析库导入、解析、指标可靠性分别记录。

用户的六步工作流、报告章节、固定文件名和输出目录保留为调用提示词内容，不复制进Skill或设为运行依赖。分析模式允许离线工具和授权报告写入；模型运行、采集、修复及依赖安装服从当前任务授权。未提供Trace且当前Python没有HTA，本次不运行真实HTA分析，不声称数据指标或性能已验证。

## 设计边界与操作参考

这是指令型 Skill，不包含 PyTorch 环境安装器或固定模型 Benchmark。输入模型、设备、训练状态各异，测试与计时应按项目构建；不附带会被误认为通用正确的训练比较脚本。

专项协作使用 SKILL.md 中的名称和相应工作流，不硬编码本机依赖路径，不伪造 skill-to-skill 的 MCP 依赖。自动选择保持默认开启；用户的只读/实施指令始终优先。
