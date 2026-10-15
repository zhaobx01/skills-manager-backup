---
name: torch-compile-integration
description: Analyze PyTorch source to choose torch.compile boundaries and insertion points, or implement and benchmark compilation when requested. Use for deciding which model, step, submodule, or tensor function to compile, adding compile switches, or comparing compilation scope. Route existing graph-break diagnosis to debug-graph-breaks when available.
metadata:
  version: "1.2.0"
---

# Torch Compile Integration

选择有证据支持的编译边界，以较小改动接入 torch.compile，并验证正确性和真实收益。保持模型算法、输入输出协议和用户要求的数值精度；不把“成功捕获图”等同于“性能提升”。

## 自包含与环境边界

本 Skill 的全部指令、验证规范和案例均在本文件夹内，以相对链接加载。无需原作者电脑的文件、MapTRv2 源码、个人流程文档、其他 Skill 或联网即可读取并开展静态分析。不要尝试定位或补齐案例的原始磁盘目录、历史文件和哈希。

分析对象是用户当前提供的项目；未提供项目时先说明所需输入，不以包内 MapTRv2 案例替代实际目标。运行编译/精度/性能检查仍需目标项目自己的 Python、PyTorch、后端及适用硬件，本包不捆绑或自动安装这些环境。普通阅读和静态分析不以 PyTorch/GPU 可用为前提。

## 工作模式

- **分析模式**：用户要求分析、审查、建议或仅给方案时，只读源码和现有结果，不修改项目，不擅自运行完整训练。可以输出建议 diff，但必须标注未应用。
- **实施模式**：用户要求添加、接入、优化实现并验证时，在其授权范围内修改代码、加入启停开关并运行适当检查；无需为普通可逆修改重复询问。
- 用户只是询问原理或方法时，回答问题，不自动修改项目。缺少运行环境时，仍完成静态分析，并区分已确认事实、推断与未验证项。
- 仅写“优化建议/筛选热点/分析 compile”时沿用分析模式；明确要求实现优化时采用实施模式。需要运行时，先采用短小且代表性的检查，不默认启动完整训练或穷举所有编译模式。

## 先理解真实执行路径

确认项目入口、模型构建与权重加载、设备迁移、训练/eval 状态、AMP、优化器、DDP/FSDP、checkpoint 和实际调用链。检查现有 compile 逻辑，避免重复包装。记录实际 Python、PyTorch、加速后端与硬件版本，不为方便接入而擅自升级环境。

从配置和 registry 确定实际模型类型、继承关系及覆盖方法；把“源码有装饰器”“配置可达”“运行时确实捕获/执行”“性能已验证”分开记录。扫描同名函数和继承实现时使用类限定名称，不能将未启用的变体计入当前配置的收益。

从项目自有源码建立函数/方法候选索引，排除第三方依赖、生成文件和未使用示例；标注扫描范围，不把静态搜索称为覆盖所有运行路径。按真实调用链和热点优先深入，避免逐个函数机械添加装饰器。

用户明确要求“所有 def”时，包含普通函数、async def、类方法及嵌套函数；给出项目文件、限定名称与行号。未深入的候选用排除原因或 D 类记录，不能把一小部分热点分析描述成全量逐函数结论。编译风险来自调用链和执行路径，不能仅用语法关键字计数判断。

有 profiler trace 时，按需使用可用的 `profile-model` 阅读它。没有 trace 时，只能给热点假设和采集建议；在用户允许的运行预算内收集小规模代表性数据。

## 选择编译边界

优先核对已安装 PyTorch 版本对应的官方说明；版本特定 API、模式、动态形状和分布式建议以该版本为准。链接是可选核对来源，联网不是加载前提。离线时使用包内规则及用户提供的版本信息继续静态分析；可用时检查目标环境的 help/签名/已有日志，不确定项标为“版本兼容性待核对”，不要求联网或自动升级环境：

- [Where to apply torch.compile](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html)
- [End-to-end tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)

比较实际相关的候选：顶层模型、单次训练/推理步骤、重复子模块、局部 Tensor 函数。选择能够稳定捕获且收益合理的较高层边界；图中断、重复编译、启动成本或内存预算不合适时再缩小。不要为了比较而强行实现所有方案。

大型框架项目可按几何变换、特征整理、查询初始化、预测解码、匹配代价等计算阶段寻找边界。对照调用者，检查是否拆出独立 Tensor 计算、隔离 CPU 匹配/随机增强，以及边界过细是否增加启动/编译开销或损失融合；函数短小本身不是编译理由。逐一梳理父子编译入口，不能假设嵌套装饰器必然产生独立编译区域或独立采用其 mode。

判断依据：调用频率、实测耗时、可融合算子、Python 开销、shape/stride/dtype/device 变化、数据相关控制流、状态修改、自定义算子注册、编译与部署成本。

不能把以下情况写成绝对禁令：

- 整个 forward 可以是合理候选，不要求所有代码完全静态。
- list/dict、静态 Python 循环、可守卫的 Python 分支不一定导致图中断。
- Tensor 值决定的 Python if/while、.item() 等可能触发图中断、同步或重复编译；具体行为需核对版本并实测。
- 动态 shape、优化器更新、随机算子、自定义 CUDA 算子和通信都需要具体判断，不能一律保持 eager，也不能保证兼容。
- 不以减少日志频率、改变随机种子、降低精度、改变分支或删除状态更新来换取表面加速。

输出候选表：

| 源码位置与符号 | 调用链/热点证据 | 分类 | 编译边界与插入位置 | 初始配置 | 风险与最小重构 | 验证方法 |
| --- | --- | --- | --- | --- | --- | --- |

分类：A 直接试编译；B 当前保持 eager；C 做小范围重构后试编译；D 证据不足待验证。A/C 是待测建议，不是收益承诺。

推荐一个首选方案，并说明其他相关候选为什么暂不选。若瓶颈主要是数据加载、搬运或通信，明确 compile 可能收益有限。

涉及模式选择、动态 shape、区域编译或 DDP/FSDP 时，按需阅读 [references/decisions.md](references/decisions.md)。出现 CPU/GPU 交换、同步或布局拷贝时阅读 [references/data-movement.md](references/data-movement.md)，沿用用户流程中的“位置/影响/方案”表；不要把数据搬运整改扩展到任务范围之外。

## 实施接入

沿用项目配置系统，提供可关闭的 compile 开关和必要的 mode 配置。默认行为及启用策略尊重项目约定和用户要求。使用该版本支持的入口；不要把某一版本的 model.compile() 或 torch.compile(model) 偏好推广到所有版本。

在合适的生命周期位置创建/启用编译入口，避开热循环中反复创建包装。验证权重加载、设备迁移、train/eval、保存恢复及参数对象行为。DDP/FSDP 的包装顺序按版本和现有框架核对，不能一套顺序通用。

检查 compile 与框架精度装饰器的从下到上包装顺序；保留 autocast、强制 FP32、梯度和随机增强语义。记录原地操作、输入 alias、缓存/buffer 首次初始化，以及训练/eval 和 shape 变化产生的 guards。为稳定形状而引入 padding、mask、length 或元数据打包时，先核对整个调用契约，覆盖空输入、容量上限、有效元素映射及任务指标，不能仅验证张量尺寸一致。

先建立 default 模式基线。有证据后再比较 reduce-overhead 或其他该版本支持的模式；不要盲选 max-autotune，也不要统一设置 dynamic=True、fullgraph=True 或关闭错误检查。fullgraph 可用于适当的诊断，不是必须达成的优化指标。

不默认开启 suppress_errors 或静默回退来伪装成功。若提供显式 fallback，报告触发条件、实际执行路径和测试结果。

## 按问题协同

其他 Skill 是可选增强，不构成安装或运行依赖。仅在目标设备确实提供且任务需要时读取；缺少它们时直接按下面的包内基础流程处理，无需安装它们。引用名称不会自动建立依赖，也不要求开启多个代理。

| 出现的情况 | 使用的现有 Skill | 本 Skill 接收的结果 |
| --- | --- | --- |
| 需要判断实际瓶颈 | profile-model | 热点、空闲时间、通信/数据开销及采集条件 |
| 加入 compile 后有图中断 | debug-graph-breaks | 具体原因、修复或边界缩小建议、复测证据 |
| 编译后显存异常或 OOM | analyze-memory-snapshot | 分配来源、生命周期、缓存/碎片等证据 |
| 普通错误、行为回归或构建失败 | systematic-debugging，或项目选定的调试 Skill | 根因与最小修复；避免重复套用多个通用调试流程 |
| 实施需要增加回归覆盖 | test-driven-development | 数值、梯度、启停开关与状态行为的针对性测试 |
| 准备声称完成 | verification-before-completion | 实际验证命令和结果，含未覆盖项 |
| 实施范围较大、涉及复杂状态或分布式 | requesting-code-review | 接口、状态、正确性及测试遗漏的审查结果 |

### 没有其他 Skill 时

- **热点分析**：读取已有 profiler trace/统计，区分 CPU 准备、传输、算子与通信耗时，查看调用次数和输入形状；没有 trace 时给出热点假设及有限采集方案，不能当作实测。
- **图中断/重编译**：在目标环境有条件运行时，以代表性小输入启用该版本支持的 graph_breaks/recompiles 日志，定位第一处具体原因，建立最小复现；选择修复、缩小边界或 eager 岛并复测，保持控制流语义。无法运行时只标记风险和待执行命令，不要求安装日志可视化工具。
- **显存排查**：先核对 shape/dtype、生命周期、allocated/reserved/峰值及分配栈或已有快照；区分活跃张量、缓存与可能泄漏。有快照且可解析时追踪增长位置，否则记录证据缺口，不把 reserved 增长直接判为泄漏。
- **普通排错与回归**：缩小复现，先对比 eager 行为，再检查最小改动、状态、梯度和开关；按 [references/validation.md](references/validation.md) 做针对性验证，不因缺少测试 Skill 停止。
- **完成检查**：实际运行适用的检查并读取结果，逐项列出通过/失败/未运行。复杂改动按调用契约、状态和回归覆盖做自查；如需独立审查可建议后续进行，不能把自查声称为独立审查。

离线、缺少其他 Skill 和没有 GPU 可以同时发生：仍完成可做的静态分析/已授权编辑，分别说明版本核对、运行正确性和 GPU 性能的缺口，不虚构通过结果。

没有证据需要自定义 Kernel 时，不转入 write-cuda-* 或 write-triton-*；若编译后仍有关键算子瓶颈，单独提出后续任务。

## 验证与交付

实施前阅读 [references/validation.md](references/validation.md)。分析模式可把其中内容作为验证计划。

交付包含：候选表、选择理由、实际改动与启停方式、环境和输入、正确性结果、冷启动与稳态性能、显存、图中断/重编译情况、运行命令及未验证项。

把编译、计算重构、布局、搬运、精度、通信和后端环境变更分列。优先在同一份重构后源码上对比编译开/关，必要时再与原始实现对比；禁止把混合优化版本的总收益全部归因于 compile。

只有目标环境中的实测数据才能支持加速结论。GPU 不可用、依赖缺失或任务预算不足时，明确标为“仅完成静态分析/代码接入，性能未验证”。没有收益或回归时如实报告，采用用户要求的保留/回退策略，不反复扩大修改范围。

维护本 Skill、了解流程来源与技术规则时，参见 [references/sources.md](references/sources.md)。该文件只是包内来源说明，不要求读取外部原件。普通任务只需加载与当前决策有关的参考文件。

需要复杂模型拆分、eager 岛、静态化契约或厂商后端实例时，按需阅读 [references/maptrv2-case-study.md](references/maptrv2-case-study.md)。这是完整随包提供的静态历史案例，不要求访问原项目；不是当前项目的执行输入，也不是必须照搬的优化配方。
