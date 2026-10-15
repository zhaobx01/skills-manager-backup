# 来源与维护说明

版本：1.2.0；整理日期：2026-10-15。

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

## 设计边界

这是指令型 Skill，不包含 PyTorch 环境安装器或固定模型 Benchmark。输入模型、设备、训练状态各异，测试与计时应按项目构建；不附带会被误认为通用正确的训练比较脚本。

专项协作使用 SKILL.md 中的名称和相应工作流，不硬编码本机依赖路径，不伪造 skill-to-skill 的 MCP 依赖。自动选择保持默认开启；用户的只读/实施指令始终优先。
