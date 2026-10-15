# 编译边界与配置决策

优先确认用户目标项目的 PyTorch 版本及实际平台，有条件时按对应版本官方文档检查支持范围。没有运行环境时可读取项目依赖声明并标记实际版本未确认；离线时按本文继续分析，把无法核对的 API/后端行为列为待验证，不要求下载文档或安装环境。以下是候选方向，不能当成所有版本和硬件的性能承诺。

## 比较边界

| 候选 | 适合考虑的情况 | 必须检查 |
| --- | --- | --- |
| 顶层 nn.Module / forward | 主要计算在模型内部、调用路径可捕获 | train/eval、buffer/随机性、shape、hooks、自定义算子 |
| 单步训练/推理函数 | 希望跨模块、loss 或优化器操作优化 | 排除无关 IO/外层迭代调度，检查 backward 和状态更新支持 |
| 重复子模块/区域 | 全模型编译时间太长，或外围控制流复杂 | 是否重复复用、是否触发重新特化、是否牺牲跨边界融合 |
| 局部 Tensor 函数 | 日志/CPU 调用等包围着独立计算热点 | 拆分是否保持异常、状态、梯度、alias 和执行顺序 |
| eager | 没有热点证据、收益被启动成本吞没或不兼容 | 记录原因；若只是证据不足，标 D 类而非宣称不可编译 |

当已安装版本支持并推荐时，顶层 nn.Module 可使用其 .compile() 接口；函数可使用 torch.compile 包装。不要假设所有版本都具有同样的接口细节。已有训练框架提供编译入口时，优先使用它，避免多处包装同一调用链。

## mode / dynamic / fullgraph

| 设置 | 使用依据 | 代价或注意点 |
| --- | --- | --- |
| default | 初始对照配置 | 不是保证最快，需要实测 |
| reduce-overhead | 小 batch、Python/启动开销突出且 CUDA Graph 适用 | 可能增加显存；输入修改、CPU 操作等会影响适用性，检查 perf_hints |
| max-autotune | 目标设备支持、矩阵计算占比高且运行足够长 | 增加调优启动成本；CUDA Graph 与内存约束仍需检查 |
| max-autotune-no-cudagraphs | 有理由做 autotune 但不使用 CUDA Graph | 不保证一定节省内存或优于 default |
| dynamic 默认策略 | 先观察代表性输入变化 | 通常从默认策略开始，发现重复编译后再定向调整 |
| dynamic=True/False | 输入分布和日志证明需要 | True 不能保证无重编译；False 可能为新 shape 特化 |
| fullgraph=True | 严格捕获要求或最小范围诊断 | 不是速度开关；会改变捕获要求，语义细节需按版本核对 |

不要为了动态 shape 直接 padding/bucketing 而忽略 mask、损失、梯度和输出协议；这类重构需要单独验证。

## 分布式与自定义算子

检查现有框架的 DDP/FSDP/DTensor 流程和版本文档，再决定 inner module、子模块或训练步骤的边界。记录参数 identity、优化器构建、权重加载、包装与编译的顺序；不得仅凭一段通用代码重排现有生命周期。

自定义 CUDA/Triton 算子并非天然不能 compile；检查注册、fake/meta 实现、autograd、mutation/alias 和动态输出 shape 的契约。若是三方扩展不支持，提出有依据的局部 eager 边界，不借机将项目迁移到另一构建或分发平台。

非 Tensor 工作可以保留在外层；不要因“Python 出现在函数中”就否定整个 forward。Tensor 值依赖的 Python 控制流需看实际跟踪行为；torch.cond 或张量化方案需满足目标版本限制及原始语义，不能机械替换。

## 阶段拆分、eager 岛与嵌套入口

优先识别可独立验证的连续阶段：例如几何矩阵变换、embedding/位置编码准备、特征格式整理、decoder 预测和匹配代价。每次拆分记录输入输出、状态、异常、梯度与调用次数；小函数要有调用频率/融合证据，不能只按装饰器数量评价优化程度。

把 SciPy 匹配、NumPy/PIL 增强等保留在 eager 岛时，检查 CPU 往返、结果对象和状态更新的成本。按安装版本使用 torch.compiler.disable 等受支持入口；项目已有 torch._dynamo.disable 只作为兼容性线索。禁用区域与 fullgraph 要求可能冲突，先明确任务目标。

建立 parent → child 编译入口图，区分独立 eager 调用和外层追踪时的内联行为。不能认为“父子都装饰”意味着双重收益，或子函数的特殊模式一定被外层尊重；依据实际日志/图与计时决定保留哪一层。

## 精度、状态与静态化契约

- 装饰器从下到上应用。分别检查 compile(force_fp32(fn)) 与 force_fp32(compile(fn)) 实际包装的代码及目标输入 dtype，避免随手换序。
- 带静态循环、容器追加、ModuleList 索引的计算可作为候选；需验证循环次数、配置分支及 train/eval 改变时的 guards。
- 输入 squeeze_、切片赋值、共享 view、缓存首次构建等可能影响重放、alias 或特化。保持原语义，并单独覆盖首次和后续调用。
- padding 必须记录容量、pad 值、mask、length 及消费者。测试 N=0、1、容量-1、容量、容量+1；明确超限行为，不截断有效数据，不移除原始空输入处理来追求静态图。
- 对匹配/排序等离散决策，除数值容差外，检查有效索引、padding 过滤和 ties 的任务允许行为；输出差异可能放大为训练目标变化。
- 元数据转换、numpy 输入和 .to(device) 的实际位置应追踪到被调用函数，不以 helper 名称判断“纯 Tensor”。

## 厂商后端与配置隔离

保存 torch 构建标识、后端/驱动、设备、容器标签和实际启动命令。不要因为厂商实现复用 torch.cuda 命名就推断是 NVIDIA CUDA。

max-autotune-no-cudagraphs 是特定 mode 候选；显式 triton.cudagraphs/cudagraph_trees、布局优化、调优环境变量和通信开关另列配置。内部 options 是否存在、是否生效及其组合由目标构建决定，不把其他平台的私有配置硬编码进新项目。

厂商平台上的精度/TF32 类开关、通信重叠和 NUMA 绑定与 compile 分开归因；不改变这些条件来制造编译收益。
