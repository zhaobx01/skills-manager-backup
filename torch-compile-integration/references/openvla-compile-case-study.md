# OpenVLA 静态案例：按子系统选择 Compile 粒度

整理日期：2026-10-15；对应 Skill v1.3.0。

本文完整随包提供，无需安装 TurboPhysAI、克隆 OpenVLA、访问原电脑或联网。路径与行号是固定提交的来源标签，不是必须存在的文件；用户当前提供的项目才是实际分析对象。

## 证据与范围

- TurboPhysAI：`e7aaad0202cf2fb7ccf8082765bfba2057e00517`，来自用户指定的 `feat/openvla-optimization` 分支快照。
- OpenVLA：`c8f03f48af692657d3060c19588038c7220e9af9`，由上述优化配置声明的基线，不以原仓库当前 main 替代。
- 本次读取并静态核对实现、注册和配置；未运行模型、FSDP、训练、性能测试或仓库测试。
- 两个 torch.compile 调用点不等于两个运行图，也不说明全部 LLM 层或视觉模块均已命中。

只纳入直接编译：两种粒度、参数、类型识别和配置驱动入口。checkpoint 外层位置仅作边界背景。BF16 修复、timm 改写、保存键名兼容、其他补丁协作、文本分桶、FA2、fused AdamW、通信、数据加载和 GC 不在本案例的实现指导范围内。

## 来源位置与接入关系

主要实现相对 TurboPhysAI 根目录位于 `turbo_physai/optimizations/models/openvla/compile_fsdp1.py`。其中：

| 编号 | 符号/位置 | 实际行为 | 条件与边界 |
| --- | --- | --- | --- |
| O01 | `_per_layer_compile_and_checkpoint`，137–152 行；compile 调用在147行 | 对符合条件的 LLM 子模块逐个调用 torch.compile，然后替换父模块中的该 child | 遍历 named_children；必须同时为 FSDP 实例，且其 `_fsdp_wrapped_module` 是传入 layer_cls 的实例；不匹配则继续递归 |
| O02 | `_compile_whole_vision_towers`，155–165 行；compile 调用在164行 | 对符合条件的 ViT 塔整体 compile，并替换 vision_backbone 中对应 child | 只检查 vision_backbone 的直接子模块；必须同时是 FSDP 实例且其 `_fsdp_wrapped_module` 是 timm VisionTransformer；不是任意视觉模块，也不是递归遍历全部视觉后代 |
| O03 | `_compile_options`，98–103 行；`run_setup_wrapper`，169–179 行 | 读取 options，关闭时返回 original，开启时提供新的 setup 方法 | compile 工厂默认 False；mode 默认 default；不是由环境变量决定该开关 |
| O04 | setup 内229–240行 | 传入 self.vlm 和 self.llm_transformer_layer_cls 执行 O01，再把 self.vlm.vision_backbone 传入 O02 | 必须结合该实例的类型、FSDP 包装层次和运行配置检查实际命中范围 |

catalog 中 `openvla.compile.fsdp1` 以 wrap 方式注册到 `prismatic.training.strategies.fsdp.FSDPStrategy.run_setup`。本文只研究其直接编译路径；该 Group 还有本文不展开的配套成员，不应把本案例当作完整可独立运行的移植脚本。

读取的基线 `prismatic/training/strategies/fsdp.py` 中没有 torch.compile 调用，新增入口来自上述 setup 包装；这一结论不推及基线整个仓库的任意其他文件。

## 编译参数与粒度

O01、O02 的直接调用均相当于：

```python
torch.compile(child, mode=compile_mode, fullgraph=False, dynamic=True)
```

该语句只展示编译入口，不是独立脚本。child 已由项目自己的类型选择和包装过程提供。

- LLM：以符合条件的 FSDP1 层为单位；源码中启用梯度 checkpoint 时，checkpoint_wrapper 位于 compiled child 外层。这是该实例边界背景，不是对所有版本和模型的统一顺序要求。
- 视觉：以符合条件的 FSDP1 包装 ViT 塔为单位，而不是为每个 ViT block 分别添加 compile。
- 迁移经验：同一模型中的重复结构与独立子系统可以采用不同粒度；需要比较捕获稳定性、启动/重编译成本、融合与实际收益，不规定所有 LLM 逐层或所有视觉塔整体。
- dynamic=True 尝试动态形状处理，不保证无特化或重编译；fullgraph=False 允许分图，不证明关键部分一定捕获。它们都是本例配置，不覆盖 Skill 的通用决策规则。
- helper 依赖 FSDP1 的特定类型与私有包装属性；不能直接推广到 FSDP2、不同框架或其他包装层次。

## 配置的两个默认层级

| 层级 | 记录的状态 | 可以据此确认什么 |
| --- | --- | --- |
| 包装工厂 | 缺少 compile 选项时 False；mode 为 default；关闭则返回原 run_setup | 该工厂具备原始方法返回路径，不代表其他优化组都被关闭 |
| recipe.yaml 39–43行与生成的 optimization.yaml 39–43行 | `openvla.compile.fsdp1` 为 enabled=True，options.compile=True，options.mode=default | 随包配置请求开启，不证明用户实际运行使用了该配置 |
| 运行实例 | 本次无模型运行记录 | 不能确定匹配数量、是否已首次调用、图数量、fallback 或性能结果 |

因此，必须区分：Group 被选中 → 工厂开关为真 → 子模块类型命中 → 实际执行并捕获 → 实测获得收益。setup 的完成日志本身没有给出两个 helper 的命中数量，不能代替覆盖检查。

## 四种场景应如何分析与验证

| 场景 | 静态可确认 | 在目标环境中还应验证 |
| --- | --- | --- |
| 关闭开关 | run_setup_wrapper 返回原 original，不执行该新 setup 的两个 helper | 实际配置是否关闭、其他路径是否仍有 compile；不把其他优化组保留的行为当成原始完整基线 |
| 开启开关 | 两个 helper 被接入 setup，使用上述参数和识别条件 | 目标类、包装层次、实际匹配数量、代表性调用、图中断/fallback及前后向结果 |
| 无匹配子模块 | 两个 helper 没有符合条件的 child 时没有 compile 调用；视觉 helper 不自动深入不匹配的子树 | 记录 0 与原因，检查模型结构变化；不能用正常结束或扩大匹配范围掩盖未生效 |
| 输入形状变化 | 代码显式指定 dynamic=True，但没有覆盖所有动态形状行为的证据 | 不同 batch/文本长度等代表性输入下的 guards、重编译、内存与稳态性能 |

这张表是验收要求与静态推论，不是已经运行的测试结果。没有目标环境时明确保留未验证项。

## 收益归因与未纳入的配套项

同一 recipe 还包含独立优化，compile Group 本身也有额外配套行为。即使同一源码切换该 Group，也不天然构成只改变两处 compile 调用的纯对照。

评价直接 Compile 时，应列出开关前后共同变化，必要时在相同模型与配套行为下单独控制直接编译入口，冻结输入、状态、精度和环境；先确认这种隔离可行，再做测量。若只能测整组或整套方案，就只报告整组/整套结果，不把总提升归因于 O01/O02。

本文不提供那些配套优化的实现、修复步骤或通用默认设置，也不承诺把这两个调用复制到其他项目即可运行。

## 公开来源（可选追溯）

- [直接编译实现](https://github.com/xgbah/TurboPhysAI/blob/e7aaad0202cf2fb7ccf8082765bfba2057e00517/turbo_physai/optimizations/models/openvla/compile_fsdp1.py)
- [Group 注册](https://github.com/xgbah/TurboPhysAI/blob/e7aaad0202cf2fb7ccf8082765bfba2057e00517/turbo_physai/optimizations/models/openvla/catalog.py)
- [recipe](https://github.com/xgbah/TurboPhysAI/blob/e7aaad0202cf2fb7ccf8082765bfba2057e00517/turbo_physai/optimizations/models/openvla/configs/recipe.yaml)
- [生成配置与基线约束](https://github.com/xgbah/TurboPhysAI/blob/e7aaad0202cf2fb7ccf8082765bfba2057e00517/turbo_physai/optimizations/models/openvla/configs/optimization.yaml)
- [OpenVLA 基线 FSDPStrategy](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/training/strategies/fsdp.py)

以上链接无需下载即可使用本文；若另行研究更新提交，那是新的源码分析任务。
