# OpenWAM：共享 block 编译与内外层分派

本文件是完整随包提供的静态案例。无需原电脑文件、OpenWAM/TurboPhysAI 安装或联网即可阅读；实际项目验证仍需自身环境。按问题查阅见 [跨模型参考](compile-problem-patterns.md)。

## 固定来源与证据

来源：[iMountTai/TurboPhysAI，feat/openwam 固定提交 e4528c392096d2ad490e716af6f39ee3603af103](https://github.com/iMountTai/TurboPhysAI/tree/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam)。以下文件相对该目录。

`configs/provenance.json` 声明来自 iMountTai/OpenWAM 的 `bfce20b575a9b34aea385030fcb00a9eaf61fcb5`，适配基线为 `1204dde1b9e9c232131fc99a3f2c6b78330e7f64`。本轮仅检查 TurboPhysAI 固定快照，未独立对照完整 OpenWAM 来源和基线；不将移植机制全部称为本仓库原创改动。

## 五类直接编译入口

| 编号 | 位置与对象 | 配置条件及缓存 | 显式参数与范围 |
| --- | --- | --- | --- |
| OW01 | `global_compile.py:compile_mot_layer` 的局部 layer 函数 | compile_mot 派生 global_compile/mot scope；driver._compiled_layer 首次创建后复用 | dynamic=False；fullgraph=not fragmented；options=_options()；无显式 mode |
| OW02 | `global_compile.py:compile_vae_encoder`，`vae.model.encoder.forward` | compile_vae 派生 vae scope；prepare_model 找到原生 VAE 后替换 forward | dynamic=False、fullgraph=True、options=_options()；无显式 mode，不替换模块对象 |
| OW03 | `pointwise.py:_compiled(fn)` | pointwise_compile 且第一个参数 is_cuda；按 fn 使用无界 lru_cache；受 SAC 优先分派及外层编译分支影响 | dynamic=True、fullgraph=True；RMS、modulate、gate、rotate 等 Tensor 表达式 |
| OW04 | `pointwise.py:_compiled_vae_norm(dim, scale, rank, apply_silu)` | vae_pointwise_compile 且 is_cuda；按四项 key 使用无界 lru_cache；外层编译时直接执行表达式 | dynamic=True、fullgraph=True、按支持项传精度 options；normalize 或 normalize+SiLU |
| OW05 | `attention.py:_compiled_nonempty()` | fa2_mask_fused；lru_cache(maxsize=1)，从有效 FA2 padding 路径调用 | dynamic=True、fullgraph=True；编译 torch.where 的输出选择表达式 |

五类是不同工厂/入口，不是五个实际图。OW03/04 会按函数或配置产生多个包装及特化；OW01 共享 callable 也可能因模块类型、参数、状态和输入 guards 生成多个图。源码没有为这些入口显式传 mode，不得照搬 BEV 的 mode。

`_options()` 仅在目标 Inductor 暴露 `emulate_precision_casts` 时传 True，意图保留 BF16 表达式边界的舍入；不保证数值已正确，也不是跨版本必有的参数。OW04 的 rank 是包装缓存 key 的一项，不是完整运行兼容性证明。

来源：[较大入口与动态标记](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/global_compile.py)、[小函数及归一化](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/pointwise.py)、[输出掩码](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/attention.py#L180)。

## MoT 可达路径与共享入口

`ports/mot.py` 的 Driver 初始化设置 `_compiled_layer=None`。同一 driver 之后复用这个包装，具体 block 对象从 ModuleList 在图外查出后作为参数传入；局部 layer 调用 `_step_impl(..., layer_id=0, _blocks=(video_block, action_block))`，用显式 blocks 而非固定第零层进行计算。目的在于减少按层索引特化，不证明所有层共享一张图。

有两条入口，不能只记 checkpoint 一条：

| 路径 | 到达条件及行为 |
| --- | --- |
| `_step_checkpointed` | step 在 use_gradient_checkpointing 且 action backbone 为 training 时调用；global_compile 开启且 scope 含 mot 后选择 compiled callable |
| 非 checkpoint 的 `step` | suppress_inner_attn_ckpt 且 global_compile，并且 scope 含 mot 时也可选共享 compiled callable；否则走 _step_impl |
| `run_joint_loop` 调度 | checkpoint 层数为总层数减 checkpoint_skip_layers；尾部 partial 层可设置 suppress_inner_attn_ckpt。需核对训练配置、实际层区间及调用者，不能直接称为“只编译 checkpoint 层”或“全部层都编译” |

两条编译分支都拒绝 `vstate.extras['vace']` 非 None 的路径，源码错误提示限定原生 Wan/no-VACE。不同架构、直接调用 step、训练/eval 和层区间均需核对实际可达性。

来源：[Driver 两条分支与循环](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/ports/mot.py#L138)。

## 动态标记与 eager 边界

- `mark_mot_text_dynamic` 在 text_trim 开启时，进入编译前标记视频/动作 context 的 axis=1 和 mask 的最后一轴；None、长度 0/1 和已标记轴跳过。dynamic=False 的工厂参数与显式轴标记并存，不能将其解释成所有尺寸静态或所有尺寸动态。
- OW04 在特定 channels_last_3d 条件下规范视图 stride，并将归约维度、gamma 及 Tensor bias 标为静态。这里只解释编译 guards 的意图；as_strided 的存储、alias、布局和数值必须单独验证，不提供通用布局整改流程。
- SAC `_invoke` 对特定区域输出，在原输入带动态轴、输出维数/对应长度相等且长度>1 时传递标记；不说明任意 eager 变换都保留动态性。
- OW01 的 fragmented 条件是 mot_split_attn 或 fa2_padding 或 SAC active；据此选择 fullgraph。`_compiled_attention` 是否 disable 又有独立条件：mot_split_attn 或 attention_boundary，不能混为同一个开关。
- FA2 padding 在全局编译且正在捕获时可转入显式 eager；SAC `_invoke` 也通过 disable 边界调用。它们与 fullgraph 选择需一起核对，不为追求完整捕获删除行为。

来源：[SAC 分派与标记传递](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/sac.py#L323)、[Attention eager 边界](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/attention.py#L195)。

## 内外层分派与 checkpoint 状态

OW03 `_run` 先检查 extras_enabled：成立时走 SAC pointwise 分派；否则在 is_compiling 且 global_compile 时直接执行原 Tensor 表达式，交由外层捕获；只有其余满足设备/开关条件时才调用独立编译缓存。OW04 外层捕获时同样直接执行 normalize 表达式。不能把“避免嵌套编译”写成所有内层路径一律禁用。

SAC dispatcher 通过私有 `torch._dynamo.eval_frame.skip_code` 避免薄分派器作为独立 frame 编译；缺少入口时警告并返回原函数。配置初始化后 launch 会重新处理已标记的 dispatcher。其注释陈述了减少 callable guards 特化的动机，具体行为需版本验证，不推荐其他项目无条件复制私有 API。

checkpoint `_run` 对 vstate、astate、payload 做浅拷贝，将参与重算的 Tensor 字段写入局部容器，返回后再更新外层状态引用。这隔离容器字段赋值，不隔离内部 Tensor 的原地数据修改。需用真实 forward/backward、重算及参数更新验证；本案例不展开 checkpoint 层数或 SAC 策略调优。

## 配置冻结、初始化与消融

| recipe | pointwise_compile | compile_mot | vae_pointwise_compile | compile_vae | fa2_mask_fused |
| --- | --- | --- | --- | --- | --- |
| 基础 recipe.yaml | false | false | false | false | false |
| recipe.online.yaml | true | true | true | true | true |
| recipe.cached.yaml | true | true | true | true | true |
| recipe.cache-build.yaml | true | false | true | true | true |

`settings.py` 从实际已安装且携带 Group 标记的 wrapper 收集 options，不从优化环境变量读取。未选择 Group 不激活其布尔优化；已选择 Group 合并默认值和 options，初始化检查后冻结，force 初始化另有入口。global_compile 与 scope 由 compile_mot/compile_vae 派生，不是用户可任意独立设置的 Group 参数。

`prepare_model` 在训练准备/DeepSpeed 前处理 VAE；找不到原生 VAE 则警告并跳过。OW02 仅替换 encoder.forward，不是 Cosmos3 的 torch.export/AOTInductor 包，也没有 .pt2 变体分派。cached 配置仍可开启 compile_vae，但缓存 latent 是否绕过实际编码要沿当前数据路径核对，准备成功不证明运行命中。

来源：[设置与冻结](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/settings.py)、[准备模型](https://github.com/iMountTai/TurboPhysAI/blob/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/__init__.py#L44)、[四份 recipe](https://github.com/iMountTai/TurboPhysAI/tree/e4528c392096d2ad490e716af6f39ee3603af103/turbo_physai/optimizations/models/openwam/configs)。

`ports/joint.py` 的 `run_joint_loop` 路径里有一段 try/except：若 `self._compiled_mot_run_joint_loop` 非空且未开启梯度 checkpoint，则调用它，失败时置 None、记录警告并回退 eager 的 `driver.run_joint_loop`。但在该提交的整个仓库里，这个属性只在此处读取和置 None，没有任何位置赋成可调用对象，所以这条回退目前不可达，不能当作已验证的 fallback 能力；MoT 层与小函数编译的创建和执行，本次读取未发现通用的异常回退，需自行核对。

关闭 compile_mot/compile_vae 不自动关闭 OW03–05，也不移除已经安装的 forward、driver/lru 缓存或私有 dispatcher 标记。使用新进程和明确配置做消融，保持其他计算改写、精度、布局和 checkpoint 条件一致；全部组合关闭与纯 compile 对照不是一回事。编译创建/执行未提供通用异常 fallback，不把报错称为自动回退。

## 最小验证与当前证据

| 检查项 | 目标项目要求 |
| --- | --- |
| 可达与命中 | 各 Group 实际安装、四份 recipe、架构、两条 MoT 分支、层区间及 VAE 数据路径 |
| 复用 | 同 driver 的不同层、多个 driver、形状/模块类型变化及缓存生命周期；编译 wrapper 数与实际图数分开 |
| 动态性 | 文本 0/1/多长度、可选 mask、已有动态标记、eager 输出标记传递及 VAE 静态归约维 |
| 状态与训练 | 容器引用、Tensor 原地行为、checkpoint 重算、输出/loss/梯度/参数更新与保存重载 |
| 性能 | Trace/日志确认实际执行，关闭 profiler 和诊断后测准备成本、稳态、端到端吞吐与峰值显存 |

本轮证据仅为固定快照静态核对与源码提取的隔离控制检查；未独立运行真实 engine、PyTorch 编译、模型、GPU 数值、checkpoint backward、吞吐或显存测试。源码中的意图与日志文字不是已实测结论；共享 callable 不等于共享唯一图。完整规范见 [验证参考](validation.md)。
