# Cosmos3：局部 AOT 编译与减少重复编译

这是包内静态案例，解释完整随包提供；公开链接只用于追溯。使用本 Skill 不需要安装 Cosmos3、克隆仓库、读取原电脑文件或联网。下述内容为源码经验，不是本次完成的精度或加速验证。

## 先用简单的话理解

- **局部编译**：外层继续切分视频、循环处理、拼接结果；只提前编译“处理一个视频片段”的内部计算。每次把上一个片段的状态缓存传进去，再接收更新后的缓存。
- **减少重复编译**：某些长度参数变化会产生更多编译版本。对符合契约的参数分桶，并让编译器知道真实的尺寸关系，尝试提高已编译版本的复用率。
- 两者不是固定搭配。普通项目仍先比较 `torch.compile` 边界；只有部署方式、重复计算和准备成本适合时才考虑 export/AOTInductor。AOT 不等于普通 `torch.compile` 的一个 mode。

## 固定来源与继承关系

检查快照：[iMountTai/TurboPhysAI，bb302a6363a06582e303f052c743469c55af65bf](https://github.com/iMountTai/TurboPhysAI/tree/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3)。表内路径均相对该仓库，且只作定位说明。

[provenance](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/configs/provenance.json) 声明基线为 `9726697a83315540c6885baefd2fe353d9c74920`、适配来源为 `291c4e231809273d850910f07e9cdb252e9bd244`。

本次取得并静态对照了 [NVIDIA/cosmos-framework 基线 VAE 文件](https://github.com/NVIDIA/cosmos-framework/blob/9726697a83315540c6885baefd2fe353d9c74920/cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py#L1238)：局部 AOT、变体枚举、多卡分配和权重绑定已经存在，不是 TurboPhysAI 新发明。适配来源提交在该公开仓库的文件入口返回 404，故该提交仍属于“来源声明，未独立对照”；不能据此认定提交不存在。

对照已取得的基线可确认：基线 `_collect_warmup_shapes` 使用 `W, H` 解包，当前适配使用 `H, W`；基线初始化会同步重建缓存目录，适配改为创建目录并允许复用已有文件；基线加载后有逐变体正确性检查，适配未保留这道检查。前两项的运行正确性仍需目标环境验证，省略正确性检查不作为推荐优化。

`_ChunkEncodeForAOT` 等对象不是在此适配文件中独立实现，而是通过 `_source_binding.bind_missing_globals` 从原始 `cosmos_framework` 模块补齐。基线 wrapper 将输入 chunk 和缓存转交 `_encode_chunk_impl`，并持有归一化参数。包内案例解释这一关系，但不要求读者访问原模块才能理解。

## 源码与配置对应表

| 编号 | 源码位置与符号 | 做法与条件 | 证据边界 |
| --- | --- | --- | --- |
| C01 | `cosmos3/catalog.py`：`VAE_COMPILE`；`configs/recipe.yaml` | `cosmos3.vae_compile` 替换 `_collect_warmup_shapes`、`Wan2pt2VAEInterface.compile_encode`；依赖 `cosmos3.vae_layout`；recipe 开启两组 | 注册及配置已确认；不证明调用了 compile_encode |
| C02 | `cosmos3/vae/wan_vae.py`：`wan_vae_encode_chunk_impl` | encoder → conv1 → 拆分 mu/log_var → 归一化；输入 chunk、特征缓存和 scale，输出特征与新缓存 | 只编译局部计算；不是全模型或反向传播 |
| C03 | 同文件：`interface_compile_encode` | `torch.export.export(strict=False)` → `torch._inductor.aoti_compile_and_package` → `.pt2` → `aoti_load_package`、`load_constants` | 导出、加载和数值正确性需分别验证；kwargs 不作为普通 torch.compile 参数使用 |
| C04 | 同文件：`interface_compile_encode` 的变体循环 | 标准 cache_t=0/1/2；可追加 exact-duration 尾块；按 rank 轮转分配编译，汇总包路径后各 rank 加载 | 多节点需验证包路径可见性；不是无需准备开销 |
| C05 | 同文件：`wan_vae_inner_encode`、内部 `_run_chunk` | `(T_chunk, H_patch, W_patch, cache_t)` 查表；命中调用 AOT，未命中调用 `_encode_chunk_impl` | key 未覆盖 batch/dtype/device/layout/权重身份等，不能假定任意输入均可命中并执行 |
| C06 | `cosmos3/sequence_packing.py`：`_bucket_flash_attn_max_seqlen`；`mot_attention.py`：`_fa_max_len` | 桶为 1024 至 65536 的二倍序列；真实长度字段保留；只有 HIP＋正在编译＋存在 fa_* 字段时读取桶值 | 分桶的是特定 launch 参数，不是通用 padding；不保证只有一个图 |
| C07 | `cosmos3/mot_attention.py`：`multi_control_two_way_attention` 及内部 `_sdpa` | `_check` 声明分区末端、KV 等长及正长度；构造累计长度，绕过 frontend 的 max/item 参数推导路径 | 约束依赖实际分段契约；不能用断言掩盖合法空输入 |
| C08 | `cosmos3/compile_options.py`；`common/pytorch/compile_options.py` | 最小 recompile_limit=64、cache_size_limit=512、累计上限=4096；duck shape 默认关闭；按版本尝试设置内部配置 | 不创建编译入口；设置缺失时跳过，不代表参数生效或重编译减少 |
| C09 | `cosmos3/attention/flash2_hip.py`：编译状态检查分支 | 一处空批次检查只在非编译状态执行，注释称其引入 recompiles | 不是删除检查的通用许可；必须核对空输入支持与替代检查 |

定位链接：[注册依赖](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/catalog.py#L197)、[局部计算](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/vae/wan_vae.py#L105)、[AOT 分派](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/vae/wan_vae.py#L219)、[编译准备](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/vae/wan_vae.py#L327)、[长度分桶](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/sequence_packing.py#L15)、[形状契约](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/mot_attention.py#L295)、[受限检查分支](https://github.com/iMountTai/TurboPhysAI/blob/bb302a6363a06582e303f052c743469c55af65bf/turbo_physai/optimizations/models/cosmos3/attention/flash2_hip.py#L330)。

## 局部 AOT 如何工作

1. 外层接受视频，处理时间补齐和 patchify；第一帧单独建立因果缓存，后续按固定窗口逐片段执行。特定 exact-duration 输入保留尾块，最后拼接并按原长度裁剪。
2. 准备阶段按分辨率和宽高比枚举形状。cache_t=0 表示首次无缓存，1 表示过渡状态，2 表示稳态；尾块另外枚举。不能把这些数字当成其他模型通用的缓存状态。
3. 参考缓存由 eager 核心函数生成；导出使用 batch=1、BF16、cuda 命名设备和特定布局的示例。厂商后端也可能使用 torch.cuda 命名，不由此推断设备品牌。
4. 编译包不把常量一起封入 so；加载后绑定现有 encoder 权重。需验证目标 PyTorch 构建的私有 API、权重生命周期、包兼容性和内存所有权。
5. 运行时仅查表分派，不逐片段重新创建编译包装。缓存状态显式输入输出；函数仅浅拷贝缓存列表，这不隔离其内部 Tensor 数据或所有原地副作用。

这条编码路径由 no_grad 入口调用，wrapper 在准备阶段 eval。它不证明可训练 VAE 的 backward、优化器或参数更新兼容。适配加载完包后会清理缓存目录，不能描述为默认提供跨进程持久编译缓存。

### 失败与关闭路径

| 情况 | 快照中的行为 | 应验证什么 |
| --- | --- | --- |
| 未安装 AOT 函数表／形状未命中 | 走局部 eager 核心 | 同样切块、状态及布局条件下输出一致 |
| 某变体导出或编译失败 | 警告，返回 None，继续其他变体 | 完整记录成功覆盖范围，不把失败当成功 |
| 某包加载或权重绑定失败 | 警告，不加入 loaded_fns | 失败形状是否走预期 eager 路径 |
| 没有任何可加载函数 | 末尾抛 RuntimeError | 不能声称全部失败自动回退 |
| 已命中 AOT，但执行抛错 | `_run_chunk` 不捕获该执行异常 | 不声称执行异常会自动 eager 回退 |
| Group 关闭或整套组合回退 | 实际行为取决于 engine 与原始 Cosmos；基线本身也有 compile_encode | 重启后的真实调用链和函数表；关闭 Group 不等于禁止全部编译 |

复用现有 `.pt2` 时，文件名主要反映形状与缓存状态，没有完整标识代码、后端、版本等条件。初始化不清空旧目录可能复用中断任务残留；不能把文件存在当作兼容性证明，也不要照搬目录清理到用户共享路径。

## 可迁移的决策方法

- 外层调度复杂而核心重复计算稳定时，比较局部 `torch.compile` 与按需 AOT；记录准备成本、覆盖率、权重和状态约束。不能只因存在循环就拆分，也不能默认 AOT 更快。
- 优先从重编译日志找出具体 guards。分桶只用于允许取上界的参数，保留真实有效长度及消费者；覆盖桶边界、零、负值和超上限。输入张量形状和其他 guards 仍可能产生新图。
- `_check` 只能表达已成立的契约，不能为满足编译器人为收窄用户输入。绕过 item 的替代路径需保留 attention 的累计偏移和有效 token 语义。
- 缓存上限和跳过分支只作为诊断背景。先核对目标版本实际生效值，再评估版本数量、内存和准备成本；不通过移除必要检查制造捕获成功。
- 将编译、分桶／契约重构和布局分别归因。尤其 VAE_COMPILE 依赖 VAE_LAYOUT，单纯切换整个组合不构成同一代码的纯 eager/AOT 对照。

## 验证要求与当前证据

目标项目验收应覆盖：首次和连续片段、缓存重置、多个宽高比和非方形输入、尾块、batch/dtype/layout、空／无效输入契约、未命中形状、部分和全部失败、权重重载。对比最终特征及每层缓存；不只对比输出 shape。

编译关闭／开启需在相同重构和布局下比较，并记录实际 AOT 命中率、加载变体数及 eager 回退。重新采集代表性 Trace 看执行路径；正式测速关闭 profiler 和诊断日志，分别测准备成本、局部稳态、端到端吞吐和峰值显存。对于普通 torch.compile 路径另看 graph_breaks/recompiles，不把 AOT 变体查表称为 Dynamo 重编译。

| 证据 | 本次状态 |
| --- | --- |
| 固定 TurboPhysAI 快照、配置、分派与失败处理 | 静态源码确认 |
| 原始 Cosmos 基线 VAE 机制与差异 | 已取得固定文件并静态对照；未覆盖整个原始仓库 |
| provenance 的适配来源提交 | 公开原始文件入口不可取得，未独立对照 |
| 隔离轻量替身 | 可检查分桶与分派控制，不证明实际编译、数值或硬件兼容 |
| 模型运行、数值、梯度、实际命中、吞吐、显存 | 本轮未运行，不能声称通过或提速 |

已有通用规范仍见 [决策参考](decisions.md) 与 [验证参考](validation.md)。
