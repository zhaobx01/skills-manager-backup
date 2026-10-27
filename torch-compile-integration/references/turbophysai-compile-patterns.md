# 让代码可被追踪，以及包装顺序与 key 兼容

本文收录 TurboPhysAI 各分支里与 torch.compile 直接相关、但原有案例没有展开的两类做法。全部来自静态阅读源码，没有运行模型、训练或性能测试，不含任何加速数据。迁移前须在目标项目重新验证。

读取的提交：BEVFormer 与 Cosmos3 为 `bb302a6363a06582e303f052c743469c55af65bf`，OpenVLA 为 `e7aaad0202cf2fb7ccf8082765bfba2057e00517`，MapTRv2 为 `bc8ced4a4b55e106579502edcb00b47b124f3fe4`。路径均相对 `turbo_physai/optimizations/models/`。

## A. 让周围代码可被追踪

### A1. 外部 kernel 注册为带 fake 实现的 custom_op

| 来源 | 做法 | 注意 |
| --- | --- | --- |
| BEVFormer `msda.py` | `mmcv::ms_deform_attn` 与 `mmcv::ms_deform_attn_backward` 各用 `torch.library.custom_op` 注册，各自带 `register_fake`，再用 `torch.library.register_autograd` 连接反向 | 文件开头检查 `torch.library.custom_op` 与 `register_autograd` 可调用，缺少则直接报错，不静默退回 |
| BEVFormer `mdc.py` | `mmcv::modulated_deform_conv2d` 及其 backward 同样的结构 | 同上 |
| Cosmos3 `vae/fused_ops/concat_conv_bias.py`、`conv_bias.py` | `turbo_physai::wan_vae_concat_conv_bias`、`turbo_physai::wan_vae_conv_bias`，带 `register_fake` | 默认关，需 `COSMOS_WAN_VAE_CLEAN_FUSED_OP=1`；开启时若没有可导入的 hipdnn 绑定会报错；torch 低于 2.4 时 `_hipdnn.py` 提供不做注册的占位装饰器 |

迁移含义：自定义算子要被 Dynamo/AOTAutograd 当作一个节点追踪，需要 fake 实现给出输出形状与 dtype，需要 autograd 注册提供反向。这是从注册方式得出的推断，本次没有运行确认这些位置不再产生 graph break。fake 实现与真实 kernel 的形状、dtype 必须一致，须单独核对。

### A2. 为让周围代码进图而改造 eager 岛

- MapTRv2 `grid_mask.py`：官方实现用 `np.asarray(PIL)` 得到只读数组，再 `torch.from_numpy(...).cuda()`。源码注释称前者让张量不可写，周围代码被编译时会变成 graph break 或硬失败，后者把设备固定为 CUDA。替换版保留官方的掩码构造，只修这两点（拷贝出可写缓冲区、跟随输入设备），并在首次使用时交给 `torch._dynamo.disable`，原因是随机数与 Python 控制流追踪价值低。结论是：该函数本身留在图外，改造的目的是让它不破坏周围区域。
- OpenVLA `vision_timm.py`：timm 的 `get_intermediate_layers` 里对已有的 `set` 再做 `set(n)`，Dynamo 的内置 `set()` 处理在 torch 2.7.1 触发 `assert source is None`。源码注释称这是 Dynamo 内部限制，改用基于 list 的等价重写，不再对带 source 的 set 调用 `set(...)`。版本相关，换 torch 版本需要重新确认问题是否仍存在。
- OpenVLA `compile_fsdp1.py`：编译开启时 `MixedPrecision` 使用 `buffer_dtype=None`，源码注释称这是为避开 Dynamo 的 fake tensor `set_` dtype 报错。

## C. 包装顺序与 checkpoint key 兼容（OpenVLA FSDP1 路径）

- 逐层编译与 checkpoint 位置：模块 docstring 称基线的 `apply_activation_checkpointing` 会把 checkpoint HOP 放进编译单元内部，所以改为对每个 FSDP1 包装的 LLM 层单独 `torch.compile`，再把 `checkpoint_wrapper` 放在编译单元外层，顺序为 FSDP(layer)、torch.compile、checkpoint_wrapper(compiled)。视觉塔内部没有 checkpoint HOP，docstring 称整塔编译在那里是安全的。docstring 没有说明 HOP 在编译内部时具体会发生什么，本次也没有运行验证。
- 运行时解析 FSDP 类：编译分支构造 FSDP 时用 `_fsdp_class()` 在调用时查找，不用模块级绑定。源码注释称模块在 Group 安装前的准备阶段就被导入，模块级名字绑定的一定是未打补丁的类；`openvla.fsdp.prefetch` 组会把 `FullyShardedDataParallel` 替换成注入 `limit_all_gathers`、`forward_prefetch`、`backward_prefetch` 的子类，只有动态查找才能看到它，否则两组同时开启时预取会被静默丢弃（`compile_fsdp1.py` 的 `_fsdp_class`，预取组在 `fsdp_prefetch.py`，注册在 `catalog.py`）。
- 保存键名：逐层编译后 state_dict 键里出现 `._orig_mod.`。`save_checkpoint_wrapper` 在保存时去掉这段（`strip_orig_mod`），并从 `self.vlm._orig_mod` 取回原模块。加载端是否对称，本次未核对。

适用范围：依赖 FSDP1 的私有类型与包装属性，不能直接推广到 FSDP2 或其他框架。
