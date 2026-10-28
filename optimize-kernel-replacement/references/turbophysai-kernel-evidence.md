# TurboPhysAI 与 GR00T 算子替换证据

静态读源码所得，没有运行模型，没有性能或精度数据。"源码自述"指源码注释或文档的原话，不是我们的测量。路径相对 `turbo_physai/optimizations/`，除非另写。固定提交：MapTRv2（zhaobx01/TurboPhysAI main）`bc8ced4`；BEVFormer、Cosmos3（iMountTai/TurboPhysAI main）`bb302a6`；OpenWAM（feat/openwam）`e4528c3`；OpenVLA（xgbah/TurboPhysAI feat/openvla-optimization）`e7aaad0`；GR00T 为本地工作区快照（含未提交改动）。

## 厂商库的归一化 kernel（OpenWAM）

- 开关 `lightop_norm`，默认关（`models/openwam/settings.py:27`）。`models/openwam/norms.py` 在 `LayerNorm.forward` 与 `lightop_rms_norm` 里调用 `lightop.op` 的 `layernorm_forward_autograd`、`rmsnorm_forward_autograd`。
- 进入条件（`norms.py:25-34`）：CUDA 设备且为 HIP 构建、dtype 为 fp16/bf16/fp32、`x.shape[-1] >= 64`、参数与输入同 dtype 且连续。`lightop_ops()` 会检查库里确有这两个函数，否则抛 `RuntimeError`。
- 回退：编译中（且开启 `global_compile`）返回 None 走原实现；处于特定的选择性激活检查点表达式时也回退；`LayerNorm` 的 `normalized_shape` 与最后一维不符时走 `super().forward`。
- 没有 LayerNorm 与残差加法合并的实现。

## 逐元素表达式交给编译器（OpenWAM）

- `pointwise.py`：`_compiled(fn)` 用 `torch.compile(fn, dynamic=True, fullgraph=True)`，`lru_cache` 缓存；开关 `pointwise_compile` 默认关（`settings.py:25`）。覆盖的表达式有 `modulate`（`x * (1 + scale) + shift`，`ports/dit.py:88-91`）、`gate`（`x + gate_value * residual`）、`rms_norm` 的 `rsqrt` 形式、`rotate`。
- 嵌套保护（`pointwise.py:22-39`）：已处于外层编译中直接 eager 调用，源码注释称嵌套分派会让共享函数按表达式特化。
- 实数 RoPE：`ports/dit.py:94-102` 在开关 `rope_real`（默认开，`settings.py:26`）打开时调用 `pointwise.rotate`，用实部虚部分量做乘加；原版用 `view_as_complex` 与 `view_as_real`。两者的中间计算都用 float64（`pointwise.py:124` 与 `ports/dit.py:100`），所以它是重新表达，不是低精度加速；是否被融合取决于 `pointwise_compile`。
- `linear_bias2d`：把视频投影改走 bias2d addmm 路线，保留模块与 Parameter 不动；有 hook、自定义 forward 或编译中则回退原调用（`linear_bias2d.py` 文件头）。默认关（`settings.py:28`）。

## FlashAttention 私有接口的白名单（OpenWAM）

`_fa2_varlen.py`：仅在 HIP 上，对 `FlashAttnVarlenFunc.forward` 与 `backward` 的源码计算 SHA256，只有与内置的两个哈希之一匹配才使用私有的 `_wrapped_flash_attn_varlen_forward`；未知构建或私有 API 不可用时日志提示并回退公开入口。注释称"Private FA2 entry points are not a stable API"。相关开关 `fa2_padding` 默认开，`fa2_mask_fused` 默认关（`settings.py:49-50`）；源码另有校验 `fa2_mask_fused` 依赖 `fa2_padding`（`settings.py:158`）。

## HIP 上的 FlashAttention v2（Cosmos3）

`models/cosmos3/attention/flash2_hip.py`：
- 版本范围：CUDA 保持上限 `2.7.4.post1`，HIP 放宽到 `2.8.4.post1`（`:48-70`），`flash2_supported()` 据此重算。
- `_CosmosFlashAttnVarlenFunc`（`:75` 起）：自定义 `autograd.Function`，docstring 称用于避免 FA 内部依赖数据的 cuseq 裁剪；forward 断言私有的 `_wrapped_flash_attn_varlen_forward/backward` 存在，并对 `cu_seqlens` 做 `clamp`。HIP 构建下 `flash2_attention` 选择它，其他后端用库自带的 `flash_attn_varlen_func`（`:343-350`）。
- 后端列表：HIP 下 `get_backend_list` 返回只含 `["flash2"]`（`:529-530`）；CUDA SM90 下才是 flash3、cudnn、natten、flash2 的顺序。
- 重编译提示：`:330` 注释"This check introduces recompiles"，空 batch 的检查只在非编译状态执行。

## 融合优化器

- OpenVLA：`models/openvla/fusedAdamW.py` 对 AdamW 工厂做 `kwargs.setdefault("fused", True)`，调用方显式传入的值优先。
- BEVFormer：`models/bevformer/training.py:225-232` 要求 `optimizer.type=AdamW`，否则抛 `ValueError`，随后强制 `optimizer["fused"] = True`。
- OpenWAM：`ports/trainer.py:607` 在开关 `fused_adamw`（默认关，`settings.py:62`）时给 `torch.optim.AdamW` 传 `fused=True`；注释自述"One fused kernel per tensor group instead of seven foreach passes over the fp32 ZeRO partition; same AdamW update, rounding may differ slightly."
- MapTRv2：`models/maptrv2_optimization/optimizer.py` 用 Apex `FusedAdam`，开关组 `maptrv2.fused_adamw`。限制：`amsgrad`、`maximize`、`capturable`、`differentiable`、`master_weights` 为真时报错；`bias_correction` 必须为真、`adam_w_mode` 必须为真；状态里出现 `step`、`exp_avg`、`exp_avg_sq` 以外的字段报错；同一组内各参数 step 不相等时报错，不静默取其一（`:38-84`）。`state_dict` 转回普通 AdamW 格式，`load_state_dict` 再转成 Apex 格式，使 checkpoint 与标准 AdamW 互通。
- GR00T：训练配置默认 `optim: str = "adamw_torch_fused"`（`gr00t/configs/training/training_config.py:55`）。
- BEVFusion：本次在 `bevfusion/training.py` 里没有找到融合优化器设置。

## 注册为 custom_op 的外部 kernel

- BEVFormer `models/bevformer/mdc.py`：`import hipdnn`，用 MIOpen fusion graph 构造 DCN 的前向、权重梯度、输入梯度，按 shape 缓存图（`build_fusion_graph` 等，`:190` 附近）；环境变量 `ENABLE_MIOPEN_DEFORMER_CONV_FP16` 打开时把 input、offset、mask、weight 转为 fp16 计算（`:287-294`）；`BEVFORMER_DCN_DEBUG=1` 打印图缓存构建。同一文件把 `mmcv::modulated_deform_conv2d` 及其 backward 注册为 `torch.library.custom_op`，带 `register_fake` 与 `register_autograd`。
- BEVFormer `models/bevformer/msda.py`：`mmcv::ms_deform_attn` 及 backward 同样注册，文件头先检查 `torch.library.custom_op` 与 `register_autograd` 可调用。
- MapTRv2/共享层 `operators/multi_scale_deformable_attention.py`：`lightop` 缺失时模块仍可导入；`mmcv.msda` 组用 `lightop_available`（检查所需函数均可调用）做 `runtime_condition`，缺失则保留原 mmcv 扩展（`common/mmcv/catalog.py:28-33`）。
- Cosmos3 `vae/fused_ops/`：`turbo_physai::wan_vae_conv_bias` 与 `turbo_physai::wan_vae_concat_conv_bias` 为 custom_op，带 `register_fake`；开关 `COSMOS_WAN_VAE_CLEAN_FUSED_OP=1`，默认关。`fused_conv3d_bias_ndhwc`（`conv_bias.py:264-300`）的进入条件：融合开关开启、dtype 受支持（仅 bf16，`_hipdnn.py:158-161`）、x 与 weight 都是 `channels_last_3d` 连续、通道数通过白名单（输入通道在 {160,320,640} 且为 16 的倍数）、x/weight/bias 都不带 `requires_grad`；任一不满足则走 `_reference` 回退。`groups` 不为 1 或 stride、dilation 不为正会直接抛 `ValueError`（`:87-90`）。输入通道为 12 的 stem 层被显式排除，源码注释称是为避免 hipDNN 选到退化 tile（原因未验证）。`concat_conv_bias.py` 的融合版本我只确认了函数存在，未逐条核对其条件。

## 与正确性相关的后端选择（GR00T）

- `gr00t/model/modules/dit.py:38-57`：`GR00T_DIT_SDPA_MODE` 可取 `math` 或 `default`；未设置时，HIP 构建默认强制 `sdpa_kernel(SDPBackend.MATH)`，源码注释称"DTK/HIP default SDPA backends produce NaN gradients in DiT backward (bf16)"。这是为避免 NaN 而用较慢后端，不是提速。
- `qwen3_backbone.py:349-361`：`use_flash_attention`（配置默认 True，`gr00t/configs/model/gr00t_n1d7.py:49`）时尝试 `flash_attention_2`，导入 `flash_attn` 失败则警告并回退 `sdpa`。
- `qwen3_backbone.py:292-311`：把 transformers 的 `flash_attention_forward` 用 `torch._dynamo.disable` 包起来，并把同一对象在 `AttentionInterface` 注册表里一并替换，注释称只改集成模块会让 Qwen3-VL 仍调用旧引用。

## 没找到的内容

- 手写 Triton kernel：在上述四个 TurboPhysAI 仓库里搜索 `@triton`、`triton.jit`，无命中；GR00T 未单独搜索 Triton。
- LayerNorm 与残差加法合并、gated MLP 融合、fused cross-entropy：未找到。
- 针对单个 kernel 的参数调优代码：未找到；调优主要体现为环境变量和 `max-autotune` 编译模式。
- BEVFusion 的融合优化器：未找到。
- 以上均基于关键词搜索，不代表这些内容在仓库的其他位置不存在。
