# TurboPhysAI channels-last 源码机制

静态读源码所得，路径均相对 `turbo_physai/optimizations/`（除非另写）。没有运行模型，没有性能、精度数据；"未验证"表示原因或效果来自注释或推断。各模型实现独立，不能互相套用。固定提交：

- MapTRv2：[zhaobx01/TurboPhysAI](https://github.com/zhaobx01/TurboPhysAI/tree/bc8ced4a4b55e106579502edcb00b47b124f3fe4) `bc8ced4`，目录 `models/maptrv2_optimization/`。
- BEVFormer、BEVFusion、Cosmos3：[iMountTai/TurboPhysAI main](https://github.com/iMountTai/TurboPhysAI/tree/bb302a6363a06582e303f052c743469c55af65bf) `bb302a6`。
- OpenWAM：[iMountTai/TurboPhysAI feat/openwam](https://github.com/iMountTai/TurboPhysAI/tree/e4528c392096d2ad490e716af6f39ee3603af103) `e4528c3`。
- OpenVLA：[xgbah/TurboPhysAI feat/openvla-optimization](https://github.com/xgbah/TurboPhysAI/tree/e7aaad0202cf2fb7ccf8082765bfba2057e00517) `e7aaad0`。

## MapTRv2

- 开关：`TURBO_PHYSAI_CHANNELS_LAST`（`configs/runtime.yaml:36` 设为 "1"，`training.py:121-122` 读取，代码默认 True）；作用域 `TURBO_PHYSAI_CHANNELS_LAST_SCOPE`，取 `model` 或 `backbone`，`runtime.yaml:37` 设为 `model`，其他值抛 ValueError（`training.py:42-51`）。
- `model` 作用域对整个模型做 `model.to(channels_last)`，会触及所有 4D 参数；`backbone` 作用域只转 `img_backbone`（`training.py:54-62`）。
- 输入边界：给 `img_backbone` 装 `forward_pre_hook`，仅当第一个输入为 4D 且尚非 channels_last 时调用 `contiguous`（`training.py:71-85`）。模型没有 `img_backbone` 时整段静默跳过（`:123`）。转换发生在 `original(...)` 之前（`:133`），DDP 包装的具体位置未核实。
- ROCm：`PYTORCH_MIOPEN_SUGGEST_NHWC=1`（`runtime.yaml:13`）、`MIOPEN_FIND_MODE=1`（`:17`）、`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1`（`:29`）。
- 算子约定：DCN 按输入是否 channels_last 选 memory_format，并把 input/offset/mask/weight 对齐到同一格式，反向同理（`turbo_physai/operators/modulated_deform_conv.py:220-238,276-299,463-466`）。GridSampler 仅当 input、output（反向为 grad_output、grad_input）全为 ChannelsLast 才走 NHWC kernel（仓库根 `kernel/grid_sample/GridSampler.cu:615-616,703-705`）。两者在训练中是否实际命中未核实。
- 命名陷阱：`bev_pool_fix.py` 中的 `channels_last` 指 BEV 池化输出的轴排列（`[B,Z,H,W,C]`），不是 torch memory_format。
- `TransposeImage`：`note.md:79` 描述为把输入转成 channels_last 的 pipeline；`README.md:171-173` 称 grep 只命中定义处、没有 config 引用，是未接线的实验代码（我只读了 README 的说法，没有自行全仓库搜索）。

## BEVFormer

- 输入：扁平化后的 `(B*cams,C,H,W)` 图像在 grid_mask 之后、`img_backbone` 之前无条件 `to(channels_last)`，没有开关（`models/bevformer/backbone.py:39`）。整段 `compiled_extract_img_feat` 被编译时，这个转换落在编译区域内；`TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1` 返回 eager 函数（`backbone.py:73-80`）。neck 输出不转换。
- 整模型：runtime_hcu 路径无条件 `model.to(channels_last)`（`training.py:242`），与输入转换是两个独立机制；是否触及非卷积 4D 参数未验证。
- 算子：hipDNN 可变形卷积 forward 返回 NHWC buffer 的 `.permute(0,3,1,2)` 视图（`mdc.py:325`，fake 实现 `:504`），wrw 中强制 `grad_output` 为 channels_last（`:342`）。`msda.py:90` 只检查 `grad_output.is_contiguous()`。
- ROCm：`PYTORCH_MIOPEN_SUGGEST_NHWC=1`（`configs/runtime.yaml:15`）、`MIOPEN_FIND_MODE=1`（`:16`）；没有设 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION`。

## BEVFusion

- 输入：先 `view(B*cams,...)`，再由 `MMDET3D_CHANNELS_LAST` 决定是否 `contiguous(channels_last)`（`models/bevfusion/backbone.py:51-52`）；`_env_flag` 未设置时为 False，`runtime.yaml:11` 设为 "1"。
- 权重：`_convert_conv2d_weights_to_channels_last` 只转 Conv2d，用 `weight.data = weight.data.contiguous(...)` 原地替换（`training.py:455-466`）；开关 `options.channels_last` 或 `MMDET3D_CHANNELS_LAST`，默认 True（`training.py:505`，`recipe.yaml:27`，`optimization.yaml:119`）。Conv3d 与 sparse conv 不转；与优化器、DDP 的顺序未核实。
- 编译边界：只编译 camera backbone，`fullgraph=False, dynamic=False`（`training.py:473-487`），输入转换在编译区域之外。
- ROCm：`PYTORCH_MIOPEN_SUGGEST_NHWC=1`（`runtime.yaml:10`）、`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1`（`:14`）。
- 命名陷阱：`depth.py:191` 的 `permute(0,2,3,1).contiguous()` 是为 BEV pooling 准备的逻辑 NHWC，不是 memory_format；`bev_pool` 的布局契约未读。

## Cosmos3

- 总开关：`training_channels_last_enabled()` 在 HIP 构建上自动为真（`common/pytorch/memory_format.py:21`），`vae/layout.py:43` 据此得到 `_USE_CHANNELS_LAST`，没有单独的用户开关。融合 HCU causal Conv3d 需 `COSMOS_WAN_VAE_CLEAN_FUSED_OP=1`，默认关（`vae/fused_ops/_hipdnn.py:47-51`）。
- 权重：生成器的 Conv2d/Conv3d 权重由 `apply_channels_last_conv_weights(net)` 转换（`omni_model.py:80`），分别对应 channels_last 与 channels_last_3d；调用早于 FSDP 包装的说法来自调用点上下文，包装顺序未逐行核实。VAE 在 `__init__` 中对 Conv2d/Conv3d 模块分别 `module.to(memory_format=...)`（`vae/wan_vae.py:44-49`）。
- 激活：分块时 `x_chunk.contiguous(channels_last_3d)`（`wan_vae.py:241-242`）；`vae/layout.py` 提供 `_to_channels_last_2d/3d` 与 `_is_layout_contiguous`（`:46-58`）。Qwen 视觉 patch-embed 在 5D 输入进入 Conv3d 前调用 `to_channels_last`（`qwen.py:91,105`）。
- 融合算子契约：规范 NDHWC stride 为 `[C*T*H*W, 1, H*W*C, W*C, C]`（`_hipdnn.py:179-182`）。`fused_conv3d_bias_ndhwc`（`conv_bias.py:264`）要求 x、weight 均 `is_contiguous(channels_last_3d)`（`:282-283`）；另一个融合算子是 `fused_causal_cache_pad_conv3d_ndhwc`（`concat_conv_bias.py:344`）。其他条件来自源码：bf16、通道数在 {160,320,640} 且为 16 的倍数、不带 requires_grad。不满足时回退 `F.conv3d`，输入为 channels_last_3d 则输出也转回该格式（`conv_bias.py:132-133`）。
- 输入通道 12 的 stem 层被明确排除，源注释称为避免 hipDNN 选到退化 tile（`_hipdnn.py:160-176`，原因未验证）。
- 注册：`torch.library.custom_op` 加 `register_fake`，torch 低于 2.4 时用桩（`_hipdnn.py:69-82`）。
- AOT：`interface_compile_encode`（`wan_vae.py:327`）导出分块编码，按变体编译，分块与缓存在导出路径中用 `assert _is_layout_contiguous(...)` 保持 NDHWC（`:122-128`）。
- ROCm：`PYTORCH_MIOPEN_SUGGEST_NDHWC=1`、`MIOPEN_FIND_MODE=1`（`configs/runtime.yaml:11-12`）；没有设 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION`。

## OpenWAM

- 开关：`vae_channels_last`，默认 True（`models/openwam/settings.py:53`，`configs/optimization.yaml:145`）；`vae_pointwise_compile`、`compile_vae` 默认关，编译仅在 `global_compile` 范围含 vae 时进入。
- 权重：`prepare_model` 调用 `torch.nn.utils.convert_conv3d_weight_memory_format(vae, channels_last_3d)` 和 Conv2d 版本（`__init__.py:76-77`），在 DeepSpeed 之前一次性完成。
- 激活：`ports/vae.py` 的 `_layout`（`:16-22`）5D 走 `vae_layout.channels_last`，4D 走 `contiguous(channels_last)`；边界覆盖 `CausalConv3d_forward`（`:32`）、`Resample_forward` 中 5D 与 4D 间的 `rearrange`（`:98-100`）、`ResidualBlock_forward`（`:116` 起）及 VAE 编码输入。
- 缓存：`_clone_cache` 把 feat_cache 条目 clone 为 `channels_last_3d`（`ports/vae.py:26-28`），缓存与激活同格式。
- 编译：`vae_layout.py:30-35` eager 用 `contiguous`，编译中用私有 `torch._inductor.inductor_prims.force_stride_order(x, [c*t*h*w, 1, h*w*c, w*c, c])`；docstring 称 `contiguous` 可能在融合图中消失（推断，未运行）。`launch.py:21` 的 `vae_layout.validate()` 在 `force_stride_order` 缺失时抛错。
- 图复用：`pointwise.py:85-93` 对 channels_last_3d 的 5D 张量用 `as_strided` 统一单元维的等价 stride，以共享编译图，随后 `mark_static`。
- ROCm：`MIOPEN_FIND_MODE=1`、NHWC、NDHWC 提示（`configs/runtime.yaml:9-11`，`runtime.optimized.yaml:11-13`）；没有设 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION`。

## OpenVLA

在 `models/openvla/` 中搜索 `channels.?last|nhwc|memory_format|layout|permute\(|Conv2d|patch_embed`，只命中 `vision_timm.py:73` 的 `self.patch_embed(x)`，它是为避开 Dynamo 问题而改写，不是布局转换。`configs/runtime.yaml:14` 仅设 `MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32`。结论限于该提交与目录，不代表其他分支。

## 未检查

- 这些路径是否在实际训练或推理中命中，包括 DCN、GridSampler、`bev_pool`、spconv 的布局契约。
- AMP 与 channels-last 的交互（所读文件里没有相关代码）；FSDP/DeepSpeed 与权重转换的先后顺序仅按调用点推断。
- MapTRv2 assigner 的 cudagraphs 与 channels-last 是否冲突；仓库内说明与配置并不一致，未运行核实。
- 各远端分支今天的 HEAD 是否仍等于上述提交；部分克隆为浅克隆。
- 所有性能、显存、精度结论。
