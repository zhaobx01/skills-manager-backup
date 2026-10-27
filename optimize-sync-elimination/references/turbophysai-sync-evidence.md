# TurboPhysAI 跨模型同步消除证据

以下来自静态读源码（本地克隆的固定提交），不代表运行时一定命中，也不代表正确性或性能已验证。"源码中的做法"表示改写本身已核实；带"源码自述"的句子是源码注释或文档里的原话，不是我们的测量；没有任何性能数字——所有数字均未重新测量。

固定提交：MapTRv2（zhaobx01/TurboPhysAI main）`bc8ced4`；BEVFormer、BEVFusion、Cosmos3（iMountTai/TurboPhysAI main）`bb302a6`；OpenWAM（iMountTai/TurboPhysAI feat/openwam）`e4528c3`；OpenVLA（xgbah/TurboPhysAI feat/openvla-optimization）`e7aaad0`；GR00T 为本地工作区快照（含未提交改动）。

## 已核实的案例

### MapTRv2

- **损失归一化留在设备上**：三个 `*_head.loss_single`（`maptrv2_head`、`maptr_head`、`bevformer_head`）把正样本计数的 `.item()` 换成 `.reshape(())`：`num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).reshape(())`（`sync_elimination.py:119,255,376`）。改写前的 `.item()` 做法见同文档 `v5_GroupA.md` 的 A2 一节描述。
- **删除布尔 mask 的冗余 `nan_to_num` 清洗**：`BEVFormerEncoder.point_sampling` 构造的 `bev_mask` 是 `torch.bool`，不可能含 NaN/Inf，所以原本按 PyTorch 版本分支调用的 `nan_to_num`（Torch 路径或旧版 NumPy 路径）被整段删除，只保留投影计算和可见性判断（`sync_elimination.py` 中 `bevformer_point_sampling`，对照 `v5_GroupA.md` A1）。`v5_GroupA.md` 自述：旧版 NumPy 分支"还涉及设备往返"；实机记录里 profiler 确认优化路径没有 `aten::nan_to_num` 事件，方法级微测墙钟中位数约 2.004 ms 对 1.998 ms，文档明确说"此微小差异仅为该环境的单次测量记录，不作为稳定加速承诺"。
- **安装方式**：四个替换作为一个原子组安装，任一目标检查失败则整组不装（`v5_GroupA.md` 开头）。
- **必须留在 CPU 的求解**：`assigner.py:142` 的 `_hungarian_match_impl` 对代价矩阵 `cost.detach().cpu()` 一次性搬运，`linear_sum_assignment` 的结果再一次性 `torch.as_tensor(..., device=device)` 搬回。这是 P7 模式，不是消除，是收敛成一次。
- **优化器 step 标量**：`optimizer.py:31` 的 `_integer_step` 在校验 checkpoint 的 step 值时调用 `.item()`，这是加载时的一次性检查，不在训练热路径上，不建议作为消除对象。

### BEVFusion

- **loss 聚合单次 all_reduce 加单次 `cpu().tolist()`**：`parse_losses`（`training.py:39-64`）把所有标量 loss 堆成一个张量，做一次 `dist.all_reduce`，最后一次 `reduced.cpu().tolist()`，取代逐个 loss 分别 `.item()`。
- **Hungarian 匹配单次 D2H/H2D**：`transfusion.py:505-519` 的 assign 函数对代价矩阵 `(...).detach().cpu()`，`linear_sum_assignment` 结果 `np.stack(...)` 后 `torch.from_numpy(...).to(bboxes.device)` 一次性搬回，再 `.unbind(0)` 拆成两个索引张量。同 P7 模式。
- **BEVPool 用 `where`/`index_select` 替代布尔索引**：`common/mmdet3d/bev_pool.py` 的 `NativeQuickCumsumCuda.forward`（约 36-50 行）用 `interval_starts = torch.where(kept)[0].int()` 得到索引；`quick_cumsum_forward`（约 84-96 行）用 `kept_indices = torch.nonzero(kept, as_tuple=False).flatten()` 配合 `torch.index_select` 读写，而不是 `x[kept]` 式的布尔索引。"改写前"是推断（未看到对应的旧版实现），按 P3 模式收录，标注为"源码中的做法"而非对照验证。

### BEVFormer

- **SCA 预取路径把 `.max().item()` 移到数据侧**：`geometry_sca.py:277-296` 的 `encoder_forward` 先尝试 `_prefetched_sca_inputs`；命中时直接用预取的 `indexes`/`index_lengths`，不执行现场的 `int(index_lengths.max().item())`（`:285`）。预取侧对应的 `_prepare_bev_meta_group`（`data.py:260` 附近）在 CudaPrefetchLoader 的后台线程里用 numpy 数组算出同样的分桶结果，不在 GPU 上同步。
- **仍存在的同步点（候选，未验证是否常被触发）**：`geometry_sca.py:285` 的现场路径——验证/测试 loader 不走预取包装时，仍会执行 `.max().item()`。
- **DataLoader 异步 H2D**：`CudaPrefetchLoader`（`data.py:33-124` 附近）用独立 CUDA stream、`pin_memory`、`non_blocking=True` 搬运，配合 `cuda.Event` 让主线程只在真正取数据时等待一次，而不是在 `.cuda()` 调用处隐式同步。

### Cosmos3

- **让 CPU 侧元数据留在 CPU**：`batch_transfer.py` 包装 `misc.to(...)`，只有同时满足"HIP 构建、显式传 `device`、没有传 `dtype`/`memory_format`/额外 kwargs、`data` 是 Mapping 且含 `image_size`"时，才把 `image_size` 单独 `original(value, device="cpu")`，其余字段正常搬到目标 device；不满足任一条件就透传给原函数（`batch_transfer.py:28-55`，已读完整函数体确认条件判断和 fallback 路径）。
- **自建 `cu_seqlens` 绕开 `.max().item()`**：`mot_attention.py` 里的 `_sdpa`（约 330-354 行）对单条打包序列（batch=1）直接构造 `cu_seqlens_q = torch.cat([zero, torch.tensor([n_q], ...)])`，而不经过上游 `generate_varlen_parameters`。源码注释原话："The frontend derives cumulative offsets from seqlens via `generate_varlen_parameters`, which calls `.max().item()` (a device-host sync) and is explicitly disallowed inside a torch.compile region."，并用 `torch._check(n_q > 0)` 显式声明前提。这一处同时服务于"避免图中断"（torch.compile 区域内不允许 `.item()`）和"避免同步"两个目的。
- **按 `grid_thw` 缓存辅助量，避免每步重新读取**：`qwen.py`（约 204-227 行）的 VisionModel 用 `(grid_thw.device, tuple(grid_thw.flatten().tolist()))` 做键的 FIFO 缓存，复用 `rotary_pos_emb`/`cu_seqlens`/`chunk_lengths`。这里仍有一次 `.tolist()` 用于算缓存 key，但避免了缓存未命中时重复执行的 `repeat_interleave + cumsum + pad + tolist` 链条。
- **多模态 RoPE 索引整段放在 CPU**：`qwen.py` 的 `get_rope_index`（约 286-389 行）先把 `input_ids`/`grid_thw`/`attention_mask` 全部 `.cpu()`，所有数据相关的索引逻辑在 CPU 完成，最后一次性 `.to(target_device)`。前提是这类调用通常只发生在 prefill 阶段，频率低，小张量的 H2D 开销可以接受。
- **timestep 用 pinned 非阻塞拷贝喂给 CPU 侧打包逻辑**：`omni_model.py`（约 256-260 行起）`timesteps_vision.to("cpu", non_blocking=True)`（仅当 `is_cuda`），结果传给 `_pack_input_sequence`，在 CPU 上继续构造。已确认消费方（`_pack_input_sequence` 及后续 `sample_ts[i:i+1].expand(...)` 的构造）都在 CPU 上进行，没有在拷贝完成前读取的风险；但该函数本身是否在所有调用路径上都确认了 pinned 来源，未逐一追溯到 `timesteps_vision` 的分配处，标注为部分验证。

### OpenWAM

- **`host_async` 组（默认关，`settings.py:63`）**：模块 docstring 原话："The default loop reads loss/grad-norm with .item() every step, DeepSpeed's throughput timer calls cuda.synchronize() inside engine.step, and a few small CPU tensors are uploaded with blocking copies."（`host_async.py:5-13`）。
  - `configure_deepspeed` 关闭 DeepSpeed 吞吐计时器的同步（`timers["throughput"] = {"enabled": False, "synchronized": False}`，`host_async.py:24-31`）。
  - `LaggedMetrics.push`（`host_async.py:49-87`）把 loss 三项和 grad_norm 堆成一个零维张量数组，拷到 `pin_memory=True` 的 host buffer，用 `cuda.Event` 记录；`flush`（`:113-130`）在下一次 `push` 时才 `event.synchronize()` 并一次性 `host.tolist()`，即延迟一步读取（P5 模式）。
  - 分布式场景下有 `_gather_aside`（`:89-111`），在独立 `cuda.Stream` 上做 `all_gather_into_tensor` 并等待，避免训练主 stream 被这个小集合通信卡住。
  - `upload`（`:34-39`）对 CPU 源张量先 `pin_memory()` 再 `non_blocking=True` 搬运，非 CPU 源则走普通 `.to()`。
  - `rope_freqs`（`:133-143`）按 `(grid, device)` 缓存设备端 RoPE 表，避免每次重建。
- **文本长度在 CPU 上算，避免 `.tolist()` 读 GPU 张量**：`text.py` 的 `_build_padding_plan`（约 94-105 行）和 `t5_prefix_plan` 相关逻辑（约 120-137 行）都在 CPU 上计算长度；`ports/encode.py:50-67` 用 `host_lengths = mask.gt(0).sum(dim=1).long() if cpu_plan else None`，消费时 `host_lengths.tolist()` 只在已经是 CPU 张量时调用。`text_trim` 组默认开（`settings.py:47`），在 `text.py:140-163` 里把 mask 先 `keep_cpu = mask.detach().to(device="cpu")` 再在 CPU 上找裁剪边界。
- **未找到**：`ports/vae.py` 和 `pointwise.py` 里没有命中 `.item()`/`.cpu()`/`.tolist()`/`non_blocking`/`synchronize`/`nonzero`/`masked_select`/`timestep`/`sigma`/`scheduler` 等关键词（已用 grep 核实），说明 OpenWAM 的同步消除集中在训练循环和文本/MoT 侧，VAE 与采样路径这次没有发现对应改动。

### OpenVLA

- **跳过 FA2 unpad 路径避免 varlen 的 `.max().item()`**：`skip_fa2_unpad.py` 包装 `LlamaModel._update_causal_mask`；当 `flash_attention_2` + 有显式 mask + `past_seen_tokens==0` + 右 padding（`attention_mask[:, 0].all().item()`）时直接返回 `None`，让 FA2 不再进入 varlen/unpad 路径。源码注释原话（`skip_fa2_unpad.py:14-27`）："The `.item()` is a scalar D2H sync, far cheaper than the one this Group removes: the varlen path needs a data-dependent `torch.nonzero` plus `seqlens_in_batch.max().item()` on every step."——即改写后仍保留一次标量同步（判断是否右 padding），但去掉了 varlen 路径里"每步一次 `nonzero` 加 `.max().item()`"。左 padding 的批次会显式回退到原函数，不是静默改变行为。

### GR00T

- **Qwen3-VL 视觉位置预处理整段放在 CPU**：`qwen3_backbone.py` 的 `_apply_cpu_position_preprocessing`（约 404-441 行）把 `visual.fast_pos_embed_interpolate`、`visual.rot_pos_emb`、`rope_model.get_rope_index` 替换成 CPU 版本（`_cpu_fast_pos_embed_interpolate:142`、`_cpu_rot_pos_emb:211`、`_cpu_get_rope_index:248`），并保留 `_gr00t_original_get_rope_index` 以便文本单模态路径走回原实现。相邻注释原话（约 395 行）提到要避免"thousands of synchronous scalar DtoH copies per forward"。这是本次读到的、措辞最直接的同步消除案例，但本次没有重新测量，也没有对照验证"数千次"这个说法。
- 该文件里还有 `_apply_vision_patch_embed_channels_last`（约 443-477 行），那是内存布局优化，不是同步消除，不要混为一谈。

## 候选，未验证（仅静态扫描/阅读命中，未逐一确认运行时行为）

- Cosmos3 `losses.py` 约 115-120 行：HIP 路径用 `ordered_all_reduce` 对齐 RCCL 顺序（通信顺序问题，不是同步消除），但紧接着的 rank0 日志仍会对 `loss_avg` 调用 `.item()`，这一处本身未被消除。
- Cosmos3 `checkpoint_dcp.py` 约 149、208 行：出现 `self.staging_stream.synchronize()` 和 `dist.barrier()`，这些是 checkpoint 保存流程里显式需要的同步点，未被作为消除对象处理，也不一定应该被消除。
- BEVFusion `transfusion.py` 约 337-385 行：训练目标生成（Gaussian heatmap）把 GT 搬到 CPU 循环绘制再搬回 GPU，减少了逐元素的标量同步，但整体仍有一次 D2H（`gt_bboxes_cpu.cpu()`）和一次 H2D（`heatmap_cpu.to(device, non_blocking=True)`）；是否命中取决于 GT 数量和调用频率，未重新测量。
- BEVFormer 非预取路径（`geometry_sca.py:285`）：如上所述，验证/测试 loader 大概率仍会触发 `.item()`。
- MapTRv2、BEVFusion 里是否还有自定义 CUDA/HIP 扩展内部含 `cudaStreamSynchronize`/`hipStreamSynchronize`：本次阅读停在 Python 绑定层，没有读扩展的 C++/HIP 源码，未验证。

## 邻近但不是同步消除的话题（仅指向）

- Cosmos3 `sequence_packing.py` 的 FlashAttention `max_seqlen` 分桶（约 15-37 行）：目的是减少 `torch.compile` 因 Python int guard 变化而重编译，属于编译稳定性，不是同步消除；但它和 P4 模式用的是同一类"自己构造 varlen 参数"的手法，值得对照着看。
- OpenWAM 的 `deferred_gather.py`、`zero.py` 里的双缓冲 `cuda.Event`，以及 `common/pytorch` 下通用的 FSDP ordered all-gather/reduce-scatter：解决的是分布式集合通信的顺序与重叠问题（RCCL/NCCL 要求同 communicator 全局顺序一致），不是 host-device 同步；不要把"有序调度"和"消除同步"混为一谈。
- channels-last、AOT 编译相关的改动（见 `optimize-channels-last` skill）：即使文档里提到"避免了某次拷贝"，那通常是设备内的内存布局转换，不是 D2H/H2D 同步。

## 与现有 compile 相关 skill 的对照

`torch-compile-integration`（见本仓库同名 skill）主要覆盖"哪里加 `torch.compile`、图中断诊断、重编译、嵌套编译"，只在一处提醒"`.item()` 等可能触发图中断或同步，需要实测"，没有把同步消除本身当作系统主题，也没有收录上述任何具体案例（如 `host_async.py`、`batch_transfer.py`、`skip_fa2_unpad.py` 等）。本 skill 和它的边界是：凡是"在 `torch.compile` 捕获区域内因为 `.item()`/动态分支导致图中断或报错"，优先交给 compile 相关 skill 的图中断诊断；凡是"在 eager 路径或编译区域之外发生的阻塞 D2H/H2D、或库内部默认同步"，属于本 skill。两者在 Cosmos3 的 `mot_attention.py` 和 `sequence_packing.py` 上有交集，上面已分别说明。
