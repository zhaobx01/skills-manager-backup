# TurboPhysAI 源码证据

以下为固定提交的静态源码观察，不代表该路径在任何运行中启用，也不代表正确性或性能已验证。各仓库/分支的文件不同，不能把某模型的结论套用到其他模型。逐模型的开关、边界、算子约定和未核实项见 [turbophysai-channels-last-evidence.md](turbophysai-channels-last-evidence.md)。

| 模型 | 转换在哪里发生 | 要点 |
| --- | --- | --- |
| MapTRv2（`bc8ced4`） | 整模型或仅 `img_backbone`（`TURBO_PHYSAI_CHANNELS_LAST_SCOPE`），加 backbone 入口 pre-hook | 开关由代码和 `runtime.yaml` 同时给出；DCN、GridSampler 要求输入输出格式一致才走 NHWC 路径；配置里同时有 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1` |
| BEVFormer（`bb302a6`） | backbone 入口无条件转换；另有整模型 `to(channels_last)` | 两处独立机制，无开关；自定义 hipDNN 可变形卷积输出是 NHWC buffer 的 permute 视图 |
| BEVFusion（`bb302a6`） | backbone 入口由环境开关决定；Conv2d 权重另行转换 | 先 `view` 合并 batch/camera，要关注输入 stride 契约；Conv3d 与 sparse conv 不转；只编译 camera backbone |
| Cosmos3（`bb302a6`） | HIP 构建下自动：Conv2d/Conv3d 权重、VAE 激活、缓存、Qwen patch-embed | 融合 NDHWC 算子要求规范 stride、bf16、通道白名单；不满足则回退 `F.conv3d` |
| OpenWAM（`e4528c3`） | VAE 权重、激活、缓存；默认开 | eager 用 `contiguous`，编译中用私有 `force_stride_order`；`as_strided` 统一单元维 stride 以复用图；私有接口须按版本验证 |
| OpenVLA（`e7aaad0`） | 所读 `runtime.yaml` 与 `vision_timm.py` 未显示转换 | 这不是全仓库或其他分支没有该优化的证明 |

**常见陷阱**（出自上述模型）：

- 名字相同、含义不同：MapTRv2 `bev_pool` 里的 `channels_last` 和 BEVFusion `depth.py` 的 `permute(0,2,3,1)` 是轴排列，不是 memory format。
- 整模型 `.to(memory_format)` 与只转卷积权重不等价，前者会触及其他 4D 参数。
- 自定义算子通常按输入格式选分支，且常要求所有相关 tensor 一致；格式不齐时静默回退，不报错。
- 缓存张量要与激活保持同一格式（Cosmos3、OpenWAM 都对缓存做了处理）。
- 配置里写了 ROCm 变量不等于路径生效；实际生效需用独立进程比较。
