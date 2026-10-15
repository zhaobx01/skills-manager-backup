# MapTRv2 实践案例：阶段编译与 eager 岛

案例采集日期：2026-10-14；可迁移整理日期：2026-10-15；对应 Skill v1.2.0。本文件是完整随包提供的静态历史案例，记录具体做法及其风险，不要求其他项目照搬。

**独立阅读约定**：不需要安装 MapTRv2、打开原始源码、访问原电脑、获取基础版本或重新核对历史哈希。下列路径、符号、行号和摘要只是采集时的定位标签，不是本 Skill 要读取的外部文件。日常使用只分析用户当前提供的目标项目。

## 证据边界

- **源码已采用**：本次静态 AST 检查找到 19 个 torch.compile 装饰入口与 2 个 torch._dynamo.disable 边界，均逐项列在下表。装饰器统计不等于实际生成图数量。
- **用户报告有效**：用户说明该项目进行了很多有效 compile 优化，这是项目整体层面的反馈，没有给每个入口单独归因。
- **日志支持**：在提供目录内未找到可关联的训练、trace 或 A/B 性能日志。源码有 benchmark 工具不代表已经运行或有结果。
- **本次运行复测**：未执行模型、训练、容器或 GPU 编译；没有可公布的加速百分比或精度验证结果。

逻辑来源为“MapTRv2 优化快照”和“MapTRv2 基础快照”。表中源码路径相对优化快照的项目根；摘要表中的 `MapTRv2/` 仅表示历史快照中的项目子目录。基础快照历史 HEAD 为 `e03f097abef19e1ba3fed5f471a8d80fbfa0a064`；优化快照未发现 Git 元信息。两份快照的内容差异不等于完整或唯一的优化历史，也不构成安装或运行要求。

## 配置与调用范围

本次沿启动脚本的默认配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py` 追踪：MapTRv2 → MapTRv2Head → MapTRPerceptionTransformer → LSSTransform 与 MapTRDecoder，训练配置使用 MapTRAssigner。LSSTransform 继承 BaseTransform，forward 中存在向子类 get_cam_feats/get_mlp_input 的动态派发。

“可达”仅指配置与静态调用链支持，不代表所有输入和分支在运行中均命中。encoder 中 BaseTransformV2、LSSTransformV2 及其他实现不能仅因存在于同一文件就计入默认配置。3D 点归一化是 gt_z_flag 条件路径；默认配置未启用该覆盖，不能把其收益计入默认 2D 情况。训练 assigner 与随机增强也不能用推理 benchmark 证明。

下表位置是采集时的装饰器起始行；未导入项目依赖执行。路径相对历史项目根。每一项均只有“源码采用”的独立证据；性能和精度尚未逐项复测。迁移到其他电脑时直接阅读下表，无需这些文件存在。

## 19 个编译入口与 2 个 eager 边界

| ID | 源码位置与限定符号 | 调用关系/配置路径 | 做法 | 迁移条件与验证风险 |
| --- | --- | --- | --- | --- |
| M01 | `projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py:117` — `MapTRAssigner.hungarian_match` | assign → hungarian_match；训练匹配分支 | eager CPU 岛：detach/cpu → SciPy → 设备索引 → AssignResult | cost 列过滤、GT 标签索引、零 GT、ties、D2H/H2D 与原地标签赋值 |
| M02 | `projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py:136` — `MapTRAssigner.assign` | Head loss/target → assign；训练分支 | 编译匹配代价；显式 options 开 CUDA Graph、关 trees；调用 eager 岛 | 固定 GT 打包契约、mask 过滤、order_index 对应；原空输入提前返回被注释，需回归验证 |
| M03 | `projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py:25` — `normalize_3d_pts` | Head.loss_single 等条件路径；gt_z_flag 启用时 | default；clone/切片/缩放 | 非默认 2D 路径；z 范围、梯度、clone/alias；不能计入默认配置收益 |
| M04 | `projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py:38` — `normalize_2d_bbox` | Head.loss_single → 归一化 | default；坐标格式转换与缩放 | 范围常量的 guards、切片原地语义与调用频率 |
| M05 | `projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py:50` — `normalize_2d_pts` | Head.loss_single → 归一化；2D 条件分支 | default；clone/缩放 | 点数、范围、dtype、梯度及调用频率 |
| M06 | `projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py:61` — `denormalize_2d_bbox` | Head.loss_single → 反归一化 | default；框坐标反变换 | 容差、原地切片与底层 view；与 assigner 中同名函数区分 |
| M07 | `projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py:71` — `denormalize_2d_pts` | Head.loss_single → 反归一化 | default；clone/坐标反变换 | 输出/梯度/存储行为；不能仅因函数短就推广编译 |
| M08 | `projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py:281` — `MapTRv2Head.compute_decoder_predictions` | Head.forward → compute_decoder_predictions | default；静态层循环、list 追加、分类回归与 one2one/one2many 拆分 | 循环次数/配置 guards、输出顺序、回归梯度、辅助分割分支 |
| M09 | `projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py:339` — `MapTRv2Head.prepare_transformer_inputs` | Head.forward → prepare_transformer_inputs | no-cudagraphs；embedding/位置编码/mask 准备 | train/eval 的 num_vec、配置分支、CPU mask 创建后 .to(device)、mask 切片修改 |
| M10 | `projects/mmdet3d_plugin/maptr/detectors/maptrv2.py:74` — `MapTRv2.extract_img_feat` | detector.forward 路径 → extract_img_feat | default；backbone/neck 周围的图像特征阶段；内部 GridMask eager | img.squeeze_ 输入修改、字典转 list、batch/queue guards、eager 增强成本 |
| M11 | `projects/mmdet3d_plugin/maptr/modules/decoder.py:23` — `MapTRDecoder.forward` | transformer.forward → self.decoder；配置 MapTRDecoder | no-cudagraphs；整段 decoder forward，固定层循环及回归分支 | 层数/分支 guards、detach/梯度、reference 更新；不是所有 forward 都必须保持 eager |
| M12 | `projects/mmdet3d_plugin/maptr/modules/encoder.py:93` — `BaseTransform.matmul_1` | LSSTransform → BaseTransform.forward → get_geometry_v1 → matmul_1 | no-cudagraphs；矩阵变换及相邻 cat/逐元素计算 | shape/view、dtype、数值误差、拆分后的融合和启动开销 |
| M13 | `projects/mmdet3d_plugin/maptr/modules/encoder.py:106` — `BaseTransform.matmul_2` | get_geometry_v1 → matmul_2 | no-cudagraphs；组合矩阵乘与平移 | points 原地加减及 alias、梯度、连续性；前后的 inverse 仍在外层 |
| M14 | `projects/mmdet3d_plugin/maptr/modules/encoder.py:115` — `BaseTransform.matmul_3` | get_geometry_v1 → matmul_3 | no-cudagraphs；独立几何矩阵计算 | 微小矩阵批处理收益待测，不因含 matmul 就判为主要热点 |
| M15 | `projects/mmdet3d_plugin/maptr/modules/encoder.py:272` — `BaseTransform.extract_metas` | BaseTransform.forward → extract_metas → stack_metas | no-cudagraphs；元数据堆叠/转换和矩阵切片 | stack_metas 内含 NumPy/list 分支及 H2D；不能描述为已消除 CPU 工作的纯 Tensor 核 |
| M16 | `projects/mmdet3d_plugin/maptr/modules/encoder.py:1112` — `LSSTransform.get_cam_feats` | BaseTransform.forward 动态派发至 LSSTransform.get_cam_feats | no-cudagraphs；外层 compile 包住 force_fp32 包装 | 精度装饰器顺序、DepthNet、softmax/乘法、view/permute、前后向数值 |
| M17 | `projects/mmdet3d_plugin/maptr/modules/encoder.py:1128` — `LSSTransform.down_sample` | LSSTransform.forward → super.forward → down_sample | no-cudagraphs；permute/contiguous/unbind/cat 和卷积下采样 | 视图与真实复制区别、通道映射、layout 与 compile 分开归因 |
| M18 | `projects/mmdet3d_plugin/maptr/modules/encoder.py:1203` — `LSSTransform.get_mlp_input` | BaseTransform.forward 动态派发至 LSSTransform.get_mlp_input | no-cudagraphs；相机参数切片/stack/cat | 元数据语义、shape、顺序和配置特化；不缓存不同样本的值 |
| M19 | `projects/mmdet3d_plugin/maptr/modules/transformer.py:288` — `MapTRPerceptionTransformer.format_feats` | transformer.forward → format_feats；返回传入 decoder | no-cudagraphs；多尺度固定循环、embedding 相加、flatten/cat | 层数/相机数、形状元数据与 device 分配、输出使用情况需按 decoder 跟踪 |
| M20 | `projects/mmdet3d_plugin/maptr/modules/transformer.py:317` — `MapTRPerceptionTransformer.initialize_queries_and_bev` | transformer.forward → initialize_queries_and_bev | no-cudagraphs；query expand、位置预测和 BEV 布局整理 | expand/view alias、batch/query/BEV 尺寸 guards、张量创建 |
| M21 | `projects/mmdet3d_plugin/models/utils/grid_mask.py:85` — `GridMask.forward` | extract_img_feat → self.grid_mask；use_grid_mask 且训练分支 | eager 随机岛；auto_fp16 外包 dynamo.disable | NumPy/PIL 随机性、self.l 修改、概率分支、.cuda() 设备与传输；不能删除增强来宣称加速 |

## 重点契约：固定容量 GT 并非只加装饰器

`maptrv2_head.py` 中 pad_to_static_list（定义行 801）使用 max_len=200，生成 `(out, mask, length)`；loss 中分别包装 GT bboxes、points、labels，交给 assigner。此处 200 是项目常量，不应变为通用默认值。

`MapTRAssigner.assign` 使用 gt_bboxes[0]/gt_pts[0]/gt_labels[0] 计算 padded cost，再用 `cost[:, gt_bboxes[1]]` 筛选有效列，传入 gt_bboxes[2] 作为真实 GT 数。bool mask 筛选后的动态尺寸是否造成捕获/重编译问题，需要日志证明，不能声称仅通过 padding 就彻底静态化。

与相邻基础源码相比，原空 GT/空预测提前返回分支在优化版本被注释。这是必须回归的契约风险，不是本次已证明的缺陷；必须覆盖空 GT、空 query、1/199/200/201 个 GT，以及有效 mask/length/标签和 order_index 对应。当前 out[:length]=t 没有显式定义超出 200 时的回退策略，不应把该做法无条件复制。

## 编译之外的共同变化

外层 dockerfile 采用含 Hygon/DTK 标识的镜像，并有 fastpt 与扩展构建步骤；启动脚本设置 MIOpen/ROCBLAS、通信重叠、NUMA，以及 TORCHINDUCTOR_LAYOUT_OPTIMIZATION 和 TORCHINDUCTOR_COORDINATE_DESCENT_TUNING。实际运行时 torch 精确版本、补丁和环境变量生效情况没有日志确认，不能只按 torch.cuda API 名称认定后端。

项目同时使用 default、max-autotune-no-cudagraphs，以及 assigner 的显式 triton.cudagraphs=True / triton.cudagraph_trees=False。这些是不同作用域的选择；支持与生效取决于构建版本，不能写成通用私有配置模板。布局、精度/TF32类策略、通信和 CPU 数据准备的收益必须与 compile 分开。

`tools/maptr/benchmark.py` 有同步墙钟计时、跳过 5 次预热的推理测试框架。固定预热次数是否足够、所有变化 shape 是否已经编译仍需检查；该工具不是训练迭代收益或任务精度的证明。

## 可迁移的检查顺序

先追踪实际配置与继承派发，再对计算阶段划分候选；逐项标记 CPU/随机 eager 岛和既有父子编译入口；验证包装顺序、输入修改和静态化契约；最后在冻结环境下以原实现 A、重构但关闭 compile 的 B、同源码开启 compile 的 C 进行对照。没有 B/C 对照时，不把跨版本总收益归因给 compile。

## 历史源文件摘要（不参与使用或安装校验）

SHA256 只标识案例采集时读取的内容，不代表运行或性能验证，也不是需要重新计算的验收项。保留这些记录只是为了历史追溯；加载、使用和迁移 Skill 均不需要原始文件。如果用户另行提供了新版本源码并要求重新研究，那是新的项目分析任务，应对其实际输入独立取证。

| 相对优化目录的文件 | SHA256 |
| --- | --- |
| `MapTRv2/projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py` | `d26b336e815fca293176e6f9aa700e60d8eb3741d87ab2f1c13a2d8235ddb6aa` |
| `MapTRv2/projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py` | `3498b230298b55a51aeac6cccfbaaa2dffe51861178bbcb5f9e99263f2a7a1ab` |
| `MapTRv2/projects/mmdet3d_plugin/maptr/detectors/maptrv2.py` | `5995d47d21364c61d1d688f4760f718d8e8c14f06289c8b1003f9600e356ad46` |
| `MapTRv2/projects/mmdet3d_plugin/maptr/modules/decoder.py` | `f5c60751aa1ceb80cedcb3ede586464f95539a9b9a96c6577df3710978353369` |
| `MapTRv2/projects/mmdet3d_plugin/maptr/modules/encoder.py` | `a0bb5d2acd9297fc6114b8558dd0d86953e43a17b98c56cd3ed1eb142a4ec3cd` |
| `MapTRv2/projects/mmdet3d_plugin/maptr/modules/transformer.py` | `12fa93e8c6adde38066b28a60e39091382a238ad2bb6205f6ee9be7f9030820d` |
| `MapTRv2/projects/mmdet3d_plugin/models/utils/grid_mask.py` | `e5d004f25d449bc803ad9a2cbfaf92c66cf43c7e206ee5f0860342c3e73cd20d` |
| `MapTRv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py` | `7aadef9e550d1268b97fdbd9795572e1f38707502857449e544edab34621f369` |
| `start_maptrv2.sh` | `43e2e8464aa289b6737812e39410f23028ae54d40efabb62e3f6ec4def53025c` |
| `start_mmdet3d.sh` | `95638a12407723d71913a37343afcedb7af94c0a6a81a37e6e94f322ec55f485` |
| `dockerfile` | `a20c966ea1ff4c18213486df0471216c15d31cf437d49168d6c9b66bda435498` |
| `MapTRv2/tools/maptr/benchmark.py` | `9f8b592533e87fa46a90c4237847ce24b73762f25e7a56b2a0ebf46465a9e352` |
