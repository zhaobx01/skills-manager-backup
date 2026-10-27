# 布局与后端

4D NCHW 的典型 channels-last stride 为 `(H*W*C, 1, W*C, C)`；5D NCDHW 的典型 channels-last-3d stride 为 `(D*H*W*C, 1, H*W*C, W*C, C)`。尺寸为 1、空张量、视图或非稠密张量可能产生不同或有歧义的 stride，不能仅用固定公式判断有效性。显式转换可能分配并复制，保留数值但可能改变 alias；检查原地修改和返回对象身份。

`module.to(memory_format=...)` 原地转换模块的适用参数和 buffer，可能影响其他模块；对混合 Conv2d/Conv3d 子图分别选择。参阅安装版本的 `torch.nn.utils.convert_conv2d_weight_memory_format` 和 `convert_conv3d_weight_memory_format`，若可用则评估它们限定权重转换的适用性。初始化、加载、设备迁移、优化器、分片和 compile 的顺序须在目标框架验证，尤其要检查优化器参数引用及分布式一致性。

在 reshape 合并相机或视频维度之前先确认语义及 stride；在实际卷积输入转换。输出送给依赖连续 NCHW/NCDHW 的扩展或外部接口时，明确其 stride 契约，必要时只在该边界 `.contiguous()` 并计入开销。不要假定第三方扩展理解任意 stride。

ROCm 的建议变量可能因 PyTorch 构建、版本和使用的卷积路径不同而不可用或无效果。检查目标版本源码/文档及运行日志，以独立进程比较 unset 和指定值，记录进程环境；不要把变量名称作为布局生效证明。`MIOPEN_FIND_MODE=1` 在 MIOpen 文档中对应 normal/full find，可能增加搜索与缓存成本，默认值及行为以安装版本为准。控制持久 FindDb/PerfDb 与预热状态并记录；不要删除用户缓存作为默认步骤。精度相关变量属于独立实验。

来源：[PyTorch memory format 教程](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)、[PyTorch 卷积权重转换实现](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/memory_format.py)、[MIOpen 环境变量](https://rocm.docs.amd.com/projects/MIOpen/en/latest/reference/env_variables.html)。
