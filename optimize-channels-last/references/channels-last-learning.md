# 理解 Channels-last 与 MIOpen

这份说明用于理解布局与两个后端设置。示例是学习代码；实际项目先检查模型调用路径，再按 [验证方法](validation.md) 测试正确性和性能。

## 1. Channels-last 改了什么？

一张图有两个像素，每个像素有红、绿、蓝三个通道。设 `R1` 是第一个像素的红色值：

```text
像素 1 = (R1, G1, B1)      像素 2 = (R2, G2, B2)

默认连续布局（按通道集中摆放）：
内存 →  R1  R2  |  G1  G2  |  B1  B2

channels-last（同一像素的通道相邻）：
内存 →  R1  G1  B1  |  R2  G2  B2
```

这是简化示意：元素的值没有改变，改变的是元素在内存中的顺序，也就是 stride（跨维度移动一个位置需要跨过多少元素）。某些硬件和卷积实现适合通道相邻的访问方式，可能因此更快；实际收益取决于形状、数据类型、后端和转换开销。

**逻辑维度不变。** 4D 图像张量依然按 `[N, C, H, W]` 访问，5D 张量依然按 `[N, C, D, H, W]` 访问；D 可以表示深度或视频时间。不要为了设置 memory format 把模型接口改成 `[N, H, W, C]`。

```text
逻辑形状：描述每个维度代表什么     例：[N, C, H, W]
物理布局：描述元素在内存怎么排     例：channels-last
```

### 如何转换 2D？

下面使用独立的 Conv2d 演示，真实项目通常选择卷积子模块和进入它的输入：

```python
import torch
from torch import nn

conv2d = nn.Conv2d(3, 16, kernel_size=3, padding=1)
images = torch.randn(2, 3, 32, 32)  # N, C, H, W

conv2d = conv2d.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)

print(images.shape)   # 仍是 [2, 3, 32, 32]
print(images.stride())
print(images.is_contiguous(memory_format=torch.channels_last))
output = conv2d(images)
```

### 如何转换 3D？

```python
import torch
from torch import nn

conv3d = nn.Conv3d(3, 16, kernel_size=3, padding=1)
video = torch.randn(2, 3, 4, 32, 32)  # N, C, D, H, W

conv3d = conv3d.to(memory_format=torch.channels_last_3d)
video = video.to(memory_format=torch.channels_last_3d)

print(video.shape)   # 仍是 [2, 3, 4, 32, 32]
print(video.stride())
print(video.is_contiguous(memory_format=torch.channels_last_3d))
output = conv3d(video)
```

实际工程中不要按维数盲目转换所有张量，也不要在每一层反复转换。`to(memory_format=...)` 可能产生复制；其他算子或自定义扩展可能改变布局或要求不同 stride。整体转换模块也可能影响其他适用参数和 buffer。检查调用边界、优化器和分布式生命周期，并把转换开销算进性能比较。

尺寸为 1 的维度可能让默认连续格式和 channels-last 格式的判断同时为真；这时仅凭一个布尔值无法判断后端实际使用了哪种算法。PyTorch 的 4D 教程建议显式转换时使用 `to(memory_format=...)`，而不是依靠 `permute` 或退化形状下的 `contiguous` 来表达意图。

## 2. `PYTORCH_MIOPEN_SUGGEST_NHWC=1` 做什么？

它是 **PyTorch 的 MIOpen 卷积布局选择开关**。在支持该变量的构建中，设为 `1` 后，PyTorch 可以在满足其他条件时，为 MIOpen 卷积选择 channels-last 内存格式。

```text
代码转换输入或权重的内存格式
                ↓
PyTorch 判断设备、数据类型、输入及权重的格式
                ↓
开关允许且条件满足 → 建议 channels-last 卷积布局
                ↓
MIOpen 路径准备所需布局并执行卷积
                ↓
验证结果，并测量包含转换成本的实际性能
```

这个开关不会把整个模型自动转换，也不保证选择某个具体算法或获得加速。张量的 `is_contiguous(...)` 为真也不能单独证明实际使用了哪个 MIOpen 算法。其他后端、编译生成的 kernel 和厂商分支可能有不同路径，需核对目标版本并实测。

以 [上游 PyTorch 2.10.0 的实现](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/ConvUtils.h#L327-L360) 为例：这个 `NHWC` 开关同时参与 4D 和 5D 卷积布局选择；输入或权重至少一方须建议对应格式，且代码排除了 float64 等条件。该版本会根据 MIOpen 版本选择默认值，因此显式设为 `1` 可能与默认行为相同。它不读取单独的 `PYTORCH_MIOPEN_SUGGEST_NDHWC`；厂商版可能不同，要核对实际构建，不能从变量名称推断其已生效。

启动目标进程前设置，例如在运行训练的 Linux 容器中：

```bash
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
python train.py
```

在远程容器运行时，要在容器的启动环境设置，而不是只在本机终端设置。先核对已有环境变量与版本默认值，以独立进程比较 unset 与 `1`；不要认为 unset 必然等于关闭。

## 3. `MIOPEN_FIND_MODE=1` 做什么？

同一个卷积可能有多种可用实现。MIOpen 的 Find 过程会寻找适用方案并评估性能。`MIOPEN_FIND_MODE=1` 对应 **NORMAL / Full Find**，对候选求解器进行较完整的搜索和测量：

```text
给定输入、权重、卷积参数和布局
                ↓
MIOpen 查找 / 测量适用的候选实现
                ↓
得到候选结果供调用方使用
                ↓
后续运行可受进程内及数据库缓存状态影响
```

它调节的是**搜索策略**，不负责把张量转换成 channels-last。搜索需要时间和资源；是否有收益取决于调用路径、硬件、形状和缓存状态。对于长期训练、重复使用稳定形状的卷积，较高的首次成本可能被后续收益抵消；短任务或形状频繁变化时，搜索成本可能不划算。

例如，下面的数字仅为说明，**不是实测结果**：

```text
默认策略：首次准备 2 秒，后续每步 100 毫秒
Full Find：首次准备 20 秒，后续每步 95 毫秒
```

Full Find 多花 18 秒，每步节省 5 毫秒，在其他条件相同且忽略额外变化时约需 3600 步才能抵消；不足这些步数时总耗时可能更长。真实环境不保证 Full Find 一定更快。不同 MIOpen 版本的默认模式及行为可能不同，按安装版本文档确认。

在实际运行进程启动前设置：

```bash
export MIOPEN_FIND_MODE=1
python train.py
```

## 4. 如何一起使用和验收？

不要同时改变所有设置后把总收益归因于某一个变量。建议在相同模型、数据、精度、硬件和 compile 配置下，分别用独立进程比较：

```text
A 原始布局 + 原有环境                         → 建立基线
B channels-last 布局 + 原有环境              → 看布局变化
C channels-last 布局 + NHWC=1                → 看后端开关的增量
D C 的配置 + FIND_MODE=1                     → 看搜索策略的增量
```

记录原有变量值和版本默认值；若默认已启用 NHWC，B 与 C 不一定形成有效开关消融，应按目标版本设计对照。每组核对输出、训练梯度和任务指标；分别记录首次启动/搜索、预热后多次测量、端到端耗时及显存。固定并记录 FindDb/PerfDb、预热与缓存状态，避免将缓存差异误认作变量收益。出错或没有收益时恢复基线并复测。详细步骤见 [验证方法](validation.md)。

参考：[PyTorch channels-last 教程](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)、[PyTorch 2.10 卷积布局源码](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/ConvUtils.h#L327-L360)、[MIOpen Find 文档](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html)、[MIOpen 环境变量](https://rocm.docs.amd.com/projects/MIOpen/en/latest/reference/env_variables.html)。
