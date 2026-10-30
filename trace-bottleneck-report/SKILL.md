---
name: trace-bottleneck-report
description: 从已采集的 PyTorch Profiler trace(多 rank)及可选旁证数据(运行记录、编译日志、key_averages、HTA 输出)统计 GPU 耗时构成、空闲归因、同步、数据搬运、通信等数据表,并给出"可回收上限"排行,快速指出新模型最值得优化的瓶颈方向。只做统计,不实施优化、不启动训练或采集。用于拿到 trace 后先看瓶颈在哪;没有 trace 时只给采集清单。
---

# Trace 瓶颈速览

把已有的 Profiler trace 变成一份可复算的统计报告:每类开销占多少、空闲从哪来、同步值不值得改、拷贝和通信暴露了多少,最后给出前 5 个可回收上限的瓶颈。**只统计数据、指出方向类别,不改项目源码、不启动训练、不替用户做采集。**

## 输入

| 数据 | 必需 | 说明 |
| --- | --- | --- |
| Profiler trace | 是 | `trace_rank_<rank>_*.json`(可 `.gz`),所有 rank,同一次运行,含设备 kernel 事件和 `ProfilerStep` |
| 运行记录 | 否 | 通用格式见 [collection.md](references/collection.md) §2(`command.json`、`step_metrics/rank-*.json`,由用户的训练脚本自行写出);给出 batch/卡数/环境开关和不受 profiler 影响的真实 step 时间 |
| 编译日志 | 否 | `TORCH_LOGS=graph_breaks,recompiles` 的输出,统计图断/重编译次数及原因 |
| key_averages 文本 | 否 | rank 0 的算子表(常在 `train.log`),只作 top 算子交叉校验 |
| HTA 输出 | 否 | 已有的 `temporal/idle/overlap` CSV,只读不运行 |

缺 trace 时不要编造数字:先读 [references/collection.md](references/collection.md),把采集清单交给用户,等用户采集完再统计。

## 流程

1. **核对数据**:确认 trace 来自同一次运行、rank 齐全、含 `ProfilerStep` 与设备事件;记录是否带调用栈(`with_stack`)。
2. **运行脚本**(只依赖 Python 标准库,不读写项目代码):

   ```bash
   python3 scripts/trace_report.py --trace-dir <trace 目录> -o <新输出目录> \
       [--run-dir <运行目录>] [--compile-log <日志>] [--key-averages <文本>] [--hta-dir <目录>] \
       [--project-hint mypkg/] [--env-prefix MYPROJ_] [--jobs 4]
   ```

   输出目录必须不存在,脚本拒绝覆盖。`--trace-dir` 的上级目录若含 `command.json`,自动当作运行记录。
3. **先看第 0 节排行**,再按"证据"列的编号读对应表(§1 总览、§2 GPU 耗时、§3 空闲归因、§4 同步、§5 搬运、§6 通信、§7 旁证)。
4. **向用户交付**:排行 + 关键证据表 + 限制;方向类别用 [references/handoff.md](references/handoff.md) 指引去向,不写实施方案。

## 常用选项

| 选项 | 默认 | 作用 |
| --- | --- | --- |
| `--step N` | 每个 rank 的中间 step | 指定 `ProfilerStep` 编号 |
| `--project-hint a/,b/` | 空 | 用路径片段识别项目帧,栈里的 `file:line` 才落在项目代码;不给则按库前缀过滤 |
| `--rules rules.json` | 内置 | 补充 kernel 分类(排在内置规则之前);未分类占比 >5% 时报告会提示 |
| `--small-gap-us` / `--big-copy-mib` / `--big-gap-ms` | 50 / 1 / 5 | launch 小间隙、大拷贝、大间隙阈值 |
| `--starve-pct` / `--mask-pct` | 50 / 20 | 同步判定"GPU 饥饿"/"被掩盖"的空闲/CPU 比例 |
| `--min-impact-pct` | 0.5 | 调用点 GPU 空闲占 step 低于此值判"影响小",不报饥饿、不进排行 |
| `--slow-rank-ms` | 10 | 最慢 rank 空闲比 rank 0 多于此值时补一份该 rank 的空闲归因 |
| `--peak-h2d-gbs` / `--peak-comm-gbs` | 无 | 提供硬件峰值才显示"带宽占峰值%",trace 里没有峰值,不猜 |
| `--env-prefix P1_,P2_` | 空 | 运行记录里额外保留的环境变量前缀(内置只保留 HIP/NCCL/RCCL/OMP/TORCH/PYTORCH/CUDA/GPU) |
| `--warmup-steps N` | 10 | 计算真实 step 中位数时跳过的前 N 步 |
| `--jobs N` | 1 | 并行解析 rank(单个 rank 约 1.3 GB 内存) |

## 输出

- `report.md`:第 0 节排行 + §1~§7 各表;每个百分比列表头写明分母。
- `summary.json`:全部数字、阈值、口径(`schema_version=1`),便于脚本比较或以后加对比功能。

## 解读边界(必须遵守)

- **单次采集、只取每个 rank 的一个 step**,各 rank 不独立:不报显著性,不把上限当收益。
- **上限 ≠ 收益**:排行的 ms 是占用上限;各候选可能覆盖同一段空闲,不能相加。
- **trace 的 step 时间含 profiler 开销**,带 `with_stack` 的更明显;测速用运行记录里的真实 step 时间,优化前后对比另做无 profiler 的测速。
- **区分标签**:`[实测]` 来自 trace 事件,`[推断]` 来自结构规则(如通信等待、搬运提示),`[上限]` 为上限估计。
- **口径**:占累计、占计算、占 step、占忙时的分母不同,不可跨列相加;HTA idle 与本报告空闲窗口和规则不同,不能互换。
- **同步不看耗时看代价**:`.item()` CPU 耗时大不等于有代价,以"调用期间 GPU 空闲"判定;设备同步期间 GPU 忙 ≥90% 说明在等真实工作,不属于同步消除。
- 栈信息缺失时调用点只到外层算子链,明确写"未定位到代码行",不猜代码行。

## 自测

```bash
python3 scripts/selftest.py   # 合成 trace,断言各表数字、分类总和与百分比定义
```

## 参考

- [references/collection.md](references/collection.md):手动采集清单(只写清单,不执行)
- [references/metrics.md](references/metrics.md):各表母体、分母、单位与归因规则
- [references/handoff.md](references/handoff.md):排行方向类别与其他优化 Skill 的衔接
- [references/sample-report.md](references/sample-report.md):用合成数据生成的示例报告
