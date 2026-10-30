# 排行方向类别与其他优化 Skill 的衔接

排行里的"方向类别"只指去向,**本 Skill 不实施**。下表列出每类对应的下一步和可用的 Skill/资料;用户确认要做时再切换到对应 Skill。

| 方向类别 | 先核对的证据 | 去向 |
| --- | --- | --- |
| 数据搬运 | §5 大拷贝的带宽、拷贝前 CPU `copy_`、拷贝前 GPU 空档;§3 头部/搬运等待 | `optimize-sync-elimination`(阻塞式 H2D、pinned 与 `non_blocking` 的安全条件);项目的 DataLoader / `_prepare_inputs` 路径 |
| 同步消除 | §4 判定为"GPU 饥饿"的调用点及代码行 | `optimize-sync-elimination`(`.item()`/`nonzero`/布尔掩码索引等的改写与验证) |
| 通信重叠 | §6 暴露占通信%、重叠率、起始极差;§1 暴露通信 | 分布式通信优化(梯度归约重叠、按 owner 归约、AllGather 流水等);本 Skill 不含这部分 |
| launch 开销/融合/编译 | §3 launch 小间隙占 step%;§7 图断/重编译计数;T1 中 elementwise 占比 | `torch-compile-integration`(编译边界)、图断诊断类 Skill |
| 慢 rank 排查 | §1 最慢 rank、§3 最慢 rank 的补充归因 | 对比该 rank 的数据加载、设备状态、通信起始时间;先确认不是外部负载 |
| 计算热点(不进排行) | §2 最大计算类别及 top kernel | `optimize-kernel-replacement`(融合/厂商 kernel 替换);conv 类可看 `optimize-channels-last` |

## 使用规则

- 先看排行第 1~2 项,不要同时推进所有方向。
- 上限不是收益。动手前用同口径(无 profiler、充分预热、重复采集)的测速确认,优化后再采一份 trace,用本 Skill 同样的参数再统计一遍并对照。
- 本 Skill 第一版不含 A/B 对比;`summary.json` 带 `schema_version` 与口径元信息,对比时先核对窗口、是否带栈、batch、阈值是否一致。
- 带栈 trace 只用于定位代码行,不用它的 step 时间比较性能。
