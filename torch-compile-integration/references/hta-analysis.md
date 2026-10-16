# HTA 简明操作参考

HTA是分析已有PyTorch Profiler/Kineto Trace的Python库。profile-model组织诊断，HTA提供适用汇总，torch-compile-integration结合源码提出边界建议。本文只记录工具关联、使用条件及可靠性规则；任务步骤和报告格式由用户提示词决定。

## 选择已有环境与确认输入

- 指定了HTA Python时优先检查该解释器；不可用时说明原因，不默默改用其他环境。未指定时先检查当前Python，再检查已提供或环境管理器列出的相关环境，不递归扫描整台设备/自动连接新服务器。
- 用已有环境检查sys.executable、Python与HolisticTraceAnalysis发行版版本，以及能否导入 `from hta.trace_analysis import TraceAnalysis`。包名/版本查询和导入是两项检查；不自动pip安装、升级或创建环境。
- 官方当前说明列出Linux/macOS及Python>=3.10。目标设备、已装版本及采集后端仍须核对；不把Windows主机或某个设备事件格式假定为已获支持。离线读取已有Trace不要求GPU，也不启动项目训练/推理。
- 明确Trace实际文件和rank映射，核对同一次运行、配置/源码对应、采集时间、step数、预热/首次编译/栈采集影响。元数据不足就标注无法确认，不按文件名强行认定同次运行；不覆盖原始数据。
- 复用仍有效的解析结果/报告时，先核对来源、分析版本、ranks与窗口。优先内存复用；未经用户允许不新增Parquet等持久缓存，也不把另一进程的报告当作已有解析对象。

## 调用与返回形式

下面是本次核对版本的接口示例，不是可直接执行的独立脚本；`trace_files`、`trace_dir`、`selected_ranks` 必须来自当前输入。其他版本以实际签名和返回对象为准。

```python
from hta.trace_analysis import TraceAnalysis

analysis = TraceAnalysis(trace_files=trace_files, trace_dir=trace_dir)
# 以下调用应分别执行、分别捕获错误和警告；一个失败不跳过其他独立分析。
temporal = analysis.get_temporal_breakdown(visualize=False)
kernel_types, kernels = analysis.get_gpu_kernel_breakdown(visualize=False)
idle, idle_stats = analysis.get_idle_time_breakdown(
    ranks=selected_ranks, visualize=False, visualize_pctg=False
)
# 多rank且相关通信事件可识别时使用；无通信或输入不足时标记不适用。
overlap = analysis.get_comm_comp_overlap(visualize=False)
```

| API | 返回形式与用途 | 覆盖检查 |
| --- | --- | --- |
| get_temporal_breakdown | DataFrame；各rank计算/非计算/空闲的汇总 | 根据实际设备活动窗口核对分母；不直接等同CPU step时长 |
| get_gpu_kernel_breakdown | 两个DataFrame；类型占比与按kernel/rank统计 | 记录Top-K/筛选参数；截断后的列表不是全部kernel；累计时长不是并集 |
| get_idle_time_breakdown | 元组；首项分解表，第二项可为None | 显式传selected_ranks；默认可能只选rank0；记录stream和启发式分类条件 |
| get_comm_comp_overlap | DataFrame；已加载rank的通信/计算重叠 | 记录实际输出ranks、通信识别条件和比例分母；无通信时不能把NaN当0% |

`trace_files`可显式指定本次选取的rank→文件；若使用目录自动发现，核对实际解析/保留的文件、ranks和steps。构造参数可能默认排除最后ProfilerStep，需记录实际采样范围，不能只看输入文件总数。以上无ranks参数的API作用于该分析对象实际加载的数据。

## 每项单独记录结果

对构造/解析及每项API分别说明成功、部分成功、失败或跳过，附错误/警告、输入和实际覆盖。空表、None、NaN、丢失rank或缺列需要解释；仅导入成功/解析成功不证明所有指标可靠，不用虚构零值填平缺口。

检查返回类型后再解包或处理。保留有意义的成功结果；某项失败时继续其他独立项，最终明确哪些没有完成。工具警告即使没有异常也须核对，不能只据退出码认定可靠。

## 单位、窗口与指标解释

- 每个关键数字标明窗口、分母、单位、rank/stream和样本数；先检查原始Trace单位元信息、实际列名和安装版本实现，再换算。不同口径并列说明，不强行统一。
- 本次核对的HTA源码中temporal返回列包含`(us)`，而接口文档存在ns描述；这说明不能仅据文档一句话硬编码单位。本观察不保证其他版本沿用相同字段或单位。
- 区分CPU墙钟窗口、GPU annotation范围、kernel累计时长、设备活动区间并集。重叠执行会使累计时间大于窗口；CPU墙钟减累计kernel时间不能称为GPU idle。
- temporal空闲与每stream的idle分类不是同一对象；单stream空闲不证明整个GPU空闲。Host/Kernel/Other等待是工具按事件与规则给出的分类，不直接证明CPU代码、数据加载或通信的唯一根因。
- 通信计算重叠率需要核实分母；本次核对实现以通信区间并集时长为分母，而非整个训练step。不能只引用百分比就比较不同窗口/rank。
- 事件时间占比不是SM利用率、带宽饱和度或算力效率；这些需要额外硬件计数器/证据。累计时长相加不等于关键路径。
- 关键结论抽查原始时间线、事件关联和源码。区分观察、支持的归因和待验证假设；窗口差值和前后阶段差异不能直接写成compile收益。

## 输出与降级

遵守用户的报告格式和输出目录，默认关闭API的自动绘图，不额外导出CSV/图表；必要的临时分析命令/数据仅落在授权位置，不覆盖已有报告或原始数据。本文不规定报告章节、固定文件名或六步工作流。

没有HTA或Trace不兼容时，继续profile-model可做的原始Trace分析及源码判断，并明确HTA未运行/部分完成。只有recompile日志时分析已观察的guards/重复编译，不声称已定位graph break；没有显存快照时不声称完成分配器诊断。没有输入时列缺口，不为取证启动模型。

无GPU但有兼容Trace和HTA时可以离线分析；缺少库、解析失败或单位不明则相应降级。结果可指导后续实验，不能替代在一致条件下、关闭Profiler和诊断开销的实际性能/正确性验证。

## 官方参考（可选核对）

- [HTA仓库与安装说明](https://github.com/facebookresearch/HolisticTraceAnalysis)
- [TraceAnalysis API](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html)
- 本次接口与指标实现核对使用HTA提交 `c023a08a24ed44b4dd9f691d0e3d04464f1e76e9`：[接口](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/c023a08a24ed44b4dd9f691d0e3d04464f1e76e9/hta/trace_analysis.py)、[时间/空闲分解](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/c023a08a24ed44b4dd9f691d0e3d04464f1e76e9/hta/analyzers/breakdown_analysis.py)、[通信重叠](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/c023a08a24ed44b4dd9f691d0e3d04464f1e76e9/hta/analyzers/communication_analysis.py)。不将该提交作为目标环境必须安装的版本。
