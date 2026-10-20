# Torch Compile Integration

本仓库保留 `torch-compile-integration` Skill，以及配套的训练瓶颈分析和编译实施提示词。

## 内容

| 路径 | 用途 |
| --- | --- |
| [torch-compile-integration/SKILL.md](torch-compile-integration/SKILL.md) | 分析编译边界、接入方式与验证流程。 |
| [训练瓶颈分析提示词](prompts/training_trace_compile_analysis_prompt.md) | 结合 Profiler Trace、HTA 和源码生成瓶颈分析与编译建议报告。 |
| [Compile 优化实施与验证提示词](prompts/training_compile_implementation_prompt.md) | 根据分析报告接入可回退编译，验证正确性、捕获情况与实际性能。 |

## 使用

1. 在目标客户端安装或加载 `torch-compile-integration` Skill。
2. 分析阶段使用训练瓶颈分析提示词，填写项目路径、Trace、分析环境及优化目标；该阶段仅离线分析和编写报告。
3. 实施阶段使用 Compile 优化实施与验证提示词，填写分析报告、运行环境、设备预算和验收要求，按确认的范围修改代码和测试。其默认模式为 `max-autotune-no-cudagraphs`，启动前须核实目标版本与后端支持。

提示词中涉及的访谈、Trace、显存和图断分析 Skill 不随本仓库提供；使用前按 Skill 中的协作说明核实可用性。缺少工具时应记录限制，不将未执行的分析报告为已完成。
