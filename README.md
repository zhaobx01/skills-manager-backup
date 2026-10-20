# Torch Compile Integration

本仓库保留 `torch-compile-integration` Skill，以及配套的训练瓶颈分析提示词。

## 内容

| 路径 | 用途 |
| --- | --- |
| [torch-compile-integration/SKILL.md](torch-compile-integration/SKILL.md) | 分析编译边界、接入方式与验证流程。 |
| [训练瓶颈分析提示词](prompts/training_trace_compile_analysis_prompt.md) | 结合 Profiler Trace、HTA 和源码生成瓶颈分析与编译建议报告。 |

## 使用

1. 在目标客户端安装或加载 `torch-compile-integration` Skill。
2. 打开提示词，填写项目路径、Trace、分析环境及优化目标等输入。
3. 将填写后的提示词用于目标项目。本轮仅进行离线分析和报告编写；训练、补采、编译实验及源码修改须另行确认。

提示词中涉及的访谈、Trace、显存和图断分析 Skill 不随本仓库提供；使用前按 Skill 中的协作说明核实可用性。缺少工具时应记录限制，不将未执行的分析报告为已完成。
