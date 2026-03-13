# Agent Decision Boundary Research

## 项目概述

训练 computer use agent 学习三元决策：Act / Ask / Refuse。当前阶段：AskBench 构建。

## GPU 服务器

- 连接方式：`ssh 5090`
- 硬件：RTX 5090
- 用途：所有实验在此机器上运行

## 实验工作流

1. **本地写代码** → commit & push
2. **SSH 到 5090** → git pull → 运行实验
3. **记录结果** → 实验报告写入 `experiments/` 文件夹

实验报告命名格式：`{实验名}_{YYYYMMDD_HHMM}.md`

### 实验报告必须包含的内容

1. **代码版本**：git commit hash
2. **运行命令**：完整的命令行
3. **环境信息**：Python 版本、关键依赖版本、CUDA 版本、GPU 型号
4. **实验参数**：模型名、数据集、超参数等
5. **运行时间**：开始时间、结束时间、总耗时
6. **结果数据**：定量结果（表格/数值）
7. **结论**：对结果的分析和下一步建议

## 项目结构

```
├── CLAUDE.md                              ← 本文件
├── README.md                              ← 项目总览
├── research/
│   ├── proposal-agent-decision-boundary.md ← 核心提案
│   ├── askbench-design.md                  ← AskBench 设计
│   ├── askbench-pilot/                     ← Pilot study
│   ├── literature-notes-core-papers.md     ← 论文笔记
│   ├── computer-use-agent-safety.md        ← 竞争格局
│   └── top-venue-trends-2025-2026.md       ← 顶会趋势
├── archive/video-accel/                    ← 旧项目存档
└── experiments/                            ← 实验报告
```
