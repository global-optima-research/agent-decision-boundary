# Learning to Act, Ask, or Refuse — Calibrated Risk Decision Boundaries for Computer Use Agents

> 研究项目 | 目标：NeurIPS 2026 / ICLR 2027

## 一句话

现有 agent 安全方法都是二元决策（act/refuse），但现实操作大量处于灰色地带。我们提出三元决策框架 **Act/Ask/Refuse**，训练 computer use agent 学习**何时直接执行、何时请求人类确认、何时拒绝**。

```
[Act]                    [Ask]                      [Refuse]
打开文件浏览器          发送邮件给陌生人              rm -rf /
点击导航链接            修改系统设置                  转账到未知账户
滚动页面                删除一个用户没提到的文件        安装恶意软件
输入搜索关键词          购买商品                      关闭防火墙
                        以管理员权限运行程序
                        发布社交媒体帖子
```

## 核心贡献

1. **AskBench** — 首个评估 agent act/ask/refuse 三元决策能力的 benchmark（600 任务，8 领域）
2. **TriDecision** — 三元 DPO 训练方法（Risk-Level-Aware 权重 + 多轮风险累积感知）
3. **实证分析** — 在 AskBench + 现有 benchmark 上系统评估 frontier models 和训练后模型

## 项目结构

```
├── README.md                              ← 本文件
├── CLAUDE.md                              ← 开发指南
├── research/
│   ├── proposal-agent-decision-boundary.md ← 核心提案
│   ├── askbench-design.md                  ← AskBench 设计文档
│   ├── askbench-pilot/                     ← Pilot study (30 tasks)
│   │   ├── tasks.json                      ← 30 个 pilot 任务
│   │   ├── evaluate.py                     ← 评估脚本
│   │   ├── analyze.py                      ← 结果分析
│   │   ├── pilot-report.md                 ← Pilot 结果报告
│   │   └── results/                        ← 模型评估结果
│   ├── literature-notes-core-papers.md     ← 4 篇核心论文精读
│   ├── computer-use-agent-safety.md        ← Agent 安全竞争格局
│   └── top-venue-trends-2025-2026.md       ← 顶会趋势分析
└── archive/video-accel/                    ← 旧项目：视频推理加速调研
```

## 研究进度

| 阶段 | 状态 | 内容 |
|------|------|------|
| 文献精读 | ✅ 完成 | MOSAIC, Mind the GAP, Unsafer, Risk Knowledge |
| Benchmark 设计 | ✅ 完成 | AskBench 600 任务设计，8 领域，三元标注规范 |
| Pilot Study | ✅ 完成 | 30 任务 pilot，Claude Opus: 80% acc, 13.6% SVR |
| 全量任务构建 | 🔲 下一步 | 600 任务实例化 + 截图收集 |
| Baseline 评估 | 🔲 | GPT-4o, Claude, Qwen2.5-VL, MOSAIC |
| 偏好数据构造 | 🔲 | ~1800 偏好对 |
| TriDecision 训练 | 🔲 | DPO on Qwen2.5-VL-7B/14B |
| 评估 + 消融 | 🔲 | |
| 论文撰写 | 🔲 | |

## Pilot 结果（Claude Opus 4.6）

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 80.0% | 24/30 正确 |
| SVR | 13.6% | 该 ask/refuse 但直接 act 的比例 |
| ULR | 0.0% | 该 act 但 ask/refuse 的比例 |
| Ask-F1 | 0.812 | Ask 类判断能力 |
| Refuse Recall | 0.500 | 6 个该 refuse 的只识别了 3 个 |

错误模式：3 次 ask→act（用户指令明确时忽略风险），3 次 refuse→ask（为用户合理化意图）。

## 核心参考文献

- [MOSAIC](https://arxiv.org/abs/2603.03205) — Act/refuse 二元决策 (最直接竞品)
- [Mind the GAP](https://arxiv.org/abs/2602.16943) — Text safety ≠ tool-call safety
- [Unsafer in Many Turns](https://arxiv.org/abs/2602.13379) — Multi-turn compounding risk
- [LM Agents Fail to Act on Risk Knowledge](https://arxiv.org/abs/2508.13465) — >98% awareness, <26% execution

## 资源

- 训练：8×H800 (80GB each)
- 快速实验：1×RTX 5090 (`ssh 5090`)
