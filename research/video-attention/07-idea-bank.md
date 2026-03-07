# 论文方向备选库

> 日期：2026-03-07
> 状态：待评估，按优先级排列

---

## 方向 B：全栈注意力组合优化（系统方向）

> 适合：偏系统/工程的同学，MLSys 投稿

### 一句话

现有视频 DiT 注意力加速方法各做一个维度（稀疏/量化/缓存/蒸馏），但没人研究**怎么最优地组合它们**。本工作提出一个自动化框架，给定质量约束和硬件配置，搜索最优的多技术组合方案。

### 动机

单技术加速的收益：

| 技术 | 代表工作 | 单独加速 | 需要训练？ |
|------|---------|---------|----------|
| 稀疏注意力 | STA, Sparse-vDiT, SVG | 1.5-2x | 否 |
| 量化 | SageAttention, FP8 PTQ | 2-4x | 否 |
| 特征缓存 | TeaCache, AdaCache | 2-7x | 否 |
| 步数蒸馏 | DCM, DOLLAR | 12-50x | 是 |
| 系统优化 | torch.compile, USP | 1.5-2.65x | 否 |

理论上叠加可达 50-200x，但实际叠加时：
- 稀疏 + 量化可能互相干扰（QuantSparse 发现 "sparsity-induced loss exacerbates quantization noise"）
- 缓存 + 稀疏的交互没人研究过
- 不同技术在不同 (step, layer) 的最优配置不同
- 没有自动化工具帮你选最优组合

### 核心贡献

1. **组合空间建模**：定义 N 种技术的配置空间，每个 (step, layer, head) 可选不同技术组合
2. **质量-速度 Pareto 前沿搜索**：给定质量约束 τ (SSIM/FVD)，自动搜索最优配置
3. **干扰分析**：量化技术间的正/负交互效应（哪些能叠加，哪些会冲突）
4. **自动化工具**：输入模型 + 硬件 + 质量要求 → 输出最优配置

### 方法 sketch

```
输入: 视频 DiT 模型 M, 硬件配置 H, 质量阈值 τ

Phase 1: 单技术 profiling (自动化, ~1h)
  对每种技术 T_i:
    对每个 (layer, head):
      测量: 质量损失 ΔQ_i(l,h), 加速比 S_i(l,h)
    → 得到 per-technique per-position 的 (ΔQ, S) 表

Phase 2: 交互效应 profiling (~2h)
  对每对技术 (T_i, T_j):
    测量联合应用时的 ΔQ_{i,j} vs ΔQ_i + ΔQ_j
    → 得到交互矩阵 (正交/协同/冲突)

Phase 3: 组合优化
  min  Σ latency(config(l,h,t))
  s.t. quality(full_config) ≥ τ
       config(l,h,t) ∈ {none, sparse_only, quant_only, cache_only, sparse+quant, ...}

  可用: 贪心搜索 / 进化算法 / 混合整数规划

输出: per-(step, layer, head) 的最优技术配置表
```

### 与已有工作的区别

| 工作 | 做了什么 | 没做什么 |
|------|---------|---------|
| TurboDiffusion | 叠加 rCM+SLA+W8A8 | 手工组合，非自动搜索 |
| QuantSparse | 量化+稀疏联合 | 只做了 2 种技术，无缓存 |
| FPSAttention | FP8+稀疏 co-design | 只做了 2 种技术 |
| FastVideo | TeaCache+SageAttn+并行 | 工程集成，无最优性分析 |
| **本工作** | **N 种技术自动最优组合** | — |

### 预期结果

- 自动搜索的组合 > 任何手工组合 (Pareto 前沿上)
- 发现一些反直觉的组合规则（例如某些层稀疏+缓存比单独任一都好）
- 提供开源工具，对新模型可自动生成最优配置

### 实验计划

| 实验 | 内容 | 预计耗时 |
|------|------|---------|
| 单技术 baseline | STA/SVG/TeaCache/SageAttn 各自效果 | 3天 |
| 两两组合 | 6 种两两组合的效果 | 5天 |
| 自动搜索 | 实现搜索框架 + 跑优化 | 1周 |
| 多模型验证 | Wan 1.3B/14B, HunyuanVideo, CogVideoX | 1周 |
| 消融实验 | 去掉搜索 vs 手工组合 vs 随机组合 | 3天 |

### 投稿目标

- MLSys 2027 (系统方向最佳)
- 或 OSDI/SOSP 2027 (如果系统贡献够强)
- 备选: NeurIPS 2026 Systems track

### 难度评估

- 工程量大（需要集成多种技术）
- 理论难度中等（组合优化是成熟领域）
- 实验成本高（大量 GPU 时间）
- 适合 2-3 人合作

---

## 方向 A：注意力计算分配的最优性理论

> 适合：偏理论的同学
> 状态：待深入调研

（占位，后续填充）

## 方向 C：特定场景应用

> 适合：偏应用的同学
> 状态：待深入调研

（占位，后续填充）

## 方向 D：架构级替换（线性注意力/SSM）

> 适合：偏模型设计的同学
> 状态：待深入调研

（占位，后续填充）

---

## 竞品全景（2025-2026）

完整的竞品列表，供所有方向参考：

### 稀疏注意力
- [STA](https://arxiv.org/abs/2502.04507) — 固定 tile, ICML'25, kernel 10.45x
- [Sparse-vDiT](https://arxiv.org/abs/2506.03065) — 离线 per-head 3 模式, 1.58-1.85x
- [SVG 1&2](https://github.com/svg-project/Sparse-VideoGen) — 在线 profiling, ICML'25 + NeurIPS'25 Spotlight
- [AdaSpa](https://arxiv.org/abs/2502.21079) — 在线 per-head 自适应, ICCV'25, 1.59-2.04x
- [VSA](https://arxiv.org/abs/2505.13389) — 可训练稀疏, NeurIPS'25, 2.53x FLOPs
- [XAttention](https://github.com/mit-han-lab/x-attention) — antidiagonal block sparse, ICML'25
- [LiteAttention](https://arxiv.org/abs/2511.11062) — 跨步稀疏复用

### 量化
- [SageAttention2](https://arxiv.org/abs/2505.11568) — Q/K INT4, 3.9x
- [FPSAttention](https://arxiv.org/abs/2506.04648) — FP8+稀疏 co-design, NeurIPS'25 Spotlight, 7.09x kernel
- [ViDiT-Q](https://arxiv.org/abs/2501.00487) — W8A8 无损, ICLR'25

### 缓存
- [TeaCache](https://arxiv.org/abs/2411.19108) — 零开销预测, CVPR'25, 6.83x
- [AdaCache](https://arxiv.org/abs/2411.02397) — 运动感知, ICCV'25, 4.7x
- [DuCa](https://arxiv.org/abs/2412.18911) — 随机选择≈重要性选择

### 联合
- [QuantSparse](https://arxiv.org/abs/2509.23681) — 量化+稀疏联合, 3.68x
- [Video-BLADE](https://arxiv.org/abs/2508.10774) — block sparse + 蒸馏
- [TurboDiffusion](https://arxiv.org/abs/2512.10590) — rCM+SLA+W8A8, 100-200x

### 系统
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) — SGLang 集成, 全链路
- [FastGen](https://github.com/NVIDIA/FastGen) — NVIDIA, 8+ 蒸馏方法统一
