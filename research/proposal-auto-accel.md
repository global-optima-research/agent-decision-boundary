# Proposal: AutoAccel — Automated Multi-Technique Acceleration for Video Diffusion Transformers

> 日期：2026-03-09
> 目标：MLSys 2027（MLSys 2026 截稿为 2025.10.30，预计 MLSys 2027 截稿 ~2026 年 10 月）
> 资源：8×H800 (训练/大模型) + 1×RTX 5090 (实验/profiling)

## 一句话

现有视频 DiT 加速技术各做一个维度（稀疏/量化/缓存），组合时全靠手工。我们提出 AutoAccel：给定质量约束和硬件配置，自动搜索 per-(layer, step) 的最优多技术组合方案。

## 动机

### 现状：单技术各自为战

| 技术类别 | 代表工作 | 单独加速 | 训练？ |
|---------|---------|---------|--------|
| 稀疏注意力 | STA (ICML'25), SVG (ICML'25) | 1.5-2x | 否 |
| 注意力量化 | SageAttention2, FPSAttention (NeurIPS'25) | 2-4x | 否 |
| 特征缓存 | TeaCache (CVPR'25), AdaCache (ICCV'25) | 2-7x | 否 |

理论叠加可达 10-50x，但实际叠加时：
- **技术间存在干扰**：QuantSparse 发现 "sparsity-induced loss exacerbates quantization noise"
- **最优配置因位置而异**：不同 (layer, step) 对不同技术的容忍度不同
- **没有自动化工具**：现有组合全靠手工设计

### 已有组合工作的局限

| 工作 | 技术数 | 组合方式 | per-(layer,step)？ | 搜索？ | 干扰分析？ |
|------|--------|---------|-------------------|--------|-----------|
| QuantSparse | 2 (quant+sparse) | 联合设计 | ❌ | ❌ | ✅ 2 技术 |
| QuantCache (ICCV'25) | 3 (quant+cache+prune) | 手工层级 pipeline | 部分 | ❌ | ❌ |
| FPSAttention (NeurIPS'25) | 2 (FP8+sparse) | 规则 per-step | ✅ step 级 | ❌ | ❌ |
| TurboDiffusion | 4 (rCM+SLA+SageAttn+W8A8) | 全手工 | ❌ | ❌ | ❌ |
| FastVideo | 3+ (TeaCache+SageAttn+并行) | 工程集成 | ❌ | ❌ | ❌ |
| **AutoAccel (本工作)** | **3 (sparse+quant+cache)** | **自动搜索** | **✅ (layer, step)** | **✅** | **✅ N×N** |

### 方法论先例

**FlexGen (MLSys 2023)** 在 LLM 推理中用 LP 搜索 offloading × quantization × batching 的最优配置。我们将类似的思路应用到视频 DiT 的注意力加速领域。

## 核心贡献

1. **技术交互矩阵**：系统量化 sparse × quant × cache 三两两和三三组合的协同/冲突效应
2. **自动配置搜索**：给定质量约束 τ 和硬件 H，搜索 per-(layer, step) 最优技术配置
3. **Pareto 前沿分析**：质量-速度 tradeoff 的完整刻画
4. **开源工具**：输入模型 + 硬件 + 质量要求 → 输出最优配置

## 方法

### 概览

```
输入: 视频 DiT 模型 M, 硬件 H, 质量阈值 τ (VBench/SSIM/FVD)
                    ↓
   Phase 1: 单技术 Profiling (~1-2h)
                    ↓
   Phase 2: 交互效应 Profiling (~2-4h)
                    ↓
   Phase 3: 配置搜索 (minutes)
                    ↓
输出: per-(layer, step) 最优配置表 C*
```

### Phase 1: 单技术 Profiling

对 3 种 training-free 技术分别测量 per-(layer, step) 的质量损失和加速比：

**稀疏注意力**（选 SVG 或 STA 作为代表）：
- 对每层 l、每步 t：应用稀疏 → 测量 MSE(O_sparse, O_full) 和 kernel 延迟
- 输出：ΔQ_sparse(l, t), S_sparse(l, t)

**注意力量化**（SageAttention2）：
- 对每层 l、每步 t：应用 INT4/FP8 量化 → 测量 MSE 和延迟
- 输出：ΔQ_quant(l, t), S_quant(l, t)

**特征缓存**（TeaCache）：
- 对每步 t：判断是否可安全跳过 → 测量 SSIM 损失和节省时间
- 输出：ΔQ_cache(t), S_cache(t)

### Phase 2: 交互效应 Profiling

对每对技术组合测量**实际联合损失** vs **假设独立损失之和**：

```
交互效应 I(T_i, T_j, l, t) = ΔQ_{i+j}(l,t) - [ΔQ_i(l,t) + ΔQ_j(l,t)]

I > 0 → 冲突（联合损失 > 独立损失之和）
I ≈ 0 → 正交（可安全叠加）
I < 0 → 协同（联合反而更好）
```

测量 3 个两两组合 + 1 个三三组合：
- sparse + quant → 预期冲突（QuantSparse 已发现）
- sparse + cache → 未知
- quant + cache → 未知
- sparse + quant + cache → 未知

### Phase 3: 配置搜索

**配置空间**：每个 (layer l, step t) 可选以下模式之一：

```
config(l, t) ∈ {
  none,           # 原始 full attention
  sparse,         # 仅稀疏
  quant,          # 仅量化
  cache,          # 仅缓存（跳过该步）
  sparse+quant,   # 稀疏+量化
  sparse+cache,   # 稀疏+缓存
  quant+cache,    # 量化+缓存
  sparse+quant+cache  # 全叠加
}
```

对 Wan 2.1-1.3B (30 layers × 50 steps)：搜索空间 = 8^1500（不可穷举）

**搜索算法**（从简单到复杂，选效果最好的）：

方案 A：**贪心 layer-by-layer**
```
for each layer l (按敏感度排序):
  for each step t:
    选使 latency 下降最多且 ΔQ ≤ budget 的 config
```

方案 B：**约束优化（参考 FlexGen 的 LP 方法）**
```
min  Σ_{l,t} latency(config(l,t))
s.t. Σ_{l,t} ΔQ(config(l,t)) + Σ_{l,t,l',t'} I(config) ≤ τ
```
用交互效应矩阵修正质量估计，使约束更准确。

方案 C：**进化算法**
- 种群 = 随机配置表
- 适应度 = speed / max(0, quality - τ)
- 交叉 + 变异 → 迭代 → 收敛

### 质量评估

- **快速评估**（搜索时用）：MSE / LPIPS（单帧，快，可微分）
- **完整评估**（最终报告）：VBench, FVD, SSIM, 人工评估

## 实验计划

### 模型

| 模型 | 参数量 | 硬件 | 用途 |
|------|--------|------|------|
| Wan 2.1-T2V-1.3B | 1.3B | RTX 5090 | 主实验 + profiling + 搜索 |
| Wan 2.1-T2V-14B | 14B | 8×H800 | 扩展验证 |
| HunyuanVideo (可选) | 13B | 8×H800 | 跨模型泛化 |

### 具体技术选择

| 类别 | 选择 | 原因 |
|------|------|------|
| 稀疏 | SVG (ICML'25) | 开源、per-head 自适应、Wan 2.1 支持 |
| 量化 | SageAttention2 | 开源、成熟、INT4 Q/K |
| 缓存 | TeaCache (CVPR'25) | 开源、零开销预测、广泛使用 |

备选/扩展：STA (稀疏)、FPSAttention (量化+稀疏)、AdaCache (缓存)

### 实验步骤

| 阶段 | 内容 | 耗时 | 产出 |
|------|------|------|------|
| **E0: 环境搭建** | 在 Wan 2.1-1.3B 上跑通 SVG + SageAttn2 + TeaCache 各自的 baseline | 2-3 周 | 3 个单技术的 quality-speed 数据点 |
| **E1: 单技术 Profiling** | per-(layer, step) 的 ΔQ 和 S 测量 | 1 周 | 3 个 30×50 的热力图 |
| **E2: 两两组合** | 3 种两两组合的效果 + 交互效应矩阵 | 2 周 | 交互矩阵 + 6 个数据点 |
| **E3: 搜索框架** | 实现贪心/LP/进化搜索 | 2 周 | 搜索代码 + 最优配置 |
| **E4: 端到端评估** | 搜索结果 vs 手工组合 vs 单技术 | 1 周 | Pareto 前沿图 |
| **E5: 大模型验证** | Wan 14B 上复现 | 2 周 | 跨规模泛化数据 |
| **E6: 消融** | 搜索 vs 无搜索、交互修正 vs 无修正 | 1 周 | 消融表 |
| **总计** | | **~10 周** | |

### 关键实验（决定论文成败）

**E2 是最关键的实验。** 如果三种技术之间基本正交（I ≈ 0），那么：
- 好的方面：组合安全，"全叠加"就是最优
- 坏的方面：搜索没有意义，论文核心贡献崩塌

**E2 的理想结果**：存在显著的非正交交互（某些组合冲突、某些协同），且交互效应因 (layer, step) 而异 → 搜索有价值。

**E2 的止损点**：如果所有交互 |I| < 0.05（几乎完全正交），则放弃搜索贡献，转为写"视频 DiT 加速技术组合实证研究"（贡献降级但仍可发表）。

## 预期 Story

### 理想结果

```
发现 1: sparse + quant 在中间层冲突 (I > 0.1)，但在首尾层协同 (I < -0.05)
发现 2: cache + sparse 在后期步协同，在早期步冲突
发现 3: 搜索配置比最佳手工组合快 1.3-2x，同等质量
发现 4: 配置可跨 prompt 复用 (Phase 0 已验证不变性)
```

### 论文结构

```
1. Introduction: 多技术组合是实际需求，但没有系统化方法
2. Background: 3 类技术回顾 + 已有组合工作的局限
3. Method:
   3.1 配置空间定义
   3.2 单技术 profiling
   3.3 交互效应分析
   3.4 搜索算法
4. Experiments:
   4.1 单技术 baseline (Table 1)
   4.2 交互效应矩阵 (Figure 2 — 核心图)
   4.3 搜索 vs 手工组合 (Table 2 — 核心表)
   4.4 Pareto 前沿 (Figure 3)
   4.5 Wan 14B 扩展 (Table 3)
   4.6 消融 (Table 4)
5. Analysis: 反直觉发现、设计准则
6. Related Work
7. Conclusion + 开源工具
```

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 技术间完全正交，搜索无意义 | 中 | 致命 | E2 后判断，转为实证研究论文 |
| 开源代码集成困难 | 高 | 延迟 | 优先选代码质量好的（SageAttn 最成熟） |
| 搜索找到的配置不比手工好 | 中 | 严重 | 确保手工 baseline 是合理的（非 oracle） |
| 审稿人认为是工程贡献 | 中 | 拒稿 | 强调统一抽象（参考 FlashInfer），而非单纯堆叠 |
| USV 抢占叙事 | 中 | 削弱新颖性 | 强调 training-free + 包含量化 + 显式交互分析 |
| 截稿前出现更多组合工作 | 高 | 削弱新颖性 | 快速执行，尽早投 arXiv 占位 |
| 质量评估指标不够 | 低 | 弱化 | 用 VBench + FVD + SSIM + 人工评估 |

## 快速验证计划（Phase 0.5）

**在全面投入前，用 1 周做最小验证**：

```
在 Wan 2.1-1.3B, RTX 5090 上:
1. 跑通 SVG → 记录 speed + quality
2. 跑通 SageAttention2 → 记录 speed + quality
3. 跑通 TeaCache → 记录 speed + quality
4. 手工叠加 SVG + SageAttn2 → 看是否能跑通 + quality 下降多少
5. 手工叠加 SVG + TeaCache → 同上
```

**判断标准**：
- 如果 (4) 和 (5) 能跑通且质量损失可接受 → 继续
- 如果代码层面无法集成 → 换技术选择或放弃

## 竞品对比（经验证）

### 直接竞品

| 工作 | 技术 | 方式 | 与我们的差异 |
|------|------|------|------------|
| QuantCache (ICCV'25) | 3 (quant+cache+prune) | 手工层级 heuristic | 无搜索，只在 Open-Sora |
| TurboDiffusion | 4 (rCM+SLA+SageAttn+W8A8) | 全手工 | 无 per-position 优化 |
| QuantSparse | 2 (quant+sparse) | 固定设计 | 只 2 技术，无缓存 |
| FPSAttention (NeurIPS'25) | 2 (FP8+sparse) | 手工 per-step 规则 | 只 2 技术，非搜索 |
| PAROAttention | 2 (sparse+quant) | 固定设计 | 只 2 技术，token reorder |
| FastVideo | 3+ (TeaCache+SageAttn+并行) | 工程集成 | 无最优性分析 |
| **⚠️ USV** | **3 (sparse+token merge+step reduce)** | **端到端学习 policy** | **最近竞品，见下方详析** |
| DiffAgent (2026.01) | N/A | LLM+遗传算法生成代码 | 代码级自动化，非配置搜索 |

### ⚠️ USV 详细对比（最接近的竞品）

**USV: Unified Sparsification** (arXiv:2512.05754)
- 联合优化 attention sparsity + token merging + sampling step reduction
- **端到端训练**学习 dynamic, data- and timestep-dependent sparsification policy
- 声称 "super-additive efficiency gains"
- 83.3% denoising speedup

**关键差异**：
| 维度 | USV | AutoAccel |
|------|-----|-----------|
| 优化方式 | 端到端训练（需要 GPU 训练） | Training-free 搜索 |
| 技术类型 | 3 种稀疏化维度 | sparse + quant + cache（包含量化） |
| 搜索粒度 | 学习连续 policy | 离散 per-(layer,step) 配置 |
| 交互分析 | 隐式（黑箱学习） | 显式量化交互矩阵 |
| 部署成本 | 需要针对每个模型训练 | Profile 一次即可 |

**我们的差异化必须强调**：
1. **包含量化**（USV 不做量化）— 量化是实际部署中最常用的加速手段
2. **Training-free** — 不需要训练，profile 几小时即可
3. **显式交互分析** — 提供可解释的 insight，而非黑箱
4. **实用性** — 换模型不需要重新训练，只需重新 profile

### 方法论先例

| 工作 | 关系 |
|------|------|
| **FlexGen (MLSys'23)** | LP 搜索 offloading × quant × batch（LLM 领域），我们在视频 DiT 做类似事 |
| **FlashInfer (MLSys'25 Best Paper)** | 统一注意力引擎 + composable 格式，审稿人认可"统一抽象"的贡献模式 |
| **ScaleFusion (MLSys'25)** | 视频 DiT 推理系统论文，证明这个 topic 在 MLSys 被接收 |

## 参考文献

### 核心竞品
- [QuantSparse](https://arxiv.org/abs/2509.23681) — quant + sparse 交互分析
- [QuantCache](https://arxiv.org/abs/2503.06545) — quant + cache + pruning 联合, ICCV 2025
- [TurboDiffusion](https://arxiv.org/abs/2512.16093) — 4 技术手工组合, 100-200x
- [FPSAttention](https://arxiv.org/abs/2506.04648) — FP8 + sparse co-design, NeurIPS 2025
- [PAROAttention](https://arxiv.org/abs/2506.16054) — sparse + quant via token reordering
- [USV](https://arxiv.org/abs/2512.05754) — 3 维联合稀疏化（训练学习 policy）⚠️ 最近竞品
- [DiffAgent](https://arxiv.org/abs/2601.03178) — LLM + 遗传算法自动生成加速代码 (2026.01)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) — 工程集成框架

### 方法论先例
- [FlexGen](https://arxiv.org/abs/2303.06865) — LP 搜索多技术配置, MLSys 2023

### 拟集成的技术
- [SVG](https://arxiv.org/abs/2502.01776) — 稀疏注意力, ICML 2025
- [SageAttention2](https://arxiv.org/abs/2505.11568) — INT4 注意力量化
- [TeaCache](https://arxiv.org/abs/2411.19108) — 特征缓存, CVPR 2025
