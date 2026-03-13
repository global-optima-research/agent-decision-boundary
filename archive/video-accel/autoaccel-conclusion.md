# AutoAccel 实验总结与方向转向

> 日期：2026-03-12
> 状态：AutoAccel 方向关闭，转向 Step Distillation

## AutoAccel 实验历程

### 时间线

| 阶段 | 日期 | 内容 | 结论 |
|------|------|------|------|
| Phase 0 | 03-07 | 注意力 profiling，头功能特化分析 | 78.5% 注意力在同帧，跨 prompt 高度稳定 (cos 0.956) |
| Phase 1 | 03-08 | 步间变化率分析 | 步间输出复用 NO GO，稀疏 mask 复用 GO |
| Phase 1a | 03-08 | 步间注意力变化率量化 | 平均变化率 12.7%，早期步高达 42.8% |
| Phase 0.5 | 03-10 | SageAttn + FBC 基线 | SageAttn RTX 5090 上 6.2x 更慢；FBC 1.26x @ SSIM=0.90 |
| Phase 0.75 | 03-11 | 全层 + 选择性 sparse | 全层 SSIM=0.34，选择性最佳 0.57 |
| Phase 0.75b | 03-12 | Step-adaptive sparse | 最后 5 步 SSIM=0.83 @ 1.01x，不实用 |
| Phase 0.8 | 03-12 | PAB-style 注意力输出缓存 | N=2 SSIM=0.55，NO GO |

### 技术可用性总结

| 技术 | SSIM (最佳) | 加速 | 可用？ | 不可用原因 |
|------|-----------|------|--------|-----------|
| FBC (block cache) | 0.90 | 1.26x | ✅ | — |
| SageAttention2 (quant) | N/A | 0.16x (反向) | ❌ | RTX 5090 Blackwell 架构不兼容 |
| Frame-local sparse | 0.83 | 1.01x | ❌ | 跨帧注意力不可丢弃；3 轮实验全面 NO GO |
| Attention output cache | 0.55 | 1.22x | ❌ | 步间注意力输出变化太大 |

### 交互效应总结

虽然技术不可用，但交互分析揭示了有价值的模式：

| 技术对 | 交互比率 | 类型 | 意义 |
|-------|---------|------|------|
| quant × cache (FBC) | 0.99 | 正交 | 两种技术独立 |
| sparse × cache (FBC) | 1.04-1.07 | 超加性 | 协同效应 |
| attn_cache × FBC (统一) | 0.91-0.99 | 亚加性→正交 | 间隔小时冲突 |
| attn_cache × FBC (金字塔) | 1.08 | 超加性 | 层级差异化带来协同 |

### AutoAccel 不可行的根本原因

1. **技术空间坍塌**：4 种候选技术中只有 FBC 1 种可用，AutoAccel 需要 ≥2 种技术才能做组合搜索
2. **硬件限制**：RTX 5090 (Blackwell sm_120) 不被 SageAttention2 支持，消灭了 quant 这条线
3. **Wan 2.1 注意力特性**：跨帧注意力在每层每步都是必需的（"注意力权重分布 ≠ 信息重要性分布"），无法通过 sparse/cache 跳过
4. **Training-free 天花板低**：唯一可用的 FBC 仅 1.26x，远低于 training-based 方法的 6-12x

## 方向评估

### 评估的候选方向

| 方向 | 加速潜力 | 资源匹配 | 论文故事 | 竞争度 | 综合 |
|------|---------|---------|---------|--------|------|
| A. 迁移 H800 继续 AutoAccel | 2-3x | 中 | 弱（仅 2 种技术） | 中 | ⭐⭐ |
| B. Adaptive FBC 深挖 | 1.3-1.5x | 高 | 弱（增量改进） | 低 | ⭐⭐ |
| **C. Step Distillation** | **6-12x** | **高（8×H800）** | **强** | **中** | **⭐⭐⭐⭐⭐** |
| D. 分析论文 | N/A | 高 | 中（负面结果） | 低 | ⭐⭐⭐ |

### 选定方向：Step Distillation for Video DiT

#### 为什么选 Step Distillation

1. **加速倍数差距悬殊**：Training-free 天花板 1.26x vs Distillation 6-12x（50 步→4-8 步）
2. **资源完美匹配**：8×H800 是做 distillation 的核心资源，之前一直没用上
3. **视频领域 gap 大**：图像 DiT 蒸馏已成熟（SDXL-Turbo, LCM, DMD），但视频 DiT 上：
   - AnimateLCM：较早工作，基于 U-Net 不是 DiT
   - CausVid (CVPR'25)：DMD 框架，但只做了 SVD 模型
   - rCM (ICLR'26)：NVIDIA，但主要面向大规模（需 256 GPU）
   - **Wan 2.1 上还没有公开的蒸馏方案**
4. **时间线充裕**：目标 MLSys 2027（截稿 ~2026.10），还有 ~7 个月
5. **AutoAccel 经验可复用**：对注意力结构的深入理解可指导蒸馏 loss 设计

#### 差异化思路

现有视频蒸馏工作的主要问题是**时间一致性下降**。我们的独特优势：

- Phase 0 发现的头功能特化（78 spatial + 8 global + 2 temporal + 272 mixed）
- Phase 1 量化的步间注意力变化规律（早期 42.8%，后期 2.3%）
- 跨帧注意力不可替代的定量证据

这些可以转化为：
1. **Temporal Consistency Loss**：基于跨帧注意力 pattern 的一致性约束
2. **功能感知蒸馏**：对 temporal/global 类型的头施加更强的 KD loss
3. **自适应步数**：根据 prompt 复杂度动态选择步数（简单 prompt 2 步，复杂 prompt 8 步）

#### 初步计划

| 阶段 | 时间 | 内容 |
|------|------|------|
| 1. 文献调研 | 1 周 | 精读 CausVid, rCM, AnimateLCM, DMD2 |
| 2. Baseline 复现 | 2 周 | 在 Wan 1.3B 上实现基础 consistency distillation |
| 3. 时间一致性 loss | 3 周 | 设计并验证 temporal consistency loss |
| 4. 扩展实验 | 3 周 | Wan 14B, 消融实验, 对比实验 |
| 5. 论文撰写 | 2 周 | 撰写投稿 |

## 副产品：分析论文

AutoAccel 的实验数据可以整理为 workshop 短文：

**标题候选**："Why Training-Free Attention Acceleration Fails for Video DiTs: A Systematic Study"

核心贡献：
- 4 种 training-free 技术在视频 DiT 上的系统性评估
- "注意力权重 ≠ 信息重要性"的定量证据
- 多技术交互效应的度量框架（正交/亚加性/超加性）

目标：NeurIPS 2026 Workshop 或 TMLR 短文。

## 保留的资产

以下来自 AutoAccel 的代码和数据在新方向中仍有价值：

| 资产 | 用途 |
|------|------|
| `scripts/autoaccel/` | 评估框架（SSIM 计算、warmup、多 prompt 统计） |
| Phase 0 注意力 profiling 数据 | 指导蒸馏 loss 设计 |
| Phase 1 步间变化率数据 | 理解蒸馏目标的时间动态 |
| 交互分析框架 | 评估蒸馏 + FBC 的组合效果 |
| 实验报告模板和工作流 | 直接复用 |

---

## 更新：Step Distillation 方向同样不可行（2026-03-12）

经文献调研，Step Distillation 方向也已拥挤（详见 `step-distillation-survey.md`）：

1. **纯蒸馏**：CausVid (CVPR'25), rCM (ICLR'26), TMD (2026.01) 三篇已覆盖 Wan 2.1
2. **蒸馏×推理加速组合**：TurboDiffusion (100-200x), DisCa, Video-BLADE 等 7+ 篇已覆盖
3. **自动搜索**：DiffAgent (2026.01) 用 LLM 自动搜索最优加速组合

**结论：视频推理加速整个大方向（单技术、蒸馏、组合、自动搜索）全部已被充分覆盖。需要彻底转换赛道。**
