# 特征缓存 (Feature Caching) 深入调研

> 调研日期：2026-03-06

## 目录

- [为什么特征缓存是最实用的加速方向](#为什么特征缓存是最实用的加速方向)
- [缓存粒度分类体系](#缓存粒度分类体系)
- [技术演进脉络](#技术演进脉络)
- [重点方法详解](#重点方法详解)
  - [PAB (Pyramid Attention Broadcast)](#1-pab-pyramid-attention-broadcast)
  - [AdaCache](#2-adacache)
  - [TeaCache](#3-teacache)
  - [DuCa (Dual Feature Caching)](#4-duca-dual-feature-caching)
  - [FastCache](#5-fastcache)
  - [MixCache](#6-mixcache)
  - [FasterCache](#7-fastercache)
  - [EOC (Error-Optimized Cache)](#8-eoc-error-optimized-cache)
- [其他方法速览](#其他方法速览)
- [方法横向对比](#方法横向对比)
- [视频缓存 vs 图像缓存的独特挑战](#视频缓存-vs-图像缓存的独特挑战)
- [选型建议](#选型建议)

---

## 为什么特征缓存是最实用的加速方向

| 特性 | 缓存 | 蒸馏 | 量化 | 剪枝 |
|------|------|------|------|------|
| 需要训练 | **否** | 是 | 部分 | 部分 |
| 架构无关 | **是** | 否 | 部分 | 否 |
| 即插即用 | **是** | 否 | 部分 | 否 |
| 与其他方法正交 | **是** | 是 | 是 | 部分 |
| 典型加速比 | 2-7x | 10-200x | 1.5-3x | 1.5-3x |

**核心优势**：零训练成本 + 即插即用 + 可与蒸馏/量化叠加。

**基本原理**：扩散模型相邻去噪步骤的中间特征高度相似（通常 >97%），大量计算是冗余的。通过缓存并复用这些特征，可以跳过不必要的计算。

---

## 缓存粒度分类体系

```
缓存粒度（由粗到细）
│
├── Step 级 ── 整个模型输出跨步复用
│   └── 最激进，速度最快，质量风险最大
│
├── CFG 级 ── conditional/unconditional 两路之间复用
│   └── 利用 classifier-free guidance 两路的高相似性
│
├── Block 级 ── 单个 transformer block 输出跨步复用
│   └── 不同 block 变化速率不同，可差异化缓存
│
└── Token 级 ── 单个 token 在单个 block 中跨步复用
    └── 最细粒度，精度最高，但实现复杂
```

### 缓存决策方式演进

```
静态缓存 (固定间隔复用)
  │
  ▼
动态缓存 (基于相似度阈值)
  │  ├── 输出侧检测 (需要完整前向传播)
  │  └── 输入侧预测 (TeaCache，近乎零开销)
  │
  ▼
预测性缓存 (不只复用，还预测)
  │  ├── 线性近似 (FastCache)
  │  ├── Taylor 展开 (TaylorSeer)
  │  └── Adams-Bashforth 数值方法 (AB-Cache)
  │
  ▼
混合缓存 (多粒度自适应选择)
     └── MixCache: step + cfg + block 动态切换
```

---

## 技术演进脉络

```
2024.08  PAB (Pyramid Attention Broadcast)
          │  发现：spatial/temporal/cross attention 变化频率不同
          │  → 金字塔式差异化广播间隔
          ▼
2024.10  FasterCache (ICLR 2025)
          │  从"复用"进化到"插值预测" + CFG-Cache
          ▼
2024.11  AdaCache (ICCV 2025)
          │  "Not all videos are created equal"
          │  → 内容自适应 + 运动感知 (MoReg)
          ▼
2024.11  TeaCache (CVPR 2025)
          │  用 timestep embedding 做零开销输入侧预测
          │  → 被 FastVideo 集成为标配
          ▼
2024.11  Streamlined Inference (NeurIPS 2024)
          │  Feature Slicer + Operator Grouping + Step Rehash
          │  → 内存优化 + 时间步复用
          ▼
2024.12  DuCa (Dual Feature Caching)
          │  颠覆性发现：随机选 token ≈ 重要性选择
          │  → 激进/保守交替 + FlashAttention 兼容
          ▼
2025.01  EOC (Error-Optimized Cache)
          │  不关心"在哪缓存"而关心"如何修正缓存误差"
          │  → 可叠加到任何缓存方法上
          ▼
2025.05  FastCache
          │  统计假设检验 + 可学习线性近似 + 理论误差界
          ▼
2025.08  MixCache (Mixture-of-Cache)
          │  Step/CFG/Block 三粒度自适应混合
          └── 当前多粒度混合缓存 SOTA
```

---

## 重点方法详解

### 1. PAB (Pyramid Attention Broadcast)

> [论文](https://arxiv.org/abs/2408.12588) | 2024.08

**核心发现**：视频 DiT 中三种注意力的时间步变化频率不同。

```
变化频率：Spatial > Temporal > Cross-Attention
                                    │
复用间隔：  短        中         长
           (每2步)   (每4步)    (每6步)
                                    │
              ┌─────────────────────┘
              ▼
        金字塔广播策略 (PAB-246)
```

**关键洞察**：注意力差异在去噪中间阶段（~70% 步骤）呈 **U 形模式**——中间稳定，首尾变化大。

| 配置 | 加速比 | 说明 |
|------|--------|------|
| PAB-246 (8×H100) | **10.6x** | 含广播序列并行 |
| PAB-fast (单卡) | 1.34-1.56x | 单卡基准 |

**局限**：固定间隔，无法适应不同内容的运动复杂度。被后续 AdaCache、TeaCache 大幅超越。

**地位**：奠基性工作，首次揭示视频 DiT 三种注意力的差异化缓存潜力。

---

### 2. AdaCache

> Meta，ICCV 2025 | [论文](https://arxiv.org/abs/2411.02397) | [项目页](https://adacache-dit.github.io/) | [代码](https://github.com/AdaCache-DiT/AdaCache)

**核心洞察**：**"Not all videos are created equal"** — 静态场景需要更少计算，高运动场景需要更多。

#### 工作机制

```
步骤 t:
1. 计算 DiT block 残差 p_t^l (temporal attention)
2. 计算 L1 距离: c_t = ||p_t - p_{t+k}||₁ / k
3. 查码本: 距离 → 缓存步数
   ┌──────────┬────────────┐
   │ 距离阈值  │ 复用步数    │
   ├──────────┼────────────┤
   │ 0.08     │ 6 (变化小)  │
   │ 0.16     │ 5          │
   │ 0.32     │ 3          │
   │ 1.00     │ 1 (变化大)  │
   └──────────┴────────────┘
4. 后续 τ 步直接复用缓存残差
```

#### Motion Regularization (MoReg)

```
motion_score   = ||frames[i:N] - frames[0:N-i]||₁  (帧间差异)
motion_gradient = (motion_score_t - motion_score_{t+k}) / k  (运动趋势)

最终距离 = c_t × (motion_score + motion_gradient)
           ↑ 高运动 → 距离膨胀 → 复用更少 → 更多计算
```

#### 性能数据

| 模型 | VBench | 加速比 | 对比 PAB |
|------|--------|--------|----------|
| Open-Sora 480p | 79.39% (+0.17) | **2.24x** | PAB: 1.34x |
| Open-Sora 720p | — | **4.7x** | PAB: 1.66x |
| Open-Sora-Plan | 79.30% | **3.53x** | — |
| 8卡 A100 | — | **9.22x** | 含并行收益 |

FLOPs 减少 **59%**，用户偏好测试 **70%** 偏好 AdaCache 而非 PAB。

#### 局限

- 码本需要手动设计，新模型/分辨率需重新调整
- 仅 2-2.7 秒短视频验证
- 激进缓存时仍有时间不一致伪影

---

### 3. TeaCache

> 阿里达摩院，CVPR 2025 | [论文](https://arxiv.org/abs/2411.19108) | [代码](https://github.com/ali-vilab/TeaCache)

**核心创新**：用 **timestep embedding 调制后的输入差异** 预测输出差异，几乎零额外开销。

#### 工作机制

```
传统方法：计算完整前向传播 → 比较输出 → 决定缓存（开销大）

TeaCache：比较输入侧信号 → 多项式拟合校准 → 累积阈值决策
          ↑ 成本极低                         ↑ 无需前向传播
```

**三步流程**：

1. **输入侧差异估计**：`L1_rel(F, t) = ||F_t - F_{t+1}||₁ / ||F_{t+1}||₁`
2. **多项式拟合校准**：4阶多项式将输入差异映射为输出差异估计
3. **累积阈值决策**：累积校准后的差异值，超过阈值 δ 才触发重新计算

#### 性能数据

| 模型 | 加速比 | VBench | 说明 |
|------|--------|--------|------|
| Open-Sora-Plan (65f) | **4.41-6.83x** | 80.32-79.72% | 远超 PAB 1.56x |
| Open-Sora (51f) | **1.55-2.25x** | 79.28-78.48% | |
| Latte (16f) | **1.86-3.28x** | 77.40-76.69% | |
| 8×A800 多卡 | **32.02x** | 维持 | 含并行扩展 |

**效率-质量平衡最佳**：6.83x 加速仅 -0.07% VBench 下降。

#### 生态集成

```python
# FastVideo 中使用 TeaCache（一行开启）
video = generator.generate_video(prompt, enable_teacache=True)
```

- **FastVideo V1** 标配组件，结合 SageAttention 实现 ~3x 推理加速
- AMD ROCm 适配：Wan2.1 从 118s → **72s** (-39%)
- 零代码修改：hooks 方式包装模型 forward pass

---

### 4. DuCa (Dual Feature Caching)

> 2024.12 | [论文](https://arxiv.org/abs/2412.18911)

**颠覆性发现**：**随机选择 token ≈ 重要性选择**，且更兼容 FlashAttention。

#### 激进/保守交替策略

```
步骤 1: Fresh    ── 全量计算，更新缓存
步骤 2: Aggressive ── 整个 block 跳过，直接用缓存
步骤 3: Conservative ── 随机选 10% token 计算，90% 用缓存
步骤 4: Aggressive
步骤 5: Conservative
...重复 (推荐周期 N=5, 缓存率 R=0.9)
```

#### 为什么随机选择有效

| 选择方法 | Image Reward (FLUX) |
|----------|---------------------|
| Attention scores (ToCa) | 0.9798 |
| K-norm | 0.9783 |
| Similarity (min) | 0.9811 |
| **Random** | **0.9806** |

> "Token selection should focus on their **duplication** instead of their **importance**."

随机选择天然保证**多样性**——不同步骤计算不同 token，避免同一区域反复计算而其他区域永远过期。

**关键实践优势**：随机选择不需要注意力分数，**完全兼容 FlashAttention**，带来额外 ~2x 延迟加速。

#### 性能数据

| 模型 | 加速比 | 质量 |
|------|--------|------|
| OpenSora (480p) | **2.50x** | VBench 78.83 (基线 79.41) |
| FLUX.1-dev | **3.45x** | Image Reward 0.9896 (基线 0.9898) |
| DiT-XL/2 | **2.71x** | FID 3.00 |

对比 ToCa：加速比更高 (3.45x vs 3.30x)，质量更好 (0.9896 vs 0.9731)，且无网格伪影。

---

### 5. FastCache

> 2025.05 | [论文](https://arxiv.org/abs/2505.20353)

**核心创新**：统计假设检验 + 可学习线性近似 + 形式化误差界。

#### 双重策略

```
┌─────────────────────────────────────────────────┐
│ 空间维度：显著性感知 Token 缩减                    │
│                                                  │
│ S_t = ||X_t - X_{t-1}||₂²  (时间显著性)          │
│                                                  │
│ 运动 token (S > τ_s) → 保留，走 transformer      │
│ 静态 token (S ≤ τ_s) → 跳过，用线性层近似         │
│                        H_t^s = W_c·X_t^s + b_c   │
└─────────────────────────────────────────────────┘
                    ×
┌─────────────────────────────────────────────────┐
│ 时间维度：统计检验决定 Block 缓存                  │
│                                                  │
│ δ² = ||H_t - H_{t-1}||_F² / ||H_{t-1}||_F²     │
│                                                  │
│ 卡方检验: (N·D)·δ² ~ χ²_{N·D}                   │
│ δ² ≤ χ²_{N·D, 1-α} / (N·D) → 缓存此 block       │
│ 否则 → 重新计算                                   │
│                                                  │
│ 缓存时用线性近似: H_{t,l} = W_l·H_{t,l-1} + b_l  │
└─────────────────────────────────────────────────┘
```

#### 理论误差界

```
|v(X^t) - v(B̃^t)| ≤ L·(δ + ε_cache + γ) + O(δ²)
```

其中 δ=运动残差界，ε_cache=缓存预测误差，γ=时间分布偏移。这是缓存方法中**首个形式化误差保证**。

#### 性能数据 (DiT-XL/2, ImageNet 256)

| 方法 | FID | 时间 (ms) | 内存 (GB) |
|------|-----|-----------|-----------|
| Baseline | — | 25,150 | 16.4 |
| TeaCache | 5.09 | 14,953 | — |
| AdaCache | 4.64 | 21,895 | — |
| **FastCache** | **4.46** | **15,875** | **11.2** |

最佳 FID + 36.9% 时间减少 + 31.7% 内存减少。

---

### 6. MixCache

> 2025.08 | [论文](https://arxiv.org/abs/2508.12691)

**核心创新**：Step/CFG/Block 三粒度自适应混合，当前多粒度缓存 SOTA。

#### 混合策略

```
每个缓存步骤的决策流程：

1. 计算三种粒度的 P 值:
   P_t^ψ = D_t^ψ × I^ψ
   (D = 相似度距离, I = 高斯扰动分析的精度影响)

2. 选择 P 值最小的粒度
   + 惩罚系数 5x 防止连续选同一策略

3. 自适应缓存间隔:
   质量偏差 > δ₂ → 缩短间隔（多计算）
   质量偏差 < δ₁ → 延长间隔（多缓存）
```

#### 性能数据

| 模型 | 加速比 | VBench | LPIPS |
|------|--------|--------|-------|
| Wan 14B 480p | **1.94x** | 83.90 | 0.132 |
| Wan 14B 720p | **1.82x** | 83.70 | 0.146 |
| HunyuanVideo | **1.97x** | 80.98 | 0.060 |
| CogVideoX 5B | **1.73x** | 80.15 | 0.160 |

加速比略低于 TeaCache/DuCa 的激进配置，但**质量保持最好**（LPIPS 最低）。

---

### 7. FasterCache

> ICLR 2025 | [论文](https://arxiv.org/abs/2410.19355)

**核心进化**：从"复用"到"插值预测"。

```
传统缓存：直接复制 cache[t-k] → 用于步骤 t
FasterCache：在 cache[t-2k] 和 cache[t-k] 之间插值 → 预测步骤 t

output_t ≈ interpolate(cache[t-2k], cache[t-k])
```

**CFG-Cache**：利用同一时间步内 conditional 和 unconditional 输出的高相似性，仅计算一路，另一路用线性外推。

---

### 8. EOC (Error-Optimized Cache)

> 2025.01 | [论文](https://arxiv.org/abs/2501.19243)

**独特定位**：不竞争缓存选择，而是**修正任何缓存方法引入的误差**。可叠加到任意缓存方法上。

```
传统思路：找到变化小的 block → 缓存它们 (避免误差)
EOC 思路：缓存会引入误差 → 用 trend embedding 修正误差

修正公式：
F_Attn = x + Ada·(f_Attn(x)·(1 + θ·E_Attn))
                                  └── 趋势嵌入修正
```

**效果**（DiT-XL/2, DDIM-20）：

| 缓存率 | 原始 FID | + EOC FID | 改善 |
|--------|----------|-----------|------|
| 25% | 3.870 | **3.692** | 4.6% |
| 50% | 6.857 | **5.821** | 15.1% |
| 75% | 30.454 | **21.690** | 28.8% |

缓存率越高（误差越大），EOC 改善越显著。**额外开销仅 0.22-0.56%**。

---

## 其他方法速览

| 方法 | 核心思路 | 加速比 | 特点 |
|------|----------|--------|------|
| **Streamlined Inference** (NeurIPS 2024) | Feature Slicer + Step Rehash | — | 内存降 73%（41.7→11GB） |
| **Delta-DiT** | 存储增量变化而非完整输出 | 1.6x | 概念性影响大 |
| **Foresight** | 每层自适应 MSE 阈值复用 | 1.63x | PSNR 比 PAB 高 5-7 点 |
| **BWCache** | Block-wise 动态相似度缓存 | 6x | |
| **TaylorSeer** | Taylor 展开预测缓存 | — | "Cache-Then-Forecast" |
| **AB-Cache** | Adams-Bashforth 数值方法预测 | — | 高阶数值近似 |
| **HiCache** | Hermite 多项式插值 | — | 数值稳定性好 |

---

## 方法横向对比

### 性能对比总表

| 方法 | 训练需求 | 最佳加速比 | 质量保持 | FlashAttn 兼容 | 视频验证 |
|------|----------|-----------|----------|---------------|----------|
| PAB | 无 | 10.6x (多卡) | 中 | 是 | ✅ Open-Sora |
| AdaCache | 无 | 4.7x (单卡) / 9.2x (多卡) | 高 | 是 | ✅ Open-Sora, Latte |
| TeaCache | 无 | 6.83x | **极高** | 是 | ✅ Open-Sora, Latte |
| DuCa | 无 | 3.45x | 高 | **是 (原生)** | ✅ OpenSora, FLUX |
| FastCache | 轻量线性层 | ~1.6x | 最佳 FID | 是 | 部分 (DiT) |
| MixCache | 无 | ~1.97x | **极高** | 是 | ✅ Wan, HunyuanVideo |
| FasterCache | 无 | — | 高 | 是 | ✅ |
| EOC | 轻量预采样 | +叠加 | 修正型 | 是 | 部分 (DiT) |

### 按场景推荐

```
需求 → 推荐方法

快速集成，零修改     → TeaCache (FastVideo 一行开启)
极致质量保持         → MixCache (三粒度自适应)
最大单卡加速         → TeaCache aggressive / AdaCache-fast
多卡扩展             → PAB (广播并行) / AdaCache (多卡扩展好)
FlashAttention 兼容  → DuCa (随机选择，原生兼容)
高缓存率下保质量     → EOC (叠加到其他方法上)
理论保证             → FastCache (形式化误差界)
```

---

## 视频缓存 vs 图像缓存的独特挑战

| 维度 | 图像 DiT 缓存 | 视频 DiT 缓存 |
|------|--------------|--------------|
| **注意力类型** | 1种 (spatial) | **3种** (spatial/temporal/cross)，变化频率不同 |
| **运动异质性** | 无 | **有** — 静态背景 vs 运动前景需不同缓存策略 |
| **时间一致性** | 无 | **必须保持** — 缓存不当导致闪烁 |
| **内存压力** | 分辨率² | **分辨率² × 帧数** — 量级差异 |
| **冗余来源** | 步间冗余 | 步间冗余 **+ 帧间冗余** — 双重优化空间 |
| **质量退化模式** | 模糊/伪影 | 模糊/伪影 **+ 闪烁/运动不连贯** |

**视频特有优化**：
- PAB 的金字塔策略利用三种注意力的差异化特性
- AdaCache 的 MoReg 根据运动强度调整计算预算
- Streamlined Inference 的 Feature Slicer 沿空间维度切片（保留帧间信息流）

---

## 选型建议

### 快速决策树

```
你的场景是什么？
│
├── 已有 FastVideo 环境
│   └── 开启 TeaCache → 立即 3x 加速，零成本
│
├── 需要最大加速比（可接受少量质量损失）
│   └── TeaCache aggressive 或 AdaCache-fast
│       可叠加 EOC 修正误差
│
├── 质量零妥协
│   └── MixCache（三粒度自适应）
│       或 AdaCache-slow + MoReg
│
├── 研究新的缓存方法
│   └── 基于 DuCa 的随机选择 insight
│       + FastCache 的理论框架
│
└── 与蒸馏组合使用
    └── TeaCache/DuCa + 步数蒸馏 (DCM/DOLLAR)
        → 乘法加速（典型 10-50x 总加速）
```

### 组合方案示例

```
最优实践 (工程落地):
┌──────────────────────────────────────────┐
│  Step 1: 步数蒸馏 (DCM, 4步)  → ~12x    │
│  Step 2: TeaCache 缓存        → ~2x     │
│  Step 3: SageAttention        → ~1.5x   │
│  Step 4: W8A8 量化            → ~1.5x   │
│                                          │
│  总加速: 12 × 2 × 1.5 × 1.5 ≈ 54x     │
└──────────────────────────────────────────┘
```

---

## 参考资料

### 视频缓存核心论文
- [PAB: Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588)
- [AdaCache (ICCV 2025)](https://arxiv.org/abs/2411.02397) | [项目页](https://adacache-dit.github.io/) | [代码](https://github.com/AdaCache-DiT/AdaCache)
- [TeaCache (CVPR 2025)](https://arxiv.org/abs/2411.19108) | [代码](https://github.com/ali-vilab/TeaCache)
- [DuCa: Dual Feature Caching](https://arxiv.org/abs/2412.18911)
- [FastCache](https://arxiv.org/abs/2505.20353)
- [MixCache: Mixture-of-Cache](https://arxiv.org/abs/2508.12691)
- [FasterCache (ICLR 2025)](https://arxiv.org/abs/2410.19355)
- [EOC: Error-Optimized Cache](https://arxiv.org/abs/2501.19243)

### Token 级缓存
- [Token Caching for DiT Acceleration](https://arxiv.org/abs/2409.18523)
- [Streamlined Inference (NeurIPS 2024)](https://arxiv.org/abs/2411.01171)
- [Delta-DiT](https://openreview.net/forum?id=pDI03iK5Bf)

### 其他缓存方法
- [Foresight: Adaptive Layer Reuse](https://arxiv.org/abs/2506.00329)
- [BWCache](https://arxiv.org/abs/2509.13789)

### 综述与框架
- [A Survey on Cache Methods in Diffusion Models](https://arxiv.org/abs/2510.19755)
- [FastVideo V1 (含 TeaCache 集成)](https://haoailab.com/blogs/fastvideo/)
- [FastVideo on AMD ROCm](https://rocm.blogs.amd.com/artificial-intelligence/fastvideo-v1/README.html)
