# 研究方向战略分析

> 日期：2026-03-09
> 背景：Phase 0/1 完成后，重新评估最优研究方向
> 资源：8×H800 (训练) + 1×RTX 5090 (实验)
> 目标：顶会论文（NeurIPS 2026 / ICML 2027）

## 经文献验证的真实竞争格局

### 核心结论

**视频推理加速整个领域竞争极其激烈，没有真正的蓝海。** 每个子方向都有近期顶会论文。
策略不应是"找蓝海"，而是"在半占领方向上快速做出强结果"或"组合创新"。

### ❌ 已经拥挤的方向（不建议进入）

| 方向 | 直接竞品 | 状态 |
|------|---------|------|
| 加速感知蒸馏（让模型适应稀疏） | SpargeAttention2 (2026.02), SLA2 (2026.02), VSA (NeurIPS'25), Video-BLADE (2025.08) | 4+ 篇直接竞品 |
| 纯注意力 kernel | STA, FPSAttention, VMonarch, SVG, PISA, VideoNSA (ICLR'26) | 10+ 篇 |
| 测试时计算×加速 | Video-T1 (ICCV'25), Inference-Time Scaling (CVPR'25) | 已有顶会 |
| 潜空间压缩 | DLFR-VAE (MM'25), DLFR-Gen (ICCV'25), Improved Video VAE (CVPR'25) | 3 篇顶会 |
| 自适应 per-layer/step 计算 | DiffCR (CVPR'25), DyDiT (ICLR'25), EasyCache, Foresight | 4+ 篇 |
| 传统步数蒸馏 | rCM (ICLR'26), DCM, DOLLAR, TurboDiffusion | 大厂主导 |

#### 关键竞品详情

**SpargeAttention2** (arXiv 2602.13515, 清华)
- Hybrid Top-k+Top-p masking + **distillation fine-tuning objective**
- 95% sparsity, 16.2x attention speedup
- 与之前推荐的"加速感知蒸馏"几乎完全重叠

**VSA** (NeurIPS 2025, Hao AI Lab)
- 可训练稀疏注意力，训练+推理都用稀疏
- Wan 1.3B attention 6x 加速, 14B 端到端 1274s→576s
- 2.53x training FLOPs reduction

**Video-BLADE** (arXiv 2508.10774)
- Adaptive Block-Sparse Attention + 步蒸馏协同设计
- Wan 1.3B 14.1x 端到端加速
- 数据无关蒸馏

**QuantCache** (ICCV 2025)
- 联合优化 hierarchical caching + importance-guided quantization + structural pruning
- Open-Sora 6.72x 端到端加速
- 训练无关

### ⚠️ 半占领方向（有差异化空间，但需谨慎）

#### A. 多技术自动组合优化

**已有工作**：
- QuantSparse — quant + sparse（2 种技术，有干扰分析）
- QuantCache (ICCV'25) — quant + cache + pruning（3 种联合）
- TurboDiffusion — SLA + SageAttention + rCM + W8A8（4 种手工组合，100-200x）

**没人做的**：
给定 N 种可用技术 × 质量约束 × 硬件配置，**自动搜索** per-(layer, head, step) 的最优配置。
现有工作都是手工设计或固定组合，没有最优性保证。

**风险**：可能被审稿人认为是工程/系统贡献而非研究贡献。

#### B. 视频蒸馏的时间一致性

**已有工作**：
- CausVid (CVPR'25) — DMD 视频蒸馏 + asymmetric distillation
- rCM (ICLR'26) — NVIDIA 少步视频生成
- TMD — temporal maintenance distillation（用于量化模型）

**没人做的**：
专门设计 temporal consistency loss 作为视频蒸馏的**核心贡献**。
现有工作的时间一致性是副产品，不是主要优化目标。

**风险**：蒸馏赛道整体拥挤，差异化点需足够大。

#### C. 跨模型加速策略迁移

**没人做的**：
在 Wan 1.3B 上 profile 加速配置 → 迁移到 Wan 14B / HunyuanVideo / CogVideoX。
Phase 0 验证了跨 prompt 模式不变性 (cosine 0.956)，但跨模型不变性未被研究。

**风险**：跨模型迁移效果差则整个 story 崩掉。

#### D. 视频 DiT 功能感知剪枝

**已有工作**：
- EcoDiff — 模型无关结构化剪枝（主要针对图像）
- SnapGen — 移动端 DiT 剪枝
- DyDiT (ICLR'25) — 动态宽度

**没人做的**：
利用 Phase 0 的头功能特化发现（78 spatial + 8 global + 2 temporal + 272 mixed），
做功能感知剪枝 — 根据头的实际功能决定保留/剪枝/降精度。

**风险**：剪枝方向整体关注度低，影响力上限可能有限。

### 竞争程度排序

```
从高到低：
注意力kernel >>> 可训练稀疏 > 步数蒸馏 > 自适应计算 > 潜空间压缩 >
测试时计算 > 多技术组合 > 时间一致性蒸馏 > 跨模型迁移 > 视频剪枝
```

### ❌ 分析型论文方向（已验证不可行）

**设想**：用 Phase 0/1 数据写 "Dissecting Attention in Video DiT" 分析型论文。

**直接竞品**：

专门的分析论文：
| 论文 | 日期 | 模型 | 分析内容 |
|------|------|------|---------|
| "Analysis of Attention in Video Diffusion Transformers" (arXiv:2504.10317) | 2025.04 | Mochi, HunyuanVideo, **Wan 2.1**, CogVideoX | 跨 prompt 不变性、层敏感度、逐步演变、sink heads |
| "Understanding Attention Mechanism in Video Diffusion Models" (arXiv:2504.12027) | 2025.04 | — | 扰动分析 + 信息熵，block 级 |

加速论文中内含的分析（部分列表）：
| 论文 | 头分类 | 步间分析 | 跨 prompt | 层敏感度 |
|------|--------|---------|----------|---------|
| Compact Attention (2508.12969) | **5 类**（3 spatial × 2 temporal） | ✅ | ✅ >0.8 | ❌ |
| Sparse-vDiT | 4 模式 + skip = 5 类 | ❌ | ✅ 50 prompts | ❌ |
| SVG (ICML'25) | 2 类（spatial/temporal） | 动态切换 | — | ❌ |
| AdaSpa (ICCV'25) | per-head 自适应 | ✅ 声称步间不变 | — | ❌ |
| PAB (ICLR'25) | — | ✅ U 形曲线 | — | ❌ |
| DiTFastAttn (NeurIPS'24) | — | ✅ 步间相似度 | — | ✅ |
| Radial Attention (NeurIPS'25) | — | — | — | — (能量衰减分析) |

**我们 Phase 0/1 的 5 个维度 vs 已有覆盖**：
| 我们的发现 | 已有覆盖？ |
|-----------|-----------|
| 头功能特化分类 | ✅ Compact Attention 5 类, Sparse-vDiT 5 模式 |
| 层敏感度排名 | ✅ arXiv:2504.10317 逐层 masking |
| 步间变化率 | ✅ PAB, DiTFastAttn, arXiv:2504.10317 |
| 跨 prompt 不变性 | ✅ Sparse-vDiT, Compact Attention, Efficient-vDiT |
| 稀疏 mask 跨步稳定性 | ⚠️ 部分覆盖（AdaSpa, Compact Attention, LiteAttention） |

**结论**：arXiv:2504.10317 已经在 Wan 2.1 上做了几乎相同的分析。Phase 0/1 数据不再构成"独特壁垒"。

### ❌ Sparse-vDiT 改进方向（已验证不可行）

**分析了 Sparse-vDiT 的 5 个潜在改进点**：
1. 固定几何模式 → 数据驱动 mask：**已被 SVG (ICML'25)、AdaSpa (ICCV'25) 做了**
2. 无 per-step 差异化 → per-step 策略：**已被 SVG 2 (NeurIPS'25 Spotlight) 做了**
3. Wan 2.1 表现最差 → 针对性优化：差异化不足
4. 缺少 STA baseline → 补充对比：不构成论文贡献
5. 无多技术组合 → cross-step mask 复用：**已被 LiteAttention 做了**

**结论**：三个看似可行的改进（per-step 策略 + data-driven mask + cross-step 复用）组合起来本质上就是 SVG/AdaSpa 已做的事情。不足以发顶会。

### 推荐策略

#### 策略 1：速度取胜（最务实）

不追求蓝海，在半占领方向上**快速做出强实验结果**。
选方向 B（时间一致性蒸馏）或方向 A（多技术组合），8×H800 快速迭代。
Phase 0/1 数据可复用。

#### 策略 2：组合创新（最有新颖性）

合并方向 A + C：**自动组合优化 + 跨模型泛化**。
Story："不同模型/场景需要不同加速组合，我们提出框架：profile 一次 → 自动搜索最优组合 → 跨模型迁移"。

#### 策略 3：换赛道（最大胆）

离开推理加速，用 Phase 0/1 的注意力分析数据做**视频生成质量提升**。
例如基于头功能特化和层敏感度分析，改进注意力机制提升时间一致性/细节质量。
类似 Enhance-A-Video 的方向，但有更细粒度的分析支撑。

---

## 参考文献

### 可训练稀疏注意力
- [SpargeAttention2](https://arxiv.org/abs/2602.13515) — Hybrid Top-k+Top-p + distillation fine-tuning (2026.02)
- [SLA2](https://arxiv.org/abs/2602.12675) — Learnable routing + QAT (2026.02)
- [VSA](https://arxiv.org/abs/2505.13389) — Trainable sparse attention, NeurIPS 2025
- [Video-BLADE](https://arxiv.org/abs/2508.10774) — Block-sparse + step distillation co-design
- [Sparse-to-Sparse Training](https://arxiv.org/abs/2504.21380) — TMLR 2025

### 多技术联合
- [QuantSparse](https://arxiv.org/abs/2509.23681) — Quantization + sparsification
- [QuantCache](https://arxiv.org/abs/2503.06545) — Quantization + caching + pruning, ICCV 2025
- [TurboDiffusion](https://arxiv.org/abs/2512.16093) — SLA + SageAttention + rCM + W8A8, 100-200x

### 步数蒸馏
- [rCM](https://github.com/NVlabs/rcm) — ICLR 2026, NVIDIA
- [CausVid](https://github.com/tianweiy/CausVid) — CVPR 2025, DMD for video
- [DisCa](https://arxiv.org/abs/2602.05449) — Distillation-compatible caching

### 注意力 kernel
- [STA](https://arxiv.org/abs/2502.04507) — ICML 2025
- [FPSAttention](https://arxiv.org/abs/2506.04648) — NeurIPS 2025 Spotlight
- [VMonarch](https://arxiv.org/abs/2601.22275) — Stanford/Tri Dao
- [SVG](https://arxiv.org/abs/2502.01776) — ICML 2025
- [PISA](https://arxiv.org/abs/2602.01077) — Piecewise sparse attention
- [VideoNSA](https://arxiv.org/abs/2510.02295) — ICLR 2026

### 注意力分析（分析型论文的竞品）
- [Analysis of Attention in Video Diffusion Transformers](https://arxiv.org/abs/2504.10317) — 跨模型分析 (Mochi/HunyuanVideo/Wan2.1/CogVideoX), 2025.04
- [Understanding Attention Mechanism in Video Diffusion Models](https://arxiv.org/abs/2504.12027) — 扰动+信息熵分析, 2025.04
- [Compact Attention](https://arxiv.org/abs/2508.12969) — 5 类头分类 (3 spatial × 2 temporal), 2025.08
- [Efficient-vDiT](https://arxiv.org/abs/2502.06155) — 对角块分析, 跨 prompt >90% overlap
- [PAB](https://arxiv.org/abs/2408.12588) — U 形步间变化曲线, ICLR 2025
- [DiTFastAttn](https://arxiv.org/abs/2406.08552) — 三类冗余 (spatial/temporal/conditional), NeurIPS 2024
- [Radial Attention](https://arxiv.org/abs/2506.19852) — 时空能量衰减, NeurIPS 2025

### 其他
- [Video-T1](https://arxiv.org/abs/2503.18942) — Test-time scaling, ICCV 2025
- [DiffCR](https://arxiv.org/abs/2412.16822) — Layer/timestep-adaptive compression, CVPR 2025
- [DyDiT](https://proceedings.iclr.cc/paper_files/paper/2025/file/a44a70acd5d0abc1a252ada9719dd06d-Paper-Conference.pdf) — ICLR 2025
- [DLFR-VAE](https://arxiv.org/abs/2502.11897) — Dynamic latent frame rate, MM 2025
- [DLFR-Gen](https://openaccess.thecvf.com/content/ICCV2025/papers/Yuan_DLFR-Gen_Diffusion-based_Video_Generation_with_Dynamic_Latent_Frame_Rate_ICCV_2025_paper.pdf) — ICCV 2025
