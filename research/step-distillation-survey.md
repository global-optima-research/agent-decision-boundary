# Video DiT Step Distillation 调研

> 日期：2026-03-12
> 目的：评估 step distillation 方向的可行性和差异化空间

## 竞争格局总览

Wan 2.1 上已有 **3 篇蒸馏工作**，step distillation 方向远比预期拥挤。

| 论文 | 会议/日期 | 方法 | 基座模型 | 步数 | VBench | 训练规模 | 开源 |
|------|---------|------|---------|------|--------|---------|------|
| CausVid | CVPR'25 | DMD + 因果化 | Wan 2.1 1.3B | 4 | 84.27 | 64 GPU | ✅ |
| rCM | ICLR'26 | sCM + DMD 正则 | Wan 2.1 1.3B/14B | 4 | 84.43/85.05 | ~32-128 GPU | ✅ |
| TMD | arXiv 2026.01 | DMD2 扩展 | Wan 2.1 1.3B/14B | 2 | 84.68 | 64×H100 | ? |
| AnimateLCM | SIGGRAPH Asia'24 | LCM 解耦 | SD 1.5 + AnimateDiff | 4 | — | 8×A100 | ✅ |

## 论文详细分析

### 1. CausVid — From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

**arXiv**: 2412.07772 | **CVPR 2025**
**作者**: Tianwei Yin et al. (MIT + Adobe)
**代码**: [tianweiy/CausVid](https://github.com/tianweiy/CausVid)

#### 核心方法

CausVid 做两件事：
1. **因果化 (Causal Adaptation)**：将双向注意力 DiT 改为因果注意力（当前帧只看已生成帧），支持 KV-cache 流式生成
2. **DMD 蒸馏**：将 50 步模型蒸馏为 4 步

训练分 3 阶段：
- Stage 1: 双向 DMD 蒸馏（学习少步生成）
- Stage 2: 因果 ODE 预训练（学习因果注意力）
- Stage 3: 因果 DMD 蒸馏（两者结合）

#### 关键创新：非对称蒸馏

用**双向 teacher** 监督**因果 student**：
- Teacher 看完整序列（所有帧），生成高质量分布
- Student 只看已生成帧（因果 mask），学习逼近 teacher 分布
- 避免了因果生成中的误差累积

#### 训练成本
- **64 GPU**（8 nodes × 8 GPU）
- 数据集：MixKit 6K 视频（toy）；生产级需更大数据集
- 分辨率：480×832×81 帧
- 训练迭代：toy 数据集 ~1K 迭代即饱和

#### 结果
- 4 步生成，9.4 FPS（单 GPU，因果 + KV-cache）
- VBench-Long: 84.27（首个超越双向模型的因果方法）
- 支持流式生成、视频翻译、动态 prompt

#### 局限
- 因果化是核心贡献，蒸馏本身是 DMD 的直接扩展
- 64 GPU 训练成本高
- 因果注意力引入质量-延迟权衡

---

### 2. rCM — Score-Regularized Continuous-Time Consistency Model

**arXiv**: 2510.08431 | **ICLR 2026**
**作者**: Kaiwen Zheng et al. (NVIDIA + 清华)
**代码**: [NVlabs/rcm](https://github.com/NVlabs/rcm)

#### 核心方法

rCM = sCM（连续时间一致性模型）+ DMD 正则化：

```
L_rCM = L_sCM + λ · L_DMD    (λ = 0.01)
```

- **sCM loss（前向散度）**：在 teacher ODE 轨迹上训练，覆盖所有模式（diversity），但细节模糊
- **DMD loss（反向散度）**：在 student 生成样本上训练，追求模式质量，但可能 mode collapse
- **组合效果**：DMD 作为"长跳正则器"修复质量退化，sCM 保持多样性

#### 关键创新

1. **首个将 JVP-based 一致性模型扩展到 10B+ 视频模型**
2. **不需要 GAN 判别器**（vs DMD2/TMD），避免 GAN 调参
3. 开源 **FlashAttention-2 JVP Triton kernel**（重要基础设施贡献）

#### 关键技术细节
- 需要计算 **Jacobian-Vector Product (JVP)**，内存翻倍
- 需要 velocity parameterization（Wan 2.1 原生支持）
- TrigFlow 噪声 schedule（解析转换，不需重训 teacher）
- 训练框架：原生 PyTorch + FSDP2 + Context Parallelism

#### 训练成本
- Wan 2.1-1.3B 消融：batch=64, 10K iterations
- 数据集：Wan 2.1-14B 合成的 ~250K 视频
- 论文未明确 GPU 总数，估计 32-128 GPU
- **8×H800 可训 1.3B**（消融规模），14B 需要更多资源

#### 结果

| 模型 | 步数 | VBench Total | Quality | Semantic |
|------|------|-------------|---------|----------|
| Wan 2.1-14B + rCM | 4 | **85.05** | 85.57 | 82.95 |
| Wan 2.1-1.3B + rCM | 4 | 84.43 | 85.38 | 80.63 |

- 吞吐量：Wan 2.1-1.3B 4步 → 14.6 FPS（单 H100）
- 加速：15-50x（vs 50步 teacher + CFG）
- 1 步生成质量不佳（模糊），2 步已接近 teacher

#### 局限
- JVP 计算内存翻倍
- 少步模型在**物理一致性**上仍弱于 teacher
- 1 步视频生成不可用
- 需要 FP32 override 处理三角时间嵌入的数值不稳定

---

### 3. TMD — Transition Matching Distillation

**arXiv**: 2601.09881 | **2026.01**
**作者**: (基于 DMD2 扩展到视频)

#### 核心方法

TMD 是 DMD2 的视频扩展版本（DMD2-v），核心修改：

1. **Conv3D 判别器**：时空 GAN 判别器（替换 DMD2 的 2D 判别器）
2. **知识蒸馏预热**：仅用于 1 步模型的初始化
3. **时间步偏移**：`t = γt'/(γ-1)t'+1`，防止 mode collapse

此外提出 Transition Matching 作为统一框架，声称包含 DMD 和 consistency 作为特例。

#### 结果

| 模型 | 步数 (NFE) | VBench Total | Quality | Semantic |
|------|-----------|-------------|---------|----------|
| Wan 2.1-1.3B + TMD | 2 (2.33) | **84.68** | 85.71 | 80.55 |
| Wan 2.1-1.3B + TMD | 1 (1.17) | 83.80 | 85.07 | 78.69 |
| Wan 2.1-14B + TMD | 1 (1.38) | 84.24 | — | — |

- 2 步即超越 rCM 4 步，效率更高
- 但需要 64×H100 训练

#### 训练成本
- 数据集：500K text-video pairs
- GPU：64×H100（14B 模型，FSDP）
- 迭代：12K-44K depending on step count

---

### 4. DMD2 — Improved Distribution Matching Distillation

**arXiv**: 2405.14867 | **NeurIPS 2024 Oral**
**作者**: Tianwei Yin et al. (MIT + Adobe)
**代码**: [tianweiy/DMD2](https://github.com/tianweiy/DMD2)

#### 核心方法（图像，非视频）

DMD2 对 DMD1 的 3 个改进：

| 组件 | DMD1 | DMD2 |
|------|------|------|
| 回归 loss | 需要（昂贵预计算） | **去除** |
| Fake critic 更新 | 同频 | **5x 更快** (TTUR) |
| GAN loss | 无 | **添加** |

训练目标 = 分布匹配 loss（KL 散度 via score 差异）+ GAN loss（对扩散噪声样本做对抗）

#### 关键创新
- **去除回归 loss**：消除了百万级 teacher 采样的成本（DMD1 在 SDXL 上需 700 A100-days）
- **TTUR**：fake critic 更新 5x/每次 generator 更新，维持 score 估计精度
- **GAN loss**：student 可以**超越 teacher**（因为直接学习真实数据分布）

#### 结果（图像）
- ImageNet-64: 1 步 FID=**1.28**（超越 CTM 1.92, CM 6.20）
- SDXL: 4 步 FID=**19.32**（超越 SDXL teacher 19.36）
- 500x 推理加速

#### 与视频的关系
- DMD2 本身只做图像
- CausVid 是 DMD 的视频扩展（同一作者 Tianwei Yin）
- TMD 是 DMD2 的视频扩展

---

### 5. AnimateLCM — Decoupled Consistency Learning

**arXiv**: 2402.00769 | **SIGGRAPH Asia 2024**

#### 核心方法
基于 LCM（Latent Consistency Model），解耦为两阶段：
1. **图像一致性**：在 SD 1.5 上训练 LCM（学习空间质量）
2. **运动一致性**：冻结图像 LCM，仅训练 AnimateDiff 时间注意力

#### 适用性评估
- ❌ 基于 U-Net + AnimateDiff，**不适用于 DiT**
- ✅ "解耦图像/运动蒸馏"思路可借鉴
- Wan 2.1 的时空注意力是耦合的，无法像 AnimateDiff 那样物理分离

---

## 方法对比

### 技术路线分类

```
Step Distillation 方法
├── Consistency-based
│   ├── LCM / AnimateLCM (U-Net)
│   └── rCM (DiT, JVP-based, ICLR'26) ★
├── Distribution Matching
│   ├── DMD → CausVid (DMD + causal, CVPR'25) ★
│   └── DMD2 → TMD (DMD2 + Conv3D disc, 2026.01) ★
└── Progressive Distillation
    └── (较早期工作，已被上述方法超越)
```

### 核心差异

| 维度 | rCM | CausVid | TMD |
|------|-----|---------|-----|
| 是否需要 GAN | ❌ | ✅ (DMD) | ✅ (DMD2) |
| 是否需要 JVP | ✅ (内存 2x) | ❌ | ❌ |
| 最少步数 | 4 (2 可用) | 4 | **2 (1 可用)** |
| 额外架构改动 | 无 | 因果注意力 | 无 |
| 训练稳定性 | 较好（无 GAN） | 需调 GAN | 需调 GAN |
| 代码开源质量 | ✅ NVlabs | ✅ 完整 | ? |

---

## 差异化空间分析

### ❌ 直接做 step distillation → 没有差异化

3 篇强竞品（CausVid, rCM, TMD）已覆盖主流方法（DMD, consistency, DMD2），且都在 Wan 2.1 上验证。

### ⚠️ 可能的差异化方向

#### A. Distillation × Inference Acceleration 联合研究

**Gap**: 没人研究蒸馏模型上 FBC/sparse/quant 的表现和交互效应。

核心问题：
- 蒸馏后只剩 4 步，FBC 的步间缓存还有效吗？
- 4 步模型的注意力模式变化了吗？sparse 是否变得可用？
- quant × distillation 的交互是正交还是冲突？

**优势**：AutoAccel 全套代码直接复用；用开源 checkpoint 无需训练
**故事**："50步→4步给你 12x，推理加速再给 1.3x，但叠加效果未知——我们系统研究并提出联合方案"

#### B. 时间一致性专项改进

**Gap**: rCM 明确指出"少步模型在物理一致性上弱于 teacher"。所有方法都没有专门的 temporal consistency loss。

可利用 Phase 0 的注意力分析（头功能特化、跨帧注意力重要性）设计针对性 loss。

**风险**：需要训练（8×H800 可行但需 2-3 周），且改进幅度可能有限。

#### C. 低资源蒸馏 (8×H800)

**Gap**: 现有方法都需 32-128 GPU。能否用 8 GPU 达到同等质量？

**风险**：更像工程贡献，学术故事弱。

---

## 结论与建议

### Step Distillation 作为独立方向：❌ 已拥挤

之前判断"Wan 2.1 无蒸馏方案"是错误的。3 篇强竞品已覆盖。

### 推荐方向：Distillation × Inference Acceleration（方向 A）

理由：
1. **真正的空白**——无人系统研究蒸馏模型上的推理加速
2. **AutoAccel 100% 复用**——实验框架、交互分析方法、代码全部可用
3. **实验成本极低**——用 rCM/CausVid 开源 checkpoint，只需推理实验
4. **论文故事自然**——"我们发现蒸馏模型上的加速交互效应与原始模型显著不同，据此提出联合优化"
5. **资源充足**——RTX 5090 做推理实验，8×H800 做可能的轻量微调

### 第一步行动（已作废）

~~1. 下载 rCM Wan 2.1-1.3B 4步 checkpoint~~
~~2. 在 RTX 5090 上运行 FBC/sparse 全套实验~~
~~3. 对比 50 步原始模型 vs 4 步蒸馏模型的交互效应~~

---

## 二次核查：Distillation × Inference Acceleration 方向也已拥挤

> 核查日期：2026-03-12

上述"推荐方向 A"经文献核查后发现**同样不可行**。蒸馏模型上的推理加速已有大量工作覆盖：

### 直接竞品

| 论文 | 组合方式 | 加速 | 日期 |
|------|---------|------|------|
| TurboDiffusion | rCM 蒸馏 + SLA 稀疏 + SageAttn 量化 + W8A8 | **100-200x** | 2025.12 |
| DisCa | 蒸馏 + 可学习特征缓存（专门解决兼容性） | 11.8x | 2026.02 |
| Video-BLADE | 块稀疏注意力 + 步蒸馏联合训练 | 14.1x | 2025.08 |
| QuantSparse | 量化 + 稀疏注意力 | 1.88x | 2025.09 |
| Q-VDiT | 量化 + 蒸馏（恢复量化质量损失） | 3.9x | ICML'25 |
| S2Q-VDiT | 稀疏 token 蒸馏 + 量化 | 3.9x | NeurIPS'25 |
| DiffAgent | LLM 自动搜索最优加速组合 | — | 2026.01 |

### 关键发现

1. **TurboDiffusion** 已把蒸馏+稀疏+量化三种全部叠加，达到 100-200x 加速
2. **DisCa** 专门解决了"蒸馏与缓存不兼容"的问题，这正是我们想研究的
3. **DiffAgent** 用 LLM 自动搜索最优组合，与 AutoAccel 的自动搜索思路高度重叠
4. **Video-BLADE** 联合训练稀疏+蒸馏，比分别应用效果更好

### 总结论

**视频推理加速整个大方向——无论单技术（sparse/quant/cache）、步蒸馏、还是多技术组合——都已被充分覆盖。**

竞争格局（更新版）：
```
已拥挤 ❌:
├── 单技术加速: STA, SVG, SageAttn, FBC, PAB, ... (10+ 篇)
├── 步蒸馏: CausVid, rCM, TMD (3 篇，全在 Wan 2.1 上)
├── 多技术组合: TurboDiffusion (100-200x), QuantSparse, Video-BLADE
├── 蒸馏×缓存兼容: DisCa (专门解决)
└── 自动搜索: DiffAgent (LLM-based), AutoDiffusion (ICCV'23)
```

**需要彻底跳出视频推理加速赛道，寻找新方向。**

### 参考文献（新增）

- [TurboDiffusion](https://arxiv.org/abs/2512.16093) — rCM + SLA + SageAttn + W8A8, 100-200x (2025.12)
- [DisCa](https://arxiv.org/abs/2602.05449) — Distillation-compatible learnable caching, 11.8x (2026.02)
- [Video-BLADE](https://arxiv.org/abs/2508.10774) — Block-sparse + step distillation co-design, 14.1x (2025.08)
- [QuantSparse](https://arxiv.org/abs/2509.23681) — Quantization + sparsification (2025.09)
- [Q-VDiT](https://arxiv.org/abs/2505.22167) — Quantization + distillation, ICML 2025
- [S2Q-VDiT](https://arxiv.org/abs/2508.04016) — Sparse token distillation + quantization, NeurIPS 2025
- [DiffAgent](https://arxiv.org/abs/2601.03178) — LLM-driven acceleration code generation (2026.01)
- [NVIDIA FastGen](https://github.com/NVlabs/FastGen) — Unified distillation framework (2026.01)
