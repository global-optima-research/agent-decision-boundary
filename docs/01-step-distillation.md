# 步数蒸馏 (Step Distillation) 深入调研

> 调研日期：2026-03-06

## 目录

- [技术演进脉络](#技术演进脉络)
- [核心概念：CD vs CT vs 对抗蒸馏](#核心概念cd-vs-ct-vs-对抗蒸馏)
- [重点论文详解](#重点论文详解)
  - [TurboDiffusion](#1-turbodiffusion)
  - [DOLLAR](#2-dollar)
  - [DCM](#3-dcm-dual-expert-consistency-model)
  - [Video-BLADE](#4-video-blade)
  - [NVIDIA FastGen](#5-nvidia-fastgen)
- [方法横向对比](#方法横向对比)
- [当前 SOTA 总结](#当前-sota-总结)

---

## 技术演进脉络

```
2022  Progressive Distillation (Salimans & Ho)
       │  迭代减半采样步数，log2(N) 轮蒸馏
       ▼
2023  Consistency Models (Song et al., ICML 2023)
       │  自一致性属性，任意噪声→干净数据的直接映射
       │  分支：CD (需要 teacher) / CT (无需 teacher)
       ▼
2023  Latent Consistency Models (Luo et al.)
       │  在 SD 潜空间中应用一致性模型 + LCM-LoRA 即插即用
       ▼
2023  Adversarial Diffusion Distillation / SDXL-Turbo (Sauer et al.)
       │  GAN loss + Score Distillation → 1-4 步高质量生成
       ▼
2023  VideoLCM — 首次将一致性蒸馏应用于视频 (4步)
       ▼
2024  AnimateDiff-Lightning (ByteDance)
       │  渐进式对抗蒸馏 + 跨模型蒸馏 → 1-8 步视频
       ▼
2024  sCM (OpenAI, ICLR 2025)
       │  连续时间一致性模型简化/稳定/缩放至 1.5B
       ▼
2025  rCM (NVIDIA, ICLR 2026)
       │  前向-逆向散度联合蒸馏，支持 10B+ 模型
       ▼
2025  TurboDiffusion / DOLLAR / DCM / Video-BLADE / FastGen
       └  多技术融合时代：蒸馏 + 稀疏 + 量化 + 对抗 组合
```

### 关键里程碑论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Progressive Distillation](https://arxiv.org/abs/2202.00512) | 2022 | 开创性工作，迭代减半步数 |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023 | CD/CT 范式，单步生成理论基础 |
| [Latent Consistency Models](https://arxiv.org/abs/2310.04378) | 2023 | 潜空间一致性 + LCM-LoRA |
| [ADD / SDXL-Turbo](https://arxiv.org/abs/2311.17042) | 2023 | GAN + 分数蒸馏，1-4 步 |
| [VideoLCM](https://arxiv.org/abs/2312.09109) | 2023 | 首个视频一致性蒸馏 |
| [AnimateDiff-Lightning](https://arxiv.org/abs/2403.12706) | 2024 | 渐进对抗蒸馏视频 |
| [sCM](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/) | 2024 | 连续时间 CM 规模化 |
| [rCM](https://github.com/NVlabs/rcm) | 2025 | 10B+ 模型一致性蒸馏 SOTA |

---

## 核心概念：CD vs CT vs 对抗蒸馏

### 蒸馏方法分类体系

```
扩散模型蒸馏
├── 保真蒸馏 (Fidelity Distillation)
│   ├── Progressive Distillation
│   ├── Knowledge Distillation
│   └── Score Distillation (SDS, VSD)
│
├── 轨迹蒸馏 (Trajectory Distillation)
│   ├── Consistency Distillation (CD) ← 需要 teacher
│   ├── Consistency Training (CT) ← 不需要 teacher
│   ├── Latent Consistency Models
│   ├── Consistency Trajectory Models (CTM)
│   └── Rectified Flow / Flow Matching
│
└── 对抗蒸馏 (Adversarial Distillation)
    ├── ADD / SDXL-Turbo
    ├── SDXL-Lightning
    ├── LADD (Stability AI)
    └── Diffusion Adversarial Post-Training
```

### CD vs CT 核心区别

| 维度 | Consistency Distillation (CD) | Consistency Training (CT) |
|------|------|------|
| **是否需要 teacher** | 需要预训练扩散模型 | 仅需真实数据 |
| **训练信号** | teacher 的 ODE 轨迹点对 | 自身 EMA 的 bootstrapping |
| **质量上限** | 受限于 teacher 质量 | 理论上无上限 |
| **训练稳定性** | 更稳定，方差低 | 方差高，更难训练 |
| **实际应用** | 主流选择（已有好模型时） | 研究导向，从头训练 |
| **类比** | 从老师的解题步骤学捷径 | 自学发现路径 |

### 对抗蒸馏 vs 一致性方法

| 维度 | 一致性方法 | 对抗蒸馏 (ADD) |
|------|-----------|---------------|
| **核心 loss** | 自一致性：同轨迹点映射到同终点 | 双 loss：GAN loss + 分数蒸馏 |
| **1步质量** | 可能略模糊，2-4 步更佳 | 1步即锐利清晰 |
| **多步改进** | 自然支持迭代精化 | 4步后收益递减 |
| **模式覆盖** | 更好的多样性 | 有 GAN mode collapse 风险 |
| **训练复杂度** | 相对简单 | 需训练判别器，平衡双 loss |

**关键洞察**：2025-2026 年最成功的方法都是**混合方案**——结合一致性蒸馏的轨迹一致性 + 对抗 loss 的感知锐利度 + 分数/奖励微调的质量对齐。

---

## 重点论文详解

### 1. TurboDiffusion

> 生数科技 + 清华 TSAIL，2025.12 | [论文](https://arxiv.org/abs/2512.16093) | [代码](https://github.com/thu-ml/TurboDiffusion)

**核心思路**：三种正交加速技术的乘法叠加。

#### 技术架构

```
┌─────────────────────────────────────────────────────┐
│                 TurboDiffusion Pipeline              │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ Branch A:    │  │ Branch B:    │                 │
│  │ SLA 注意力   │  │ rCM 步数蒸馏 │                 │
│  │ 稀疏化微调   │  │ 100→33步     │                 │
│  └──────┬───────┘  └──────┬───────┘                 │
│         │    权重合并 (Merge)    │                    │
│         └──────────┬────────────┘                    │
│                    ▼                                  │
│         ┌──────────────────┐                         │
│         │ 推理时叠加:       │                         │
│         │ • SageAttention2++│                         │
│         │ • W8A8 INT8 量化  │                         │
│         │ • Triton 算子优化 │                         │
│         └──────────────────┘                         │
└─────────────────────────────────────────────────────┘
```

#### rCM (Rectified Consistency Model)

rCM 是 NVIDIA 提出的改进连续时间一致性模型（ICLR 2026）：

- 标准 sCM 存在误差累积和 mode-covering 倾向（前向散度目标）
- rCM 引入**分数蒸馏作为 long-skip 正则化器**，添加 mode-seeking 的逆向散度项
- 形成**前向-逆向散度联合蒸馏**框架
- 开源了 FlashAttention-2 兼容的 JVP kernel，支持 FSDP + Context Parallelism
- 可训练 **10B+ 参数**模型

#### 推理时各层加速策略

| 层类型 | 加速技术 | 机制 |
|--------|----------|------|
| Attention 层 | SageAttention2++ + SLA | 低比特量化注意力 + 90% 稀疏 Top-K |
| Linear 层 | W8A8 INT8 量化 | 128x128 block-wise，INT8 Tensor Core |
| Norm 层 | 自定义 Triton/CUDA kernel | LayerNorm/RMSNorm 优化 |
| 采样步数 | rCM 蒸馏 | 100 → 33 步（评测）/ 44 步（推荐） |

#### 性能数据 (单卡 RTX 5090)

| 模型 | 基线耗时 | TurboDiffusion | 加速比 |
|------|----------|----------------|--------|
| Wan2.1-T2V-1.3B-480P | 184s | **1.9s** | **~97x** |
| Wan2.1-T2V-14B-480P | 1,676s | **9.9s** | **~169x** |
| Wan2.1-T2V-14B-720P | 4,767s (79min) | **24s** | **~199x** |
| Wan2.2-I2V-A14B-720P | 4,549s (76min) | **38s** | **~120x** |

#### 局限性

- **缺乏定量质量评估**：无 VBench/FVD/LPIPS 分数，仅有视觉对比
- 目前仅在 Wan 模型家族上验证
- 仅支持长英文提示词

---

### 2. DOLLAR

> Princeton，ICCV 2025 | [论文](https://arxiv.org/abs/2412.15689) | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_DOLLAR_Few-Step_Video_Generation_via_Distillation_and_Latent_Reward_Optimization_ICCV_2025_paper.pdf)

**核心思路**：VSD + CD + Latent Reward Model 三合一蒸馏。

#### 技术架构

```
Loss = L_VSD + β_CD · L_CD + β_FT · L_FT(φ)
       ├── VSD: KL散度匹配student/teacher分布
       ├── CD:  多步teacher一致性蒸馏 (m=5步teacher)
       └── LRM: 潜空间奖励模型微调
```

**三组件各自作用**：
- **VSD 单独**：质量可比 teacher，但多样性不足
- **加入 CD**：大幅提升多样性 (Vendi score)
- **加入 LRM**：对齐人类偏好，进一步提升质量

#### Latent Reward Model (LRM) — 核心创新

传统奖励微调（如 DDPO）需要解码到像素空间计算奖励，显存开销巨大。LRM 直接在**潜空间**中近似像素空间奖励：

- 显存：5.926 GB → **8.998 MB**（~660x 降低）
- 不要求原始奖励函数可微
- 千级迭代即收敛
- 使用 HPSv2 (Human Preference Score v2) 作为奖励信号

#### 性能数据

**VBench 得分**（基于 CogVideoX 架构）：

| 模型 | VBench 总分 |
|------|-------------|
| **DOLLAR (HPSv2, 4步)** | **82.57** |
| Teacher (50步 DDIM) | 81.26 |
| Gen-3 | 80.30 |
| Kling | 79.57 |
| T2V-Turbo | 73.16 |

**加速比**：
- 4步 vs 50步 teacher：**15.6x**
- 1步 vs 50步 teacher：**278.6x**

**关键发现**：4步 student **超越** 50步 teacher（82.57 vs 81.26），在 16 个 VBench 子项中 9-10 项优于 teacher。

#### 局限性

- 存在 reward overoptimization 导致 "noise flow" 伪影
- 长提示词偏差
- 视频规格较小：128帧 × 192×320

---

### 3. DCM (Dual-Expert Consistency Model)

> 南京大学 / 港大 / 上海 AI Lab / NTU，ICCV 2025 | [项目页](https://vchitect.github.io/DCM/) | [代码](https://github.com/Vchitect/DCM)

**核心洞察**：去噪轨迹存在根本性冲突——前期（高噪声）需要大幅语义变化，后期（低噪声）需要精细细节调整。单一 student 同时学习两者会梯度冲突。

#### 双专家架构

```
去噪轨迹 t=50 ──────────── t_κ=37 ──────────── t=0
                │                     │
         ┌──────┴──────┐       ┌──────┴──────┐
         │ Semantic     │       │ Detail      │
         │ Expert (SemE)│       │ Expert (DetE)│
         │              │       │              │
         │ • 语义布局    │       │ • 精细纹理   │
         │ • 场景构图    │       │ • 外观质量   │
         │ • 运动模式    │       │ • LoRA 参数  │
         │              │       │ • GAN loss   │
         │ 时间一致性loss │       │ 特征匹配loss  │
         └─────────────┘       └─────────────┘
              2步                    2步
         ◄──── 4步推理总共 ────►
```

**训练极其轻量**：仅需 2000 轮迭代，24×A100 上完成。

| 阶段 | 专家 | 迭代次数 | 训练内容 |
|------|------|----------|----------|
| Stage 1 | Semantic Expert | 1,000 | 高噪声一致性 + 时间一致性 |
| Stage 2 | Detail Expert | 1,000 | LoRA + GAN loss + 特征匹配 |

#### 性能数据

**HunyuanVideo (13B, 129帧 1280×720)**：

| 方法 | 步数 | VBench | 延迟 (s) |
|------|------|--------|----------|
| Baseline (Euler) | 50 | 83.87 | ~1504 |
| LCM | 4 | 80.33 | — |
| PCM | 4 | 80.93 | — |
| **DCM** | **4** | **83.83** | **121.52** |

4 步几乎无损（83.83 vs 83.87），比 LCM 高 **+3.5 分**，延迟降低 **~12x**。

**CogVideoX (2B)**：DCM 4步 79.99 vs LCM 78.88 / PCM 79.09。

---

### 4. Video-BLADE

> ZipLab，2025.08 | [论文](https://arxiv.org/abs/2508.10774) | [项目页](https://ziplab.co/BLADE-Homepage/)

**核心思路**：稀疏注意力 + 步数蒸馏的**联合优化**（而非分开做两遍）。

#### Adaptive Block-Sparse Attention (ASA)

```
完整注意力矩阵          ASA 处理流程
┌────────────┐         1. 每个 head 采样 k=16 个代表 token
│████████████│         2. 计算低成本 max-pool 注意力分数
│████████████│   →     3. Top 阈值选择（保留 95% 注意力质量）
│████████████│         4. 生成稀疏二值 mask
│████████████│         5. 仅计算选中的 block
└────────────┘
                       → 实现 80% 稀疏度，无性能下降
```

- 在线动态 block masking：每层、每步、每输入自适应
- ASA-GT 扩展：添加 ln(n) 个全局 token 保持语义一致性

#### TDM (Trajectory Distribution Matching) 蒸馏

关键创新：将 ASA 稀疏性**内嵌到蒸馏训练中**，而非先蒸馏再剪枝：

- Teacher（密集，完整步数）生成参考轨迹
- Student 的注意力层替换为 ASA，同时学习稀疏和快速
- "fake score model" 近似 student 的采样分布，避免训练时昂贵的迭代采样

#### 性能数据

| 模型 | 端到端加速比 | VBench-2.0 |
|------|-------------|------------|
| Wan2.1-1.3B | **14.10x** | 0.570（基线 0.563，提升了） |
| CogVideoX-5B | **8.89x** | 0.569（基线 0.534，提升了） |

注意：BLADE 不仅不降质量，还**提升**了 VBench 分数。

---

### 5. NVIDIA FastGen

> NVIDIA，2025 | [博客](https://developer.nvidia.com/blog/accelerating-diffusion-models-with-an-open-plug-and-play-offering) | [代码](https://github.com/NVlabs/FastGen) | Apache 2.0

**定位**：统一的扩散模型蒸馏工具包，将所有主流蒸馏方法放在同一框架下。

#### 支持的蒸馏方法

```
FastGen 方法矩阵
├── 轨迹类 (Trajectory-based)
│   ├── ECT / TCM / sCT / sCD
│   ├── MeanFlow (MIT)
│   └── iCT (OpenAI)
│
├── 分布类 (Distribution-based)
│   ├── DMD2 (Adobe)
│   ├── f-Distill
│   └── LADD (Stability AI)
│
├── 因果蒸馏 (Causal) ← 视频重点
│   ├── CausVid (CVPR 2025) — 双向→自回归
│   └── Self-Forcing — 弥合训练-推理 gap
│
└── 微调类
    ├── SFT / CausalSFT
    └── KD / CausalKD
```

#### 支持的视频模型

- **Wan2.1-T2V-14B** (旗舰 demo)
- **Wan2.2**
- **NVIDIA Cosmos-Predict2.5**
- **CogVideoX**
- 支持 T2I / I2V / T2V / V2V / VACE 模式

#### 性能亮点

| 场景 | 加速比 | 备注 |
|------|--------|------|
| Wan2.1-14B (50步→2步) | **50x** | 质量可比 |
| Self-Forcing 实时视频 | **~150x** | 17.0 FPS，亚秒延迟 |
| CorrDiff 天气模型 | **23x** | 预测精度维持 |
| 14B 模型蒸馏训练 | — | 64×H100 上 **16小时** |

#### 使用方式

```bash
# 安装
git clone https://github.com/NVlabs/FastGen.git && cd FastGen
pip install -e .

# 单卡训练
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_test.py

# 多卡 FSDP2
torchrun --nproc_per_node=8 train.py \
    --config=... - trainer.fsdp=True
```

---

## 方法横向对比

| 维度 | TurboDiffusion | DOLLAR | DCM | Video-BLADE | FastGen |
|------|---------------|--------|-----|-------------|---------|
| **最大加速比** | 199x | 278.6x (1步) / 15.6x (4步) | ~12x | 14.1x | 150x |
| **蒸馏步数** | 33-44 步 | 1-4 步 | 4-8 步 | 4-8 步 | 2+ 步 |
| **是否需要重训练** | 需要 | 需要 | 需要（但仅 2000 轮） | 需要 | 需要 |
| **质量评估** | 仅视觉对比 ⚠️ | VBench 82.57 ✅ | VBench 83.83 ✅ | VBench-2.0 ✅ | VBench ✅ |
| **验证模型** | Wan 系列 | CogVideoX | HunyuanVideo / CogVideoX / Wan | Wan / CogVideoX | Wan / Cosmos / CogVideoX |
| **额外加速** | 稀疏注意力 + 量化 | 无 | 可叠加 SVG | 稀疏注意力内嵌 | 方法可组合 |
| **开源** | ✅ | ✅ | ✅ | ✅ | ✅ Apache 2.0 |
| **发表** | 预印本 | ICCV 2025 | ICCV 2025 | 预印本 | NVIDIA 博客 |

---

## 当前 SOTA 总结

### 2025-2026 趋势

1. **混合方法统治**：最佳方案都是多种蒸馏技术的组合（一致性 + 对抗 + 奖励/分数蒸馏）
2. **正交加速可叠加**：步数蒸馏 × 稀疏注意力 × 量化，三者乘法关系
3. **训练成本持续下降**：DCM 仅需 2000 轮，FastGen 14B 模型仅需 16 小时
4. **因果蒸馏兴起**：CausVid / Self-Forcing 将双向模型转为自回归，实现实时交互

### 推荐选型

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| **快速验证 / PoC** | DCM | 训练轻量（2000轮），质量几乎无损 |
| **极致加速** | TurboDiffusion | 三重正交叠加，100x+ |
| **质量优先** | DOLLAR | 4步超越 teacher，有严格评估 |
| **统一工具链** | FastGen | 多方法对比，生产级工程 |
| **实时交互** | FastGen (Self-Forcing) | 17 FPS，亚秒延迟 |

---

## 参考资料

### 基础理论
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
- [Consistency Models (Song et al., ICML 2023)](https://arxiv.org/abs/2303.01469)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)
- [LCM-LoRA: Universal Acceleration Module](https://arxiv.org/abs/2311.05556)
- [Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042)
- [SDXL-Lightning: Progressive Adversarial Distillation](https://arxiv.org/abs/2402.13929)
- [A Survey on Pre-Trained Diffusion Model Distillations](https://arxiv.org/abs/2502.08364)

### 视频蒸馏
- [VideoLCM: Video Latent Consistency Model](https://arxiv.org/abs/2312.09109)
- [AnimateDiff-Lightning (ByteDance)](https://arxiv.org/abs/2403.12706)
- [rCM: Rectified Consistency Model (NVIDIA, ICLR 2026)](https://github.com/NVlabs/rcm)
- [TurboDiffusion](https://arxiv.org/abs/2512.16093) | [GitHub](https://github.com/thu-ml/TurboDiffusion)
- [DOLLAR (ICCV 2025)](https://arxiv.org/abs/2412.15689)
- [DCM (ICCV 2025)](https://vchitect.github.io/DCM/) | [GitHub](https://github.com/Vchitect/DCM)
- [Video-BLADE](https://arxiv.org/abs/2508.10774) | [项目页](https://ziplab.co/BLADE-Homepage/)
- [NVIDIA FastGen](https://github.com/NVlabs/FastGen)
- [OSV: One Step Video (CVPR 2025)](https://arxiv.org/abs/2409.11367)
- [GPD: Guided Progressive Distillation](https://arxiv.org/html/2602.01814)

### 连续时间一致性
- [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)
- [sCM: Simplifying, Stabilizing, and Scaling (OpenAI, ICLR 2025)](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)
- [Consistency Models Made Easy (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/bb166dd4de5dba363bf1023eb956a826-Paper-Conference.pdf)

### 分析与综述
- [The Paradox of Diffusion Distillation (Sander Dieleman)](https://sander.ai/2024/02/28/paradox.html)
- [CausVid (CVPR 2025)](https://github.com/tianweiy/CausVid)
- [Self-Forcing](https://arxiv.org/html/2506.08009v1)
