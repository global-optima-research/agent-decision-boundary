# 视频生成领域全景分析

> 日期：2026-03-09
> 目的：跳出推理加速，评估整个视频生成领域的研究机会
> 资源：8×H800 (训练) + 1×RTX 5090 (实验)
> 目标：顶会论文（NeurIPS 2026 / ICML 2027）

## 各子领域竞争热度

```
极热 🔴🔴🔴  推理加速 | 核心模型训练(大厂) | 注意力优化
很热 🔴🔴    可控生成(motion/camera) | 身份保持 | 3D/4D生成 | 统一理解+生成
热   🔴      视频编辑 | 世界模型/物理 | 步数蒸馏
温热 🟡      音视频联合生成 | 交互式视频/游戏引擎
热   🔴      视频RLHF/对齐（三次验证后上调）
偏冷 🟢      视频数据工程 | 视频生成评估方法
```

## 各方向详细状态

| 方向 | 代表工作 | 竞争 | 8×H800够？ | 我们的优势？ |
|------|---------|------|-----------|------------|
| 核心模型训练 | Sora, Seedance, Veo, Wan | 大厂独占 | ❌ | 无 |
| 推理加速 | （见 direction-analysis.md） | 极饱和 | ✅ | Phase 0/1 数据 |
| 可控生成(motion) | MotionAgent (ICCV'25), Wan-Move (NeurIPS'25), Motion Prompting (CVPR'25) | 很热 | ✅ | 无 |
| 可控生成(camera) | PostCam, CamCo, I2VControl | 热 | ✅ | 无 |
| 身份保持 | ConsisID (CVPR'25), MagicMirror (ICCV'25) | 很热 | ✅ | 无 |
| 3D/4D 生成 | Lyra (ICLR'26), CAT4D (CVPR'25), Phys4D | 很热 | ⚠️ | 无 |
| 统一理解+生成 | UniVideo (ICLR'26), Thinking with Video | 很热 | ❌ 需更多 | 无 |
| 视频编辑 | Video-3DGS, TokenFlow | 热 | ✅ | 无 |
| 世界模型/物理 | Cosmos (NVIDIA), Vid2World | 热(大厂多) | ⚠️ | 无 |
| 步数蒸馏 | rCM (ICLR'26), CausVid (CVPR'25), DCM | 热 | ✅ | 无 |
| 音视频联合 | MOVA, LTX-2, Seedance 2.0 | 温热 | ✅ | 无 |
| 交互式视频 | Yan, GameGen-X | 温热 | ⚠️ | 无 |
| **视频 RLHF/对齐** | **VideoDPO (CVPR'25), Flow-GRPO (NeurIPS'25) 等 10+ 篇** | **热** | **✅** | **低** |
| 视频数据工程 | — | 偏冷 | ✅ | 无 |
| 视频生成评估 | VBench-2.0, Video-Bench | 偏冷 | ✅ | 有 |

## 推荐方向排序

### ~~★★★~~ → ★★ 视频生成对齐（Video Generation Alignment）

> **热度三次修正**：
> - 初始评估："偏冷" → 二次验证修正为"温热" → **三次验证修正为"热"**
> - 每次深挖都发现比预期更多的竞品，这个方向已不再是早期阶段。

**已有工作（远比初始评估更多）**：

闭源模型：
| 模型 | 对齐情况 |
|------|---------|
| HunyuanVideo 1.5 | ✅ CT → SFT → DPO + RLHF（MixGRPO），4 维 VLM reward model |
| Kling 2.6/3.0 | ✅ RLHF，多 reward model |
| Sora 2 | ✅ 大概率做了（未公开细节） |
| Seedance 2.0 | ✅ feedback-driven learning |
| CogVideoX | ❌ 基础模型没有 |

通用对齐方法（学术）：
| 工作 | 发表 | 内容 |
|------|------|------|
| VideoDPO | CVPR 2025 | OmniScore 偏好对齐，3 个开源模型验证 |
| VideoReward | 2025.01 | 182k 偏好数据 + Flow-DPO/Flow-RWR |
| OnlineVPO | 2024.12 | 在线视频偏好优化 |
| Dual-IPO | 2025.02 | 双迭代偏好优化（已在 WanX-1.3B 上验证） |
| VPO | ICCV 2025 | 通过 prompt 优化做对齐（不改模型） |
| Discriminator-Free DPO | 2025.04 | 无判别器的视频 DPO |

**⚠️ 专门针对 Wan 2.1 的对齐工作（三次验证新增）**：
| 工作 | 方法 | 基座模型 | 发表 |
|------|------|---------|------|
| **Flow-GRPO** | 在线 RL + flow matching | Wan 2.1-14B | **NeurIPS 2025** |
| **GigaVideo-1** | Reward-guided fine-tuning | Wan 2.1 | 2025.06 |
| **GenRL** | GRPO + 多 reward | Wan 2.1-1.3B | GitHub 开源框架 |
| **Dual-IPO** | 双迭代偏好优化 | WanX-1.3B | 2025.02 |
| **Identity-GRPO** | 身份保持 RL | Wan 系列 | 阿里自己的 |

> **关键结论：Wan 2.1 官方没做对齐，但第三方已有 5 个工作在 Wan 2.1 上做了 RL/RLHF 对齐。**
> 其中 Flow-GRPO 已被 NeurIPS 2025 接收，GenRL 提供了开源框架。
> **"在 Wan 2.1 上做对齐"不再是差异化切入点。**

**仍然存在的未解决问题**：
1. Reward model 质量不够 — 现有 reward 大多基于图像指标拼凑，没有真正理解视频时间维度
2. 物理合理性 — 没有好的自动评估方法
3. 长视频对齐 — 现有工作都是短片段

**风险**：方向已经从"温热"升级为"热"，竞争强度被严重低估。

### ★★ 推理加速半占领方向（详见 direction-analysis.md）

从推理加速内部选差异化方向：
- 多技术自动组合优化
- 视频蒸馏时间一致性
- 跨模型加速策略迁移

**优点**：留在熟悉领域，Phase 0/1 数据直接复用。
**缺点**：整个加速领域竞争激烈，即使半占领方向也有被抢先的风险。

### ~~★★~~ → ★ 视频生成质量诊断 / 分析型论文

> **四次验证后下调**：已有 2 篇专门的分析论文 + 10+ 篇加速论文内含分析。
> arXiv:2504.10317 已在 Mochi/HunyuanVideo/Wan 2.1/CogVideoX 上做了跨模型注意力分析。
> Compact Attention 做了 5 类头分类。Phase 0/1 数据不再构成"独特壁垒"。

**直接竞品**：
- arXiv:2504.10317 "Analysis of Attention in Video Diffusion Transformers" — 跨 prompt、层敏感度、逐步演变、sink heads，**覆盖了 Wan 2.1**
- arXiv:2504.12027 "Understanding Attention Mechanism in Video Diffusion Models" — 扰动分析 + 信息熵

**风险**：我们的分析与 arXiv:2504.10317 高度重叠，差异化空间极小。

### ★★ 交互式视频生成

**现状**：2025 ICML 才有 position paper，Yan 框架做到 1080P/60FPS，但整体还在非常早期。

**优点**：方向新，应用前景大（游戏、模拟、embodied AI）。
**缺点**：需要 real-time streaming 工程能力，与现有专长关联不大，可能需要更大算力。

## 综合评估

| 方向 | 新颖性 | 资源匹配 | 我们的优势 | 出成果速度 | 综合 |
|------|--------|---------|-----------|-----------|------|
| 视频 RLHF/对齐 | 中 | ✅ | 低 | 中 | ★★ (下调) |
| 推理加速(半占领) | 中 | ✅ | 中 | 快 | ★★ |
| 生成质量诊断/分析 | 低 | ✅ | 低 | 快 | ★ (下调，已有直接竞品) |
| 交互式视频 | 高 | ⚠️ | 低 | 慢 | ★★ |
| 音视频联合加速 | 高 | ✅ | 低 | 慢 | ★ |

> **注意**：经过多轮验证，没有方向获得 ★★★ 评级。所有"看起来空白"的方向经深挖后均发现有竞品。

## 决策建议

**核心教训：这个领域没有蓝海。** 每个方向都有竞品，策略应转为：
1. 在已有方向上做出**更强的实验结果**（速度取胜）
2. 或找到**更细粒度的未解决问题**（而非"整个方向没人做"）

**如果想留在熟悉领域**：选推理加速半占领方向 — Phase 0/1 数据直接用，但需要快速执行。

**如果想最大化已有数据价值**：选生成质量诊断 — 用现有注意力分析做可解释性研究，转换赛道。

**如果愿意学习新技术栈**：视频 RLHF/对齐仍有空间，但竞争比预期激烈得多，需要找到非常具体的差异化点。

## 参考文献

### 视频 RLHF/对齐（通用方法）
- [VideoDPO](https://arxiv.org/abs/2412.14167) — OmniScore 偏好对齐, CVPR 2025
- [VideoReward](https://arxiv.org/abs/2501.13918) — 182k 偏好数据 + Flow-DPO/Flow-RWR (2025.01)
- [OnlineVPO](https://arxiv.org/abs/2412.15159) — 在线视频偏好优化 (2024.12)
- [Dual-IPO](https://arxiv.org/abs/2502.02088) — 双迭代偏好优化 (2025.02)
- [VPO](https://arxiv.org/html/2503.20491v1) — Prompt 优化对齐, ICCV 2025
- [Discriminator-Free DPO for Video](https://arxiv.org/abs/2504.08542) — 无判别器 (2025.04)
- [HunyuanVideo 1.5 Technical Report](https://arxiv.org/html/2511.18870v2) — 完整 DPO+RLHF pipeline
- [Unified Reward Model](https://arxiv.org/abs/2503.05236) — 统一 reward

### 视频 RLHF/对齐（Wan 2.1 专项，三次验证新增）
- [Flow-GRPO](https://arxiv.org/abs/2505.05470) — 在线 RL + flow matching, Wan 2.1-14B, **NeurIPS 2025**
- [GigaVideo-1](https://arxiv.org/abs/2506.10639) — Reward-guided fine-tuning, Wan 2.1, 4 GPU-hours (2025.06)
- [GenRL](https://github.com/ModelTC/GenRL) — 模块化 RL 框架, Wan 2.1-1.3B, 开源
- [Identity-GRPO](https://arxiv.org/abs/2510.14256) — 身份保持 RL, Wan 系列, 阿里

### 可控生成
- [MotionAgent](https://arxiv.org/abs/2502.03207) — ICCV 2025
- [Wan-Move](https://github.com/ali-vilab/Wan-Move) — NeurIPS 2025
- [Motion Prompting](https://motion-prompting.github.io/) — CVPR 2025
- [Controllable Video Generation: A Survey](https://arxiv.org/html/2507.16869v2)

### 世界模型/物理
- [Cosmos](https://research.nvidia.com/publication/2025-09_world-simulation-video-foundation-models-physical-ai) — NVIDIA
- [Vid2World](https://arxiv.org/html/2505.14357v2)
- [Phys4D](https://arxiv.org/html/2603.03485)
- [How Far is Video Generation from World Model](https://phyworld.github.io/)

### 3D/4D 生成
- [Lyra](https://github.com/nv-tlabs/lyra) — ICLR 2026, NVIDIA
- [CAT4D](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_CAT4D_Create_Anything_in_4D_with_Multi-View_Video_Diffusion_Models_CVPR_2025_paper.pdf) — CVPR 2025
- [Diff4Splat](https://openreview.net/forum?id=WRmU41PpEK)

### 统一理解+生成
- [UniVideo](https://arxiv.org/abs/2510.08377) — ICLR 2026, Kling
- [Thinking with Video](https://arxiv.org/abs/2511.04570)
- [Unified Multimodal Models: Survey](https://arxiv.org/abs/2505.02567)

### 音视频联合
- [MOVA](https://arxiv.org/abs/2602.08794) — 2026.02
- [MM-Sonate](https://arxiv.org/abs/2601.01568) — 2026.01
- [LTX-2](https://introl.com/blog/ltx-2-audiovisual-diffusion-synchronized-video-audio-2026)

### 交互式视频
- [Position: Interactive Generative Video as Next-Gen Game Engine](https://arxiv.org/abs/2503.17359) — ICML 2025
- [Yan](https://arxiv.org/html/2508.08601v1) — 1080P/60FPS
- [GameGen-X](https://openreview.net/forum?id=8VG8tpPZhe)

### 身份保持
- [ConsisID](https://openaccess.thecvf.com/content/CVPR2025/papers/Yuan_Identity-Preserving_Text-to-Video_Generation_by_Frequency_Decomposition_CVPR_2025_paper.pdf) — CVPR 2025
- [MagicMirror](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_MagicMirror_ID-Preserved_Video_Generation_in_Video_Diffusion_Transformers_ICCV_2025_paper.pdf) — ICCV 2025

### 综合 Survey
- [Video diffusion generation: comprehensive review and open problems](https://link.springer.com/article/10.1007/s10462-025-11331-6)
- [Efficient Diffusion Models: A Survey (TMLR 2025)](https://github.com/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey)
- [Video Is Worth a Thousand Images: Long Video Generation](https://dl.acm.org/doi/10.1145/3771724)
