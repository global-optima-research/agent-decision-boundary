# FastGen vs FastVideo 对比分析

> 调研日期：2026-03-06

## 概述

| 维度 | NVIDIA FastGen | Hao AI Lab FastVideo |
|------|---------------|---------------------|
| **定位** | 蒸馏方法合集（研究导向） | 端到端加速框架（工程导向） |
| **仓库** | [NVlabs/FastGen](https://github.com/NVlabs/FastGen) | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) |
| **License** | Apache 2.0 | Apache 2.0 |
| **核心能力** | 8+ 种蒸馏算法统一 API | 推理优化 + 稀疏注意力 + 蒸馏 |
| **覆盖范围** | 图像 + 视频 + 天气模型 | 聚焦视频生成 |

**结论：互补而非竞争。工程落地选 FastVideo，蒸馏研究选 FastGen，极致性能两者组合。**

---

## 项目健康度对比

| 指标 | FastGen | FastVideo | 说明 |
|------|---------|-----------|------|
| GitHub Stars | ~624 | **~3,100** | FastVideo 5x |
| 上线时间 | 2026.01（2个月） | 2024.10（**1年+**） | FastVideo 更成熟 |
| Contributors | 少量 | **45+** | |
| Open Issues | 0 | 102 | FastGen 社区参与少 |
| 下游项目 | 几乎无 | **腾讯 HunyuanVideo、Kandinsky、LongCat** | |
| PyPI 安装 | 无 | **`pip install fastvideo`** v0.1.7 | |
| API Server | 无 | **OpenAI 兼容 API** (2026.03) | |
| 硬件适配 | NVIDIA only | **NVIDIA + AMD ROCm** | |

---

## 技术路线对比

```
FastGen 路线：专注蒸馏，减少采样步数
┌──────────────────────────────────────────┐
│  50步 teacher ──蒸馏──→ 1-4步 student    │
│                                          │
│  方法池：ECT / TCM / sCT / sCD /        │
│         MeanFlow / DMD2 / LADD /         │
│         f-distill / CausVid /            │
│         Self-Forcing / KD / SFT          │
└──────────────────────────────────────────┘

FastVideo 路线：全链路优化，蒸馏 + 推理 + 注意力
┌──────────────────────────────────────────┐
│  数据处理 → 训练 → 蒸馏 → 推理优化       │
│                                          │
│  蒸馏：DMD2 + Video Sparse Attention     │
│  推理：SageAttention + STA + TeaCache    │
│  并行：Sequence Parallelism (多卡)       │
│  因果：CausalWan-MoE (Self-Forcing)     │
└──────────────────────────────────────────┘
```

### 蒸馏方法覆盖

| 类别 | FastGen | FastVideo |
|------|---------|-----------|
| **轨迹类** | ECT, TCM, sCT, sCD, MeanFlow, iCT | — |
| **分布类** | DMD2, f-distill, LADD | DMD2 |
| **因果类** | CausVid, Self-Forcing | Self-Forcing |
| **微调类** | SFT, CausalSFT, KD, CausalKD | — |
| **稀疏蒸馏** | — | **VSA + DMD2 联合训练** |

FastGen 在蒸馏方法广度上占绝对优势。FastVideo 的独特价值在于 **VSA (Video Sparse Attention) 与 DMD2 的联合训练**——同时学习稀疏化和步数压缩，避免两阶段串联的质量损失。

### 推理优化覆盖

| 技术 | FastGen | FastVideo |
|------|---------|-----------|
| SageAttention | — | ✅ |
| Sliding Tile Attention (STA) | — | ✅ (10.45x attention 加速) |
| TeaCache | — | ✅ |
| torch.compile | 支持 | 支持 |
| Sequence Parallelism | Context Parallelism | ✅ (`num_gpus=N`) |
| FSDP2 | ✅ | ✅ |

FastVideo 在推理优化上全面领先。

### 支持模型

| 模型 | FastGen | FastVideo |
|------|---------|-----------|
| Wan2.1 / 2.2 | ✅ | ✅ |
| HunyuanVideo | — | ✅ (含 1.5) |
| Cosmos / Cosmos2.5 | ✅ | — |
| CogVideoX | ✅ | — |
| LTX2 | — | ✅ |
| SD / SDXL / Flux | ✅ | — |
| EDM / EDM2 | ✅ | — |
| LongCat | — | ✅ |

FastGen 覆盖更广（图像+视频），FastVideo 在视频侧更深。

---

## 性能基准对比

### TurboDiffusion 论文中的直接对比 (RTX 5090)

TurboDiffusion 同时使用了蒸馏（FastGen 路线）和推理优化（FastVideo 路线），并与 FastVideo 做了对比：

| 模型 | 原始耗时 | FastVideo | TurboDiffusion | FV 加速比 | Turbo/FV |
|------|----------|-----------|----------------|-----------|----------|
| Wan2.1-1.3B-480P | 184s | 5.3s | 1.9s | 35x | 2.8x |
| Wan2.1-14B-480P | 1,676s | 26.3s | 9.9s | 64x | 2.7x |
| Wan2.1-14B-720P | 4,767s | 72.6s | 24s | 66x | 3.0x |

**关键洞察**：蒸馏 + 推理优化两条路线**正交可叠加**，组合后比单用 FastVideo 再快 ~3x。

### FastVideo 自身基准 (H200)

| 配置 | 14B 720P | 1.3B 480P |
|------|----------|-----------|
| FlashAttention2 (基线) | 1,746.5s | 95.21s |
| FA2 + DMD 蒸馏 | 52s | 2.88s |
| VSA + DMD + torch.compile | **13s** | **0.98s** |
| **加速比** | **~134x** | **~97x** |

### FastVideo 端到端 (含 VAE 编解码)

| 模型 | GPU | 分辨率 | 耗时 |
|------|-----|--------|------|
| FastWan2.1-1.3B | H200 | 480P | **5s** |
| FastWan2.1-1.3B | RTX 4090 | 480P | 21s |
| FastWan2.2-5B | H200 | 720P | 16s |

### FastGen 基准

| 场景 | 加速比 | 备注 |
|------|--------|------|
| Wan2.1-14B (50→2步) | **50x** | DMD2 蒸馏 |
| Self-Forcing 实时视频 | **~150x** | 17.0 FPS |
| CorrDiff 天气模型 | **23x** | 科学模型 |

---

## 生态与社区

### FastVideo 生态（更强）

```
FastVideo
├── SGLang/LMSYS 官方 fork → 生产级推理引擎
├── AMD ROCm 适配
├── 下游项目
│   ├── 腾讯 HunyuanVideo 1.5
│   ├── Kandinsky-5.0
│   ├── LongCat Video (13.6B)
│   ├── HY-WorldPlay
│   ├── DanceGRPO / SRPO / DCM
│   └── ...
├── PyPI 发布 (pip install fastvideo)
├── OpenAI 兼容 API Server
└── Gradio Demo UI
```

### FastGen 生态（NVIDIA 背书）

```
FastGen
├── NVIDIA Research 官方维护
├── 统一蒸馏 benchmark
├── rCM (ICLR 2026) 参考实现
└── 社区采用尚早期
```

---

## 选型建议

| 场景 | 推荐 | 理由 |
|------|------|------|
| **工程落地 / 部署上线** | **FastVideo** | 成熟生态、SGLang 集成、API server、pip 安装 |
| **蒸馏方法研究 / 对比实验** | **FastGen** | 8+ 方法统一 API，公平对比 |
| **极致加速（两者组合）** | **FastVideo + FastGen 蒸馏** | 推理优化 × 步数蒸馏 = 乘法加速 |
| **实时交互视频** | **FastGen (Self-Forcing)** | 17 FPS，亚秒延迟，CausVid 支持 |
| **消费级 GPU 部署** | **FastVideo** | RTX 4090 实测数据，显存优化 |
| **多模态（图像+视频+科学）** | **FastGen** | 模型覆盖更广 |
| **HunyuanVideo 加速** | **FastVideo** | 原生支持，FastGen 不支持 |

### 组合使用示例

```
最优工作流：
1. 用 FastGen 选择并训练最佳蒸馏方法 (如 DMD2/rCM)
   → 50步 → 4步
2. 将蒸馏后模型导入 FastVideo 推理管线
   → SageAttention + STA + Sequence Parallelism
3. 部署到 SGLang Diffusion 生产服务
   → OpenAI 兼容 API，自动 batching
```

---

## 参考资料

- [NVIDIA FastGen GitHub](https://github.com/NVlabs/FastGen)
- [NVIDIA Blog: Accelerating Diffusion Models with FastGen](https://developer.nvidia.com/blog/accelerating-diffusion-models-with-an-open-plug-and-play-offering)
- [Hao AI Lab FastVideo GitHub](https://github.com/hao-ai-lab/FastVideo)
- [FastVideo V1 Blog](https://haoailab.com/blogs/fastvideo/)
- [FastWan: Sparse Distillation Blog](https://haoailab.com/blogs/fastvideo_post_training/)
- [Sliding Tile Attention Blog](https://haoailab.com/blogs/sta/)
- [CausalWan-MoE Preview](https://haoailab.com/blogs/fastvideo_causalwan_preview/)
- [SGLang Diffusion (LMSYS)](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)
- [FastVideo on PyPI](https://pypi.org/project/fastvideo/)
- [FastVideo on AMD ROCm](https://rocm.blogs.amd.com/artificial-intelligence/fastvideo-v1/README.html)
- [TurboDiffusion Paper](https://arxiv.org/abs/2512.16093)
- [CausVid Project Page](https://causvid.github.io/)
- [Causal Forcing GitHub](https://github.com/thu-ml/Causal-Forcing)
