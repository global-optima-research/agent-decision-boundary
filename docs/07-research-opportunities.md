# 研究机会分析

> 调研日期：2026-03-06
> 空白验证日期：2026-03-06

基于对视频模型推理加速 6 大方向的全面调研，识别出以下研究空白和机会。**所有机会均经过文献验证。**

---

## 机会总览

| 优先级 | 方向 | 空白度 | 新颖性 | 工程量 | 潜在影响 |
|--------|------|--------|--------|--------|----------|
| **1** | 少步模型缓存失效 + 协同设计 | ⚠️ 半空白 | 中 | 中 | 直接出论文 |
| **2** | DuCa 随机性理论推广 | ✅ 确认空白 | 很高 | 低 | 理论贡献大 |
| **3** | 长视频加速 Scaling 规律 | ✅ 确认空白 | 高 | 中 | 实际需求强 |
| **4** | 自适应组合框架 | ✅ 确认空白 | 高 | 高 | 工程价值大 |
| **5** | 统一 benchmark | ✅ 确认空白 | 中 | 高 | 社区价值大 |
| **6** | 消费级 GPU 端到端优化 | ⚠️ 半空白 | 低 | 中 | 用户群大 |

---

## 1. 少步模型的缓存失效问题

### 空白验证：⚠️ 半空白 — 已有直接相关工作，但仍有开放问题

**已有关键工作**：

| 论文 | 时间 | 发现 |
|------|------|------|
| **DisCa** (CVPR'26 投稿) | 2026.02 | **最直接相关**。TeaCache 在 MeanFlow 蒸馏的 HunyuanVideo(10步) 上 **-15.5% 语义分, -5.6% VBench**。提出 Restricted MeanFlow + 可学习神经预测器，11.8x 加速 |
| **OmniCache** (ICCV'25) | 2025.08 | 在蒸馏后的 CogVideoX-5b(16步) 上，PAB/Delta-DiT/Learn-to-Cache **全部模型崩溃**。提出轨迹曲率分析替代相似度启发式 |
| **CacheQuant** (CVPR'25) | 2025.03 | 证明缓存+量化**不是正交的**——单独 FID 损失 0.76 和 4.71，组合后 **11.99**（近 3x 叠加） |
| **Video-BLADE** | 2025.08 | 稀疏注意力+蒸馏联合训练优于分开训练（VBench 0.570 vs 0.528） |

**仍然开放的问题**：

1. ~~首个系统研究蒸馏×缓存交互效应~~ → DisCa 已做，但**仅测了 TeaCache 和 TaylorSeer 两种缓存方法，仅在一个模型(HunyuanVideo)一种蒸馏(MeanFlow)上**
2. **缺失**：在多个蒸馏模型（DCM/DOLLAR/rCM/LCM）× 多个缓存方法（TeaCache/AdaCache/DuCa/MixCache）的全矩阵对比
3. **缺失**：三方法交互（蒸馏+缓存+量化），CacheQuant 只做了缓存+量化
4. **缺失**：图像蒸馏模型（FLUX Schnell 4步/SDXL-Turbo 1步）上的缓存研究
5. **缺失**：cache-aware distillation 的通用框架（DisCa 的 Restricted MeanFlow 是针对特定蒸馏方法的）

### 修正后的定位

不再是"首个研究"，而是：
> **扩展 DisCa 的发现**：在多蒸馏方法×多缓存方法×多模型的全矩阵上建立交互效应的系统认知，并探索三方法联合交互（蒸馏+缓存+量化）

### 需要 acknowledge 的先行工作

- **DisCa** (2026.02) — 首个蒸馏兼容缓存，Restricted MeanFlow
- **OmniCache** (ICCV'25) — 轨迹曲率分析，蒸馏模型上的缓存崩溃现象
- **CacheQuant** (CVPR'25) — 缓存×量化非正交性证明
- **Video-BLADE** — 稀疏注意力×蒸馏联合训练

### 实验方案（修正版）

```
实验 1：全矩阵对比（DisCa 的扩展）
  蒸馏方法: MeanFlow / DCM / DOLLAR / rCM
  缓存方法: TeaCache / AdaCache / DuCa / MixCache / OmniCache / DisCa
  模型: Wan2.1-14B / HunyuanVideo / CogVideoX-5B
  → 4×6×3 = 72 个组合的加速比和质量矩阵

实验 2：三方法交互（CacheQuant 的扩展到视频+蒸馏）
  蒸馏(4步) × 缓存 × 量化(FP8/W8A8/W4A8)
  → 验证误差是否像 CacheQuant 发现的那样乘法耦合

实验 3：通用 cache-aware distillation
  在蒸馏 loss 中加入步间特征平滑性正则
  → 对比 DisCa 的 Restricted MeanFlow（针对特定方法 vs 通用方案）
```

---

## 2. DuCa "随机 ≈ 重要性" 的理论推广 ★ 最推荐

### 空白验证：✅ 确认空白 — 无统一理论，多个独立发现等待连接

**已有相关工作（独立发现，未统一）**：

| 论文 | 领域 | 发现 |
|------|------|------|
| **DuCa** (2024.12) | 扩散模型 token 缓存 | 随机选 token ≈ 按注意力分数选，多样性 > 重要性 |
| **DART** (EMNLP'25) | 多模态 LLM token 剪枝 | 独立发现 "duplication matters more than importance"，提供 Lipschitz 连续性界 |
| **DivPrune** (CVPR'25) | 视觉 token 剪枝 | 形式化为 Max-Min Diversity Problem，多样性选择优于重要性选择 |
| **IDPruner** (2026) | token 剪枝 | 用 MMR 协调重要性+多样性 |
| **"Unreasonable Effectiveness of Random Pruning"** (ICLR'22) | 网络剪枝 | 随机剪枝在过参数化网络中表现接近精心设计的剪枝 |
| **Hidden Semantic Bottleneck** (2025) | DiT 分析 | DiT embedding 角相似度 >99%，2/3 维度可无损删除 — **极端冗余** |
| **MAE** (CVPR'22) | 视觉自监督 | 随机 mask 75% patch 仍能有效学习 — 空间冗余 |

**确认的空白**：

1. **无统一理论框架**连接以上所有发现（DuCa ↔ DART ↔ DivPrune ↔ Random Pruning ↔ MAE）
2. **无跨粒度验证**：随机选择在 token/block/layer/step 各级别是否都有效？
3. **无信息论解释**：为什么扩散模型的冗余是"均匀分布"的？与去噪过程（逐步去除均匀高斯噪声）的关系？
4. **无形式化证明**：目前最好的理论是 DART 的 Lipschitz 界，但范围窄，且未连接到扩散模型
5. DuCa 和 DART **互不引用** — 两个社区独立发现了同一现象

### 研究问题

1. 统一理论：能否用信息论（互信息/熵）或随机矩阵理论（Johnson-Lindenstrauss 投影）解释"随机 ≈ 重要性"？
2. 跨粒度验证：token/block/layer/step 四个级别，随机 vs 精心设计的选择，差距分别是多少？
3. 边界条件：在什么情况下随机选择会失效？（极高压缩率？特定模型架构？）
4. 扩散模型特殊性：这个现象是扩散模型特有的（与去噪过程有关），还是过参数化网络的普遍属性？

### 可能的贡献

- **"On the Unreasonable Effectiveness of Random Selection in Diffusion Model Acceleration"** — 桥接论文
- 连接 DuCa (扩散缓存) ↔ DART (多模态 LLM) ↔ DivPrune (视觉剪枝) ↔ 随机剪枝理论
- 信息论分析：量化 DiT 各层/各步的冗余均匀性
- 实践影响：如果随机足够好 → 大幅简化所有选择策略设计

### 实验方案

```
实验 1：跨粒度验证（核心贡献）
  在 3+ 模型 (Wan2.1-14B / HunyuanVideo / FLUX) 上：
  - Token 级: 随机 vs attention vs L2-norm vs 梯度
  - Block 级: 随机 vs 相似度 vs 方差
  - Step 级: 随机跳过 vs TeaCache 自适应 vs 固定间隔
  → 每个级别画 "随机 vs 最优选择" 的 Pareto 曲线

实验 2：信息论分析
  - 计算各 block/token/step 的互信息和条件熵
  - 用 PCA/SVD 分析特征空间的有效维度
  - 验证 "Hidden Semantic Bottleneck" 的发现是否扩展到视频模型

实验 3：边界条件
  - 扫描压缩率 10%-90%，找到随机选择失效的拐点
  - 对比不同模型规模 (1.3B vs 5B vs 14B) 的容忍度
  - 蒸馏后模型 vs 原始模型：随机选择的有效性是否变化？

实验 4：理论证明
  - 基于 DART 的 Lipschitz 界 + Hidden Semantic Bottleneck 的冗余证据
  - 尝试用 Johnson-Lindenstrauss 或浓度不等式建立更紧的界
```

---

## 3. 长视频加速的 Scaling 规律

### 空白验证：✅ 确认空白（已在上一轮验证）

**已有相关工作**：
- **LongLive** (NVIDIA, ICLR'26) — 实时长视频生成（30s-240s），short window attention + frame sink，20.7 FPS
- **HiStream** (2025.12) — 3 轴冗余消除（空间/时间/步数），anchor-guided sliding window，76-107x 加速
- **Video-Infinity** (2024.06) — 分布式 clip 并行，8 GPU 生成 2300 帧
- **BlockVid** (2025.11) — 半自回归块扩散 + 语义感知 KV cache，分钟级视频
- **SANA-Video** (NVIDIA, ICLR'26 Oral) — Block Linear DiT + constant-memory KV cache，分钟级 720p
- **FlowCache** (2026.02) — 首个自回归视频缓存框架（MAGI-1, SkyReels-V2）
- **Quant VideoGen** (2026.02) — 2-bit KV cache 量化，自回归长视频 ~700 帧
- **AdaSpa** (ICCV'25) — 自适应稀疏注意力，**唯一画了加速比 vs 视频长度曲线**（最长 24s，1.59x→4.01x）

**确认的空白**：
- TurboDiffusion (200x)、TeaCache、STA 等主流加速方法**全部在 2-5 秒短视频上验证**
- **没有论文系统研究加速方法随视频长度增长的性能变化规律**（30s/60s/120s 上各方法表现如何？）
- **没有论文对比缓存误差随长度的累积模式**
- BLADE 论文自己承认这个 gap："extending to minute-long videos remains an important next step"
- AdaSpa 是最接近的——画了 scaling 曲线，但只到 24 秒就停了

### 精确定位

~~首个系统研究长视频加速的工作~~ → **首个系统研究推理加速技术随视频长度扩展的性能变化规律的工作**

核心问题：在 5s/15s/30s/60s/120s 上对主流加速方法（缓存/稀疏注意力/蒸馏/量化）做 scaling 曲线，揭示长度依赖的冗余模式和误差累积规律。

### 研究问题

1. 主流加速方法（TeaCache/STA/蒸馏/FP8）在 30s/60s/120s 上的加速比和质量衰减曲线？
2. 缓存误差是否随长度线性累积？还是存在相变点（如场景切换时突然恶化）？
3. 长视频的冗余分布与短视频有何不同？场景内 vs 场景间的冗余模式？
4. 能否设计长度感知的自适应策略？（短视频用一套参数，长视频自动调整）

### 可能的贡献

- 首个加速方法 × 视频长度的系统 scaling 研究（5s-120s）
- 长视频特有的误差累积规律和相变点发现
- 长度感知的自适应加速策略

### 需要 acknowledge 的先行工作

- HiStream 的 3 轴冗余框架（聚焦高分辨率而非长时间）
- AdaSpa 的长度 scaling 曲线（只到 24s）
- LongCat 的 Block Sparse Attention（没做加速方法横向对比）
- FlowCache / Quant VideoGen（聚焦自回归范式，非 DiT 加速方法对比）

---

## 4. 自适应组合框架

### 空白验证：✅ 确认空白 — 有零散相关工作，但核心问题未被解决

**已有相关工作**：

| 论文 | 时间 | 做了什么 | 局限 |
|------|------|----------|------|
| **DiffBench + DiffAgent** | 2026.01 | LLM + 遗传算法自动生成加速代码，604 个任务 | **仅图像**，方法有限（ToMe/DeepCache/T-Gate/FP16），无 prompt 感知 |
| **CacheQuant** (CVPR'25) | 2025.03 | 缓存+量化组合，证明非正交 | 仅缓存+量化两方法，仅图像 |
| **DisCa** | 2026.02 | 缓存+蒸馏组合 | 仅缓存+蒸馏两方法 |
| **UniCP** | 2025.02 | 缓存+剪枝统一框架 | 仅缓存+剪枝两方法 |
| **Q-VDiT** (ICML'25) | 2025.05 | 量化+蒸馏联合 | 仅量化+蒸馏两方法 |
| **SADA** (ICML'25) | 2025.07 | 稳定性引导的自适应缓存 | 单方法内自适应，非跨方法选择 |
| **AdaCache** (ICCV'25) | 2024.11 | 运动感知自适应缓存 | 单方法内自适应 |

**确认的空白**：

1. **没有系统接收 (prompt + 硬件约束) 输出最优多方法组合** — DiffAgent 最接近但仅限图像且方法覆盖窄
2. **没有 3+ 方法的交互效应研究** — 所有现有工作只研究两两组合
3. **没有 prompt 感知的方法选择**（而非方法内参数调整）
4. **没有视频扩散模型的自动配置搜索**
5. **没有统一 benchmark 跨方向公平对比** — DiffBench 仅图像

### 修正后的定位

> **首个面向视频扩散模型的自适应多方法加速配置框架** — 接收 prompt + 硬件约束，输出最优的蒸馏/缓存/量化/稀疏注意力组合配置

### 需要 acknowledge 的先行工作

- **DiffBench/DiffAgent** — 图像领域的自动加速代码生成
- **CacheQuant/DisCa/UniCP/Q-VDiT** — 两两组合的交互研究
- **SADA/AdaCache** — 单方法内的自适应思路

### 设想

```
输入: prompt + target_quality + hardware + latency_budget
  │
  ▼
[轻量分析器] → prompt 复杂度 / 运动强度估计
  │
  ▼
[配置选择器] → 从预定义配置空间中选择最优组合
  │
  ├── 简单场景 → 激进缓存 (TeaCache aggressive) + 少步 + 高量化
  ├── 复杂场景 → 保守缓存 + 更多步数 + 低量化
  └── 低端 GPU → 自动启用 offloading + 更激进量化 + VAE tiling
  │
  ▼
输出: 最优配置 + 预估质量/速度
```

---

## 5. 统一 benchmark

### 空白验证：✅ 确认空白

**已有 benchmark**：
- **VBench / VBench 2.0** — 视频生成质量评测，不涉及加速方法对比
- **DiffBench** (2026.01) — 扩散模型加速 benchmark，**仅图像**（SD1.5/SDXL/DiT/PixArt），方法覆盖有限
- **FastGen** — 蒸馏方法对比框架，不覆盖缓存/量化/稀疏注意力
- **FastVideo** — 推理优化框架，不做跨方向公平对比

**确认没有**：一个涵盖视频扩散模型所有 6 个加速方向的统一 benchmark。

### 可能的贡献

```
VideoAccelBench:
├── 模型固定: Wan2.1-14B + HunyuanVideo
├── 分辨率/帧数: 480p/720p, 81/121 帧
├── 评测维度:
│   ├── 速度: 端到端延迟 + DiT-only 延迟 + 首帧延迟
│   ├── 质量: VBench + FVD + 人工偏好
│   ├── 资源: 峰值显存 + 平均 GPU 利用率
│   └── 鲁棒性: 100 条不同复杂度 prompt 的方差
├── 方法覆盖: 6 个方向 × 各方向 top-3 方法
├── 组合测试: 推荐组合的叠加效果
└── 硬件矩阵: RTX 4090 / H100 / A100 对比
```

---

## 6. 消费级 GPU 端到端优化

### 空白验证：⚠️ 半空白 — 社区工具丰富，学术系统性不足

**已有工具/项目**：

| 项目 | 类型 | 说明 |
|------|------|------|
| **SVDQuant / Nunchaku** (ICLR'25 Spotlight) | 学术+工具 | 4-bit PTQ，4090 Laptop 上跑 12B FLUX，HunyuanVideo 降至 ~6.5GB |
| **Wan2GP** | 社区工具 | "GPU Poor" 工具箱，6GB+ VRAM，auto-quantize INT8，支持 Wan/HunyuanVideo/LTX |
| **LTX Desktop** | 桌面应用 | 最接近 "llama.cpp for video"，自动检测硬件，RTX 3060 12GB+ 可用 |
| **stable-diffusion.cpp** | C++ 库 | 真正的 C++ 实现 + GGUF，但视频支持初期 |
| **ComfyUI + GGUF** | 生态 | 事实标准，NVIDIA CES 2026 合作，NVFP4/FP8 支持 |
| **ComfyUI 节点** (Kijai) | 社区 | Block Swap Offloading，每个 block 400-700MB 可选择性卸载 |
| **Diffusers** | 框架 | CPU offloading / VAE tiling / Group Offloading |
| **SageAttention** (ICLR'25) | 学术 | RTX 4090 上 FA2 3x 加速，零质量损失 |

**确认的空白**：

1. **没有论文系统对比消费级(24GB) vs 数据中心(80GB) GPU 上各加速方法的效果差异**
2. **没有 "24GB VRAM 优化 cookbook"** — 量化+offloading+缓存+稀疏注意力在 24GB 限制下的最优组合
3. **没有真正的 "ollama for video"** — LTX Desktop 最接近但仅支持 LTX 模型
4. **GGUF 视频模型生态远不如 LLM 成熟** — 无系统质量基准（Q2-Q8 对视频时间一致性的影响）
5. 社区工具（Kijai/city96/deepbeepmeep）**在实践上领先学术界**，但缺乏系统性分析

### 修正后的定位

不建议作为纯学术方向（社区已在工程上解决大部分问题），更适合作为：
> **系统性 benchmark 论文**：在 RTX 4090(24GB) vs H100(80GB) 上对比所有加速方法，发现 VRAM 受限下的不同最优策略

---

## 优先级修正（验证后）

```
验证前:                          验证后:
1. 缓存×蒸馏 ★最推荐            2. DuCa 随机性理论 ★最推荐 (确认空白，低工程量)
2. DuCa 随机性理论               1. 缓存×蒸馏 (DisCa 已占先，需差异化)
3. 长视频 scaling                3. 长视频 scaling (确认空白)
4. 自适应组合框架                4. 自适应组合框架 (确认空白，高工程量)
5. 统一 benchmark               5. 统一 benchmark (确认空白)
6. 消费级 GPU                   6. 消费级 GPU (社区已部分解决)
```

**最推荐路径变更**：机会 2（DuCa 随机性理论）升为第一推荐——空白度最高、工程量最低、理论贡献最大，且有 5+ 独立发现可以连接成一篇高影响力的桥接论文。

---

## 交叉机会

```
机会 2 (随机性理论) ←──理论基础──→ 机会 1 (缓存×蒸馏)
       │                              │
       │  理论 2 可解释为什么          │  DisCa 证实了蒸馏后
       │  缓存在少步模型中             │  缓存会失效
       │  失效或仍然有效               │
       │                              │
       ▼                              ▼
机会 4 (自适应框架) ←──需要──→ 机会 5 (统一 benchmark)
       │                              │
       │  框架需要 benchmark            │
       │  来评估配置效果               │
       │                              │
       ▼                              ▼
机会 3 (长视频) ←──应用场景──→ 机会 6 (消费级 GPU)
```

**最高效路径**：从机会 2（随机性理论）入手 → 积累的信息论分析工具可直接用于机会 1（解释缓存失效） → 两者的发现支撑机会 4/5。

---

## 参考

### 本调研文档
- [README](../README.md) | [技术全景图](panorama.md) | [在线可视化](https://video-infer-acc.optima.sh/)
- 各方向深入调研：[01](01-step-distillation.md) | [02](02-feature-caching.md) | [03](03-token-pruning-sparse-attention.md) | [04](04-quantization.md) | [05](05-vae-pipeline-optimization.md) | [06](06-hardware-deployment.md)

### 验证过程中发现的新论文

**缓存×蒸馏方向**：
- [DisCa: Distillation-Compatible Learnable Feature Caching (2026.02)](https://arxiv.org/abs/2602.05449)
- [OmniCache: Trajectory-Oriented Cache Reuse (ICCV'25)](https://arxiv.org/abs/2508.16212)
- [CacheQuant: Comprehensively Accelerated Diffusion Models (CVPR'25)](https://arxiv.org/abs/2503.01323)

**随机性理论方向**：
- [DART: Duplication Matters More (EMNLP'25)](https://arxiv.org/abs/2502.11494)
- [DivPrune: Max-Min Diversity (CVPR'25)](https://arxiv.org/abs/2503.02175)
- [IDPruner: Importance-Diversity Harmonization (2026)](https://arxiv.org/abs/2602.13315)
- [Hidden Semantic Bottleneck in DiTs (2025)](https://arxiv.org/abs/2602.21596)
- [Unreasonable Effectiveness of Random Pruning (ICLR'22)](https://arxiv.org/abs/2202.02643)

**自适应框架方向**：
- [DiffBench + DiffAgent (2026.01)](https://arxiv.org/abs/2601.03178)
- [UniCP: Unified Caching + Pruning (2025.02)](https://arxiv.org/abs/2502.04393)
- [Q-VDiT: Quantization + Distillation (ICML'25)](https://arxiv.org/abs/2505.22167)
- [SADA: Stability-guided Adaptive Acceleration (ICML'25)](https://arxiv.org/abs/2507.17135)

**消费级 GPU 方向**：
- [SVDQuant / Nunchaku (ICLR'25 Spotlight)](https://arxiv.org/abs/2411.05007)
- [Wan2GP: GPU Poor Toolbox](https://github.com/deepbeepmeep/Wan2GP)
- [LTX Desktop](https://github.com/Lightricks/LTX-Desktop)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

**长视频方向**（上轮验证）：
- [LongLive (ICLR'26)](https://arxiv.org/abs/2509.22622)
- [HiStream (2025.12)](https://arxiv.org/abs/2512.21338)
- [BlockVid (2025.11)](https://arxiv.org/abs/2511.22973)
- [SANA-Video (ICLR'26 Oral)](https://arxiv.org/abs/2509.24695)
- [FlowCache (2026.02)](https://arxiv.org/abs/2602.10825)
- [Quant VideoGen (2026.02)](https://arxiv.org/abs/2602.02958)
- [AdaSpa (ICCV'25)](https://arxiv.org/abs/2502.21079)
