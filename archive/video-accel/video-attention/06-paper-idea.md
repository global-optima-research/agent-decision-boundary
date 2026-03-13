# 论文思路：Video Attention Compiler

> 日期：2026-03-07
> 状态：构思中，待 Phase 2 验证

## 一句话

把视频 DiT 的注意力加速从「运行时优化」变成「离线编译」— 利用注意力模式的输入无关性，离线 profile 一次，生成 per-(layer, head) 异构 kernel 调度计划，推理时零额外开销。

## 核心观察（已验证）

| 观察 | 数据支撑 | 意义 |
|------|---------|------|
| 注意力模式输入无关 | 跨 prompt 余弦相似度 0.956 (Phase 0) | 可以离线 profile，结果对所有输入有效 |
| Head 功能特化 | 78 spatial + 8 global + 2 temporal + 272 mixed (Phase 0) | 不同 head 需要不同 kernel |
| 注意力极度集中 | top-1% 覆盖 57.8% mass (Phase 0) | 稀疏注意力有巨大空间 |
| 稀疏位置跨步稳定 | top-k% IoU = 0.82-0.88 (Phase 1d) | mask 可跨步复用 |
| 步间输出复用不可行 | 全策略 SSIM < 0.55 (Phase 1b) | 必须每步算新数值，不能复用旧输出 |
| 层位置 > 变化量 | r(change, SSIM) = +0.77 (Phase 1a/1c) | 调度需考虑层在模型中的位置 |

## 与现有工作的差异

### 为什么不是 STA？

STA 用固定的 sliding tile 模式，对所有 head/layer/step 一视同仁。我们的数据表明：
- 不同 head 的模式差异巨大（spatial vs global vs temporal）
- 用固定模式处理 global head 会丢失全局信息
- Per-head 异构比统一模式更优（待 Phase 2c 验证）

### 为什么不是 SVG？

SVG 做在线 profiling，运行时动态决策。我们的数据表明：
- 模式 95%+ 输入无关 → 在线 profiling 是浪费
- 离线编译可以做更重的优化（kernel 选择、内存布局），因为不受运行时延迟约束
- 零运行时开销 vs SVG 的 profiling 开销

### 为什么不是 TeaCache/AdaCache？

它们缓存整个 block 输出。我们的 Phase 1b 证明了注意力输出缓存在细粒度下失败。
它们的成功在于 block 级（包括 FFN）的冗余，而不是注意力级。
我们的方向不是缓存输出，而是用更少的计算得到同等质量的新输出。

### 为什么不是 LiteAttention？

LiteAttention 做跨步稀疏复用，但没有 per-head 异构和编译优化。
我们把 mask 复用作为编译框架的一个维度，而不是唯一手段。

## 论文框架

### Title 候选

- "Video Attention Compiler: Offline-Optimal Heterogeneous Sparse Attention for Video Diffusion"
- "Compile Once, Accelerate Forever: Input-Invariant Attention Optimization for Video DiT"
- "AttentionCC: Compiled Computation for Video Diffusion Attention"

### Contributions

1. **观察**：视频 DiT 注意力模式 95%+ 输入无关 + head 功能高度特化 → 离线编译可行
2. **方法**：Video Attention Compiler — 离线 profile → per-(layer, head) kernel 选择 → 跨步刷新调度 → 编译生成推理计划
3. **理论**：给定质量预算 τ，求最优计算分配的形式化问题 + 误差界
4. **系统**：Triton kernel 实现 + 多模型评估

### 方法细节

```
┌─────────────────────────────────────────────────┐
│              Video Attention Compiler             │
├─────────────────────────────────────────────────┤
│                                                   │
│  阶段 1: 离线 Profiling (一次性, ~10min)          │
│  ┌─────────────────────────────────────────┐     │
│  │ 跑 10-20 个 sample prompt               │     │
│  │ 对每个 (layer, head) 收集:              │     │
│  │   - 注意力分布特征 (entropy, block_diag) │     │
│  │   - top-k% 位置分布                     │     │
│  │   - 跨步变化率                          │     │
│  └─────────────────────────────────────────┘     │
│                    ↓                              │
│  阶段 2: 编译优化                                │
│  ┌─────────────────────────────────────────┐     │
│  │ 对每个 (layer, head):                   │     │
│  │   1. 分类: spatial/temporal/global/mixed │     │
│  │   2. 选 kernel: tile/strided/linear/full│     │
│  │   3. 选稀疏度: 5%/10%/20%/100%          │     │
│  │   4. 选刷新频率: 每 1/2/3/5 步          │     │
│  │                                         │     │
│  │ 全局优化:                               │     │
│  │   min Σ FLOPs(l,h)                      │     │
│  │   s.t. Quality ≥ τ                      │     │
│  └─────────────────────────────────────────┘     │
│                    ↓                              │
│  阶段 3: 推理 (零额外开销)                       │
│  ┌─────────────────────────────────────────┐     │
│  │ 按编译好的计划执行:                      │     │
│  │   Step 0: [L0:tile5% | L1:full | ...]   │     │
│  │   Step 1: [L0:reuse | L1:tile10% | ...] │     │
│  │   ...                                   │     │
│  │ 不需要任何运行时决策                     │     │
│  └─────────────────────────────────────────┘     │
│                                                   │
└─────────────────────────────────────────────────┘
```

### 理论部分（sketch）

**形式化**：

给定模型 M 有 L 层、H 头，去噪 T 步。
对每个 (l, h, t) 需要选择：
- 稀疏模式 p ∈ {tile, strided, linear, full}
- 稀疏度 k ∈ [0, 1]
- 是否复用上一步 mask: r ∈ {0, 1}

目标：
```
minimize  Σ_{l,h,t} C(p_{l,h,t}, k_{l,h,t}, r_{l,h,t})     # 总计算量
subject to  E[SSIM(Video_sparse, Video_full)] ≥ τ            # 质量约束
            r_{l,h,0} = 0  ∀l,h                              # step 0 必须全算
```

其中 C(·) 是计算代价函数，质量约束需要推导误差传播界。

**误差界**（关键挑战）：
- 单层稀疏误差: ||sparse_attn - full_attn|| ≤ f(k, distribution)
- 跨层传播: 需要考虑 residual connection 的稳定性
- 跨步传播: Phase 1 表明误差会放大，需要严格分析

### 实验计划

**Phase 2 (验证可行性, ~1 周)**:
- 2a: top-k% 稀疏注意力质量 → 确认稀疏本身可行
- 2b: mask 跨步复用 + 稀疏计算 → 确认复用可行
- 2c: per-head 异构 vs 统一模式 → 确认异构有价值

**Phase 3 (Triton kernel, ~2 周)**:
- 实现 tile/strided/linear 三种 kernel
- 调度器: 按编译计划分发
- 速度 benchmark vs STA/FA2

**Phase 4 (多模型 + 论文, ~2 周)**:
- 模型: Wan 1.3B, Wan 14B, HunyuanVideo, CogVideoX
- Baseline: STA, SVG, TeaCache, FA2
- 指标: kernel speedup, E2E speedup, SSIM, LPIPS, FVD
- 消融: 去掉异构/去掉编译/去掉复用 各自的影响

## 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Phase 2 稀疏注意力质量不达标 | 低 | 致命 | Phase 0 数据强烈支持 (top-1% = 57.8%) |
| Per-head 异构 ≤ 统一模式 | 中 | 高 | 退化为"带编译优化的 STA"，仍有贡献 |
| Triton kernel 效率不够 | 中 | 中 | 不规则稀疏模式 GPU 不友好，可能需要近似为规则模式 |
| 跨模型泛化性差 | 低 | 中 | 编译范式本身是模型无关的 |
| 理论误差界太松 | 中 | 低 | 实验结果足够强的话理论可以弱一些 |

## 目标投稿

- ICML 2027 (DDL ~Jan 2027) — 如果能 6 月前完成
- NeurIPS 2026 (DDL ~May 2026) — 非常紧张但如果 Phase 2/3 顺利
- 备选: ICLR 2027

## 与全景图的关系

在技术全景图中，本工作的定位：
- **不是** 步数蒸馏（不改模型）
- **不是** 特征缓存（不缓存输出）
- **是** 稀疏注意力，但加了 "编译" 维度
- 与量化正交，可叠加 (SageAttention/FP8)
- 落在"轻量配置"到"即插即用"的区间（离线 profile 几分钟，不需训练）
