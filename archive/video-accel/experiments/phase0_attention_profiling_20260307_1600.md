# 实验：Phase 0 注意力矩阵采集与分析
> 日期：2026-03-07 16:00

## 环境
- Commit: `2a98bc87ef18024f06517d013a0d80d4b82beab0`
- GPU: NVIDIA GeForce RTX 5090 (33.7 GB)
- Python: 3.10.12
- PyTorch: 2.10.0+cu128
- CUDA: 12.8
- diffusers: 0.37.0.dev0

## 运行命令
```bash
python scripts/phase0_attention_profiling.py \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --num_steps 50 \
  --num_prompts 10 \
  --sample_queries 512 \
  --seed 42 \
  --output_dir results/phase0
```

## 参数
| 参数 | 值 |
|------|-----|
| 模型 | Wan-AI/Wan2.1-T2V-1.3B-Diffusers |
| 分辨率 | 480 × 832 |
| 帧数 | 17 |
| 去噪步数 | 50 |
| Prompt 数 | 10 |
| 采样 Query 数 | 512 |
| Latent shape | [5, 60, 104] |
| Video tokens | 31200 |
| 随机种子 | 42 |
| 总耗时 | 5239.5s |

## 结果

### 性质 ① 块对角线主导

**结论: CONFIRMED**

- 平均比率: **9.067x** (同帧平均注意力 / 跨帧平均注意力)
- 中位数: 1.726x
- 预期: >2.5x (literature: 2.8x)

每头比率:
| Head | Ratio |
|------|-------|
| 0 | 9.789 |
| 1 | 8.545 |
| 2 | 14.219 |
| 3 | 4.138 |
| 4 | 8.666 |
| 5 | 5.396 |
| 6 | 6.734 |
| 7 | 10.367 |
| 8 | 9.998 |
| 9 | 3.81 |
| 10 | 12.841 |
| 11 | 14.307 |

### 性质 ② 时间局部性衰减

**结论: CONFIRMED**

- 单调递减: 是
- dist=0 注意力占比: **78.5%**
- 预期: Monotonic decay, dist=0 should have highest mass

衰减曲线:
| 帧距离 | 平均注意力 |
|--------|-----------|
| 0 | 0.784533 |
| 1 | 0.215467 |
| 2 | 0.0 |
| 3 | 0.0 |
| 4 | 0.0 |

### 性质 ③ 跨 Prompt 模式不变性

**结论: CONFIRMED**

- 方法: cosine similarity of per-head [block_diag_ratio, entropy] vectors across prompts
- 平均余弦相似度: **0.9564**
- 中位数: 0.9812
- 比较次数: 67500
- 预期: >0.9 (literature: >0.9 index overlap)

### 性质 ④ 去噪步间稳定性（U 形曲线）

**结论: NOT CONFIRMED**

- U 形: 否
- U 形比率 (边缘/中间): **1.1x**
- 前期变化: 0.695057
- 中期变化: 0.782429
- 后期变化: 0.857638
- 预期: U-shape: high→low→high change, mid ~70% steps stable

### 性质 ⑤ Head 功能特化

**结论: CONFIRMED**

分类统计:
| 类型 | 数量 |
|------|------|
| global | 8 |
| mixed | 272 |
| spatial | 78 |
| temporal | 2 |

总头数: 360
预期: 3-4 distinct types (spatial, temporal, global, sink)

### 性质 ⑦ 注意力熵分布

**结论: CONFIRMED**

| 指标 | 值 |
|------|-----|
| 平均熵 | 0.6117 |
| 标准差 | 0.2075 |
| 最小熵 | 0.0012 |
| 最大熵 | 0.9964 |
| 低熵头占比 (<0.5) | 25.3% |
| 高熵头占比 (>0.8) | 19.2% |

预期: Bimodal: some near 0 (identity-like), some near 1 (uniform-like)

### 注意力集中度

| Top-k% tokens | 覆盖注意力 |
|------|------|
| top_10pct | 0.8509 ± 0.1819 |
| top_1pct | 0.5776 ± 0.2903 |
| top_20pct | 0.9168 ± 0.1302 |
| top_5pct | 0.7735 ± 0.2264 |

## Go/No-Go 决策

| 性质 | 状态 |
|------|------|
| ① 块对角主导 | PASS |
| ② 时间衰减 | PASS |
| ③ 跨 prompt 不变性 | PASS |
| ④ U 形稳定性 | FAIL |
| ⑤ 头特化 | PASS |
| ⑦ 熵分布 | PASS |

**通过 5/6**

**决策：GO** — 进入 Phase 1（增量计算可行性验证）

## 下一步

基于以上结果，建议的下一步行动...
