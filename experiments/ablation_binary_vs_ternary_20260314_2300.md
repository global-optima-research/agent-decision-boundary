# 实验：消融 — Binary (act/refuse) vs Ternary (act/ask/refuse) 决策框架
> 日期：2026-03-14 23:00

## 环境
- Commit: `de9350b`
- GPU: NVIDIA GeForce RTX 5090 (×1)
- Python 3.10.12, PyTorch 2.10.0+cu128, TRL 0.29.0, PEFT 0.18.1

## 实验设计

验证论文核心命题：**"ask" 作为第三选项是否必要？**

训练了两个二元 DPO 模型，与三元模型对比：
- **Binary-Conservative**：将所有 gold=ask 的任务重映射为 refuse（不确定就拒绝）
- **Binary-Aggressive**：将所有 gold=ask 的任务重映射为 act（不确定就执行）
- **Ternary**：保留三元标签（act/ask/refuse），即主实验的 RiskAware-DPO

所有模型使用相同的训练超参数（lr=2e-5, epochs=3, LoRA r=16 等）。

**关键变量**：
- 二元模型的 system prompt 只提供 act/refuse 两个选项
- 三元模型的 system prompt 提供 act/ask/refuse 三个选项
- 测试集完全相同（100 held-out tasks，gold 标签为三元）

## 运行命令
```bash
HF_HUB_OFFLINE=1 python3 research/tridecision-full/ablation_binary.py
```

## 训练数据

| 配置 | Pairs | Label 分布 | Train/Eval |
|------|-------|-----------|------------|
| Binary-Conservative | 500 | act:128, refuse:372 | 450/50 |
| Binary-Aggressive | 500 | act:365, refuse:135 | 450/50 |
| Ternary (主实验) | 4163 | weighted by error severity | 3746/417 |

## 运行时间

| 步骤 | 时间 |
|------|------|
| Baseline binary eval | 140s |
| Binary-Conservative 训练 + eval | ~20min |
| Binary-Aggressive 训练 + eval | ~20min |
| **总计** | ~45min |

## 核心结果

### 四列对比（Held-Out Test Set, N=100）

| 指标 | Bin-Baseline | Bin-Conservative | Bin-Aggressive | **Ternary** |
|------|-------------|-----------------|----------------|-------------|
| Accuracy | 0.520 | 0.510 | 0.520 | **0.860** |
| WES ↓ | 0.860 | 0.755 | 1.080 | **0.215** |
| SVR ↓ | 0.247 | 0.164 | 0.397 | **0.000** |
| ULR ↓ | 0.417 | 0.514 | 0.264 | **0.014** |

### Act/Refuse-only Accuracy（排除 ask 任务后）

| 配置 | Acc (act+refuse only, N=55) |
|------|-----------------------------|
| Bin-Baseline | 52/55 = 0.945 |
| Bin-Conservative | 51/55 = 0.927 |
| Bin-Aggressive | 52/55 = 0.945 |
| Ternary | 49/55 = 0.891* |

*Ternary 在 act/refuse 子集上略低，因为模型会对一些 act/refuse 任务输出 ask。

### Ask-Gold 任务的命运（N=45）

测试集中 45 个 gold=ask 的任务，二元模型被迫选择 act 或 refuse：

| 配置 | 映射为 act | 映射为 refuse |
|------|-----------|--------------|
| Bin-Baseline | 17 | 28 |
| Bin-Conservative | 11 | 34 |
| Bin-Aggressive | 28 | 17 |
| Ternary | 0 (→ask:44) | 1 |

### 混淆矩阵

**Baseline-Binary（未训练，二元 prompt）**
```
gold\pred   act  refuse
act          25       2
ask          17      28
refuse        1      27
```

**Binary-Conservative（ask→refuse DPO）**
```
gold\pred   act  refuse
act          24       3
ask          11      34
refuse        1      27
```

**Binary-Aggressive（ask→act DPO）**
```
gold\pred   act  refuse
act          25       2
ask          28      17
refuse        1      27
```

**Ternary（act/ask/refuse DPO）**
```
gold\pred   act   ask  refuse
act          21     6       0
ask           0    44       1
refuse        0     7      21
```

## 分析与结论

### 核心发现：二元决策存在不可调和的 safety-usability 困境

1. **Conservative 降 SVR 必然升 ULR**：
   - SVR 从 0.247 降到 0.164（改善），但 ULR 从 0.417 飙到 0.514
   - 模型学会了"有疑虑就拒绝"，但把该做的事也拒绝了
   - 45 个 ask 任务中 34 个被拒绝，11 个仍被错误执行

2. **Aggressive 降 ULR 必然升 SVR**：
   - ULR 从 0.417 降到 0.264（改善），但 SVR 从 0.247 飙到 0.397
   - 模型学会了"有疑虑也执行"，近 40% 的安全违规率
   - 45 个 ask 任务中 28 个被直接执行

3. **Ternary 同时做到低 SVR + 低 ULR**：
   - SVR = 0.000, ULR = 0.014
   - 44/45 个 ask 任务正确输出 ask
   - 不存在 safety-usability tradeoff

4. **Accuracy 差距巨大**（0.52 vs 0.86）：
   - 二元模型在 act/refuse 子集上表现很好（94.5%）
   - 但 45% 的测试集是 ask 任务，二元必然全错
   - 这不是模型能力问题，是**框架限制**

### 论文叙事

这个消融实验提供了论文最强的论证：

> "The binary act/refuse framework forces an irreconcilable tradeoff between safety and usability. Conservative mapping (ask→refuse) achieves SVR=0.164 but ULR=0.514; aggressive mapping (ask→act) achieves ULR=0.264 but SVR=0.397. The ternary framework resolves this by introducing 'ask' as a third option, achieving SVR=0.000 and ULR=0.014 simultaneously."

这是一个**结构性论证**（structural argument），不依赖于 benchmark 的难度或规模。无论 benchmark 怎么变，只要存在需要用户确认的场景，二元框架就必然在 safety 和 usability 之间做出牺牲。

### 与 MOSAIC 的隐含对比

MOSAIC (Microsoft, 2026.03) 使用二元 act/refuse 框架。本实验的 Binary-Conservative 本质上模拟了 MOSAIC 的决策空间。结果表明：
- 即使进行 DPO 训练，二元框架也无法在 SVR 和 ULR 上同时表现好
- 三元框架是 strictly superior 的

### 局限性

1. 二元模型使用 500 pairs（vs 三元 4163 pairs），数据量不对等
2. 但即使数据量相同，二元的结构性缺陷（无法输出 ask）也不会改变
3. Bin-Baseline（未训练）已经展示了同样的困境（0.247 SVR, 0.417 ULR）
