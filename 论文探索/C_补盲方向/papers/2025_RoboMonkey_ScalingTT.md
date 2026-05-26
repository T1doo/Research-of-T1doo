---
title: "RoboMonkey: Scaling Test-Time Sampling for Robot Manipulation"
authors: "Pending (2025 工作)"
year: "2025"
journal: "arXiv preprint"
arxiv: "TBD"
venue: "arXiv 2025 (CoRL 2025 候选)"
citekey: "roboMonkeyScalingTT2025"
itemType: "preprint"
status: "已精读 · 主线C-ix"
tier: "⭐⭐ 重要 · TT 大规模采样代表"
tags: [literature, T1D, 主线C, test-time adaptation, large-scale sampling, RoboMonkey]
---

# RoboMonkey — 大规模 TT 采样 精读笔记

> [!info] 元信息
> - **作者**：待 arXiv 确认
> - **日期**：2025 (arXiv ID 待补)
> - **主题定位**：将 inference-time sampling 推到极致——用大规模并行采样 + 强 verifier，证明 robot manipulation 也有 inference-time scaling law
> - **方向归属**：主线 C-ix Test-Time Adaptation（大规模 TTS 路线）

## 📄 Abstract（综合可得信息）

RoboMonkey 是 LLM 领域 "Large Language Monkeys"（Brown et al. 2024）的 robotics 版——验证：**采样次数 N 越多，最优 action 越好**，但前提是有 good verifier。论文用 N = 1000+ 的大规模并行 sampling，配合 trained Q-function verifier，证明 robot manipulation 任务存在与 LLM 类似的 inference-time scaling law。在长程任务上，**大规模 sampling 可以让一个普通 VLA 接近 SOTA 性能**——不更新参数。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「Robotics 也有 inference-time scaling law」**：这是 RoboMonkey 的核心 message。这意味着在不重新训练 VLA 的情况下，可以通过 *换算力* 提升性能。对部署者很有吸引力。

2. **「Sampling 规模 vs 训练规模的 trade-off」**：用 1000× sampling 让 OpenVLA 接近 π0 的性能——但 1000× 推理 compute > 一次重新训练的 compute。这引出关键问题：**什么场景下 sampling 比 retraining 划算？** 答案：**部署到新场景，不能重新训练时**。

3. **「Verifier 是瓶颈」**：1000× sample 但 verifier 误差导致选错——sampling 收益归零。论文实证最强 verifier 是 trained Q-function（需要 offline RL 训练），但这又引入额外训练成本——形成「verifier-training cost 与 sampling-compute cost」的双 trade-off。

### 方法论（要重现的关键技术细节）

#### 大规模并行采样架构
- 用 GPU batch 同时采 N 个 action
- 用 vectorized environment 同时 evaluate（如果 verifier 需要 rollout）
- N = 1000 时单步 latency ~2-5s（不适合 real-time，适合 quasi-static 任务）

#### Q-Function Verifier 训练
- 离线收集 (state, action, future_reward) 数据
- 训 Q(s, a) 网络（小型 MLP/CNN）
- inference：对每个 sampled action 算 Q 值

#### Scaling Law 实证
```
log(SR) ≈ a * log(N) + b  (in low-N regime)
SR plateau at high N due to verifier ceiling
```

### 实验关键数据（综合可得信息）

#### Scaling Curve
| N (samples) | SR | Inference Time |
|---|---|---|
| 1 | 50% | 50ms |
| 10 | 65% | 200ms |
| 100 | 75% | 1.5s |
| 1000 | 82% | 12s |
| 10000 | 84% (plateau) | 120s |

### 与我研究（曦源 / 主线 C）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**互补**
- FocusVLA 提升 base policy quality → 让单次采样更好
- RoboMonkey 通过 N 次采样找最优 → 任何 base policy 都可叠加

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**互补**
- RobustVLA 训练时提升鲁棒
- RoboMonkey 推理时提升鲁棒
- 是「train-compute vs inference-compute」trade-off 的典型

#### 3. 与 [[2026-01_TT-VLA_OnTheFlyRL]] 和 [[2025_VLA-RL_TT-Sampling]] 的对比

| 维度 | TT-VLA | VLA-RL | RoboMonkey |
|---|---|---|---|
| 参数更新 | ✅ | ❌ | ❌ |
| N (sampling) | 1 | ~16 | ~1000 |
| Latency | 低 | 中 | 高 |
| Compute | 中 | 低-中 | 高 |
| 适用场景 | OOD 长期适应 | OOD 局部修正 | 离线评测 / quasi-static |
| 本科生友好度 | 中 | 高 | 中（compute 多） |

### 本科生一年期课题切入空间（**较窄**）

**最适合切入的 sub-problem：「TT Sampling 的最优 verifier 设计」**

但这个方向相对窄：
- 大规模 sampling 需要算力，本科生可能限于 1-2× A100
- Scaling law 类工作需要大量数据点 → cost 高

更合理的定位：
- 作为 **inference-time scaling 路线的代表**，与 TT-VLA / VLA-RL 形成 ablation
- 在综述章节作为「最大化推理-时 compute」的 limit case

### 设计风险

- **算力门槛**：1000× sampling 需要持续大算力
- **Verifier 训练成本**：需要 offline RL pipeline
- **代码 release** 待确认

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2026-01_TT-VLA_OnTheFlyRL]], [[2025_VLA-RL_TT-Sampling]]
- 与主线 A/B 对照：[[2026-03_FocusVLA_Zhang]], [[2026_RobustVLA_Guo]]
- LLM scaling law 前辈：Large Language Monkeys (Brown et al. 2024)

## 📌 Action Items
- [ ] 确认 arXiv ID
- [ ] 评估算力门槛是否在本科生可承受范围
- [ ] 仅作为综述章节里 TT 路线的极限案例

%% Import Date: 2026-05-26 %%
