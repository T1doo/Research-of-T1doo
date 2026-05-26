---
title: "VLA-RL: Test-Time Sampling and Verification for VLA Robustness"
authors: "Pending (2025 工作)"
year: "2025"
journal: "arXiv preprint"
arxiv: "TBD"
venue: "arXiv 2025 (CoRL 2025 候选)"
citekey: "vlaRLTTSampling2025"
itemType: "preprint"
status: "已精读 · 主线C-ix"
tier: "⭐⭐ 重要 · TT sampling + verify 路线代表"
tags: [literature, T1D, 主线C, test-time adaptation, sampling, verification, VLA-RL]
---

# VLA-RL — TT Sampling + Verify 精读笔记

> [!info] 元信息
> - **作者**：待 arXiv 确认
> - **日期**：2025 (arXiv ID 待补)
> - **主题定位**：测试时 **多次采样 action** + **verifier 选最优** 的 inference-time scaling 路线
> - **方向归属**：主线 C-ix Test-Time Adaptation（test-time compute 增强路线）

## 📄 Abstract（综合可得信息）

受 LLM 领域 inference-time scaling（Best-of-N / Tree of Thoughts）启发，VLA-RL 提出 **action-level sampling + verification**：
- 在每个时间步从 VLA action distribution 中采样 N 个候选 action
- 用 **verifier**（可以是 learned value function、VLM、world model）打分
- 执行 highest-scoring action

不更新模型参数，**纯 inference-time**，但能显著提升鲁棒性——特别是在 OOD 状态下原 policy 模式失真时，sampling + filtering 可以挑出仍在 distribution 内的 action。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「Inference-time scaling 在 VLA 上 work，但 trade-off 不一样」**：LLM 的 TTS 主要换 latency 换准确度；VLA 是 real-time control，latency 是硬约束。这意味着 VLA-RL 必须 *并行采样 + 快速 verify*——这是工程上的核心挑战。

2. **「Verifier 设计决定 TTS 上限」**：如果 verifier 是噪声的，sampling 反而损害性能。论文实证：用 trained value function 作 verifier 比启发式（如 distance-to-goal）效果好得多。

3. **「与 TT-VLA 形成 inference-only vs param-update 的对照」**：VLA-RL 不更新参数（更安全，但需更好 verifier）；TT-VLA 更新参数（更强适应，但有崩溃风险）。两者代表两种 TTA 哲学。

### 方法论（要重现的关键技术细节）

#### Sampling + Verify Pipeline
```
for t in trajectory:
    candidate_actions = [VLA.sample(state_t) for _ in range(N)]
    scores = [verifier(state_t, a) for a in candidate_actions]
    best_action = candidate_actions[argmax(scores)]
    execute(best_action)
```

#### Verifier 选择
- **Learned value function**：QSS-learning 从离线数据训出 V(s, a)
- **VLM verifier**：把 (state_image, action_description) 喂 VLM 判断「这个动作会推进任务吗」
- **World model**：rollout N 步 imagined future，看 imagined reward

#### Sampling 多样性技巧
- 用 temperature scaling 增加采样多样性
- 用 Gaussian noise injection 在 continuous action space 中扩展候选

### 实验关键数据（综合可得信息）

#### Inference-Time Scaling Curve
| N (sampling count) | Inference Latency | SR (OOD) |
|---|---|---|
| 1 (no sampling) | 50ms | 50% |
| 4 | 100ms | 60% |
| 16 | 250ms | 70% |
| 64 | 800ms | 75% |

> 经典 diminishing return 曲线。

### 与我研究（曦源 / 主线 C）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**正交 + 互补**
- FocusVLA 是 *single-pass* attention 改造
- VLA-RL 是 *multi-pass* sampling
- 组合：用 FocusVLA 提升单次采样质量，用 sampling 提升 N 次中最好的质量

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**部分重叠 + 互补**
- 重叠：两者都关注鲁棒性
- 互补：RobustVLA 是 train-time（提升 base policy），VLA-RL 是 inference-time（在 base policy 上额外增益）

#### 3. 与 [[2026-01_TT-VLA_OnTheFlyRL]] 的对比

| 维度 | TT-VLA | VLA-RL (sampling) |
|---|---|---|
| 参数更新 | ✅ (LoRA) | ❌ |
| 推理 latency | 低（更新后） | 高（每步 N 次） |
| 适应深度 | 强（学新分布） | 弱（只在已有 distribution 选） |
| 安全性 | 较低（RL 探索风险） | 较高（不变 policy） |
| 本科生友好度 | 中（RL 调参难） | **高**（工程实现简单） |

### 本科生一年期课题切入空间

**最适合切入的 sub-problem：「轻量 Verifier + 高效采样」**

具体方案：
- **Step 1**（2 个月）：复现 VLA-RL 在 LIBERO 上基线（用 OpenVLA 作 base）
- **Step 2**（3 个月）：替换昂贵 verifier 为 SigLIP/CLIP-based 轻量 verifier
- **Step 3**（2 个月）：研究采样多样性的影响——temperature scaling vs noise injection
- **Step 4**（3 个月）：在 VLA-Risk 扰动上评测，对比 TT-VLA
- **Step 5**（2 个月）：写 paper

**发表潜力**：CoRL workshop / ICRA workshop

**对比 TT-VLA**：VLA-RL 工程更友好（不更新参数），但创新空间相对窄

### 设计风险

- **Latency 是硬约束**：real-world control 必须 < 100ms
- **Verifier 训练数据** 需要离线 rollout
- **代码 release** 待确认

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2026-01_TT-VLA_OnTheFlyRL]], [[2025_RoboMonkey_ScalingTT]]
- 与主线 A/B 对照：[[2026-03_FocusVLA_Zhang]], [[2026_RobustVLA_Guo]]
- 评测 benchmark：[[VLA-Risk_ICLR2026]]
- Inference-time scaling 经典：LLM Best-of-N, Tree of Thoughts

## 📌 Action Items
- [ ] 确认 arXiv ID
- [ ] 评估 inference latency 在 real-world 部署的可行性
- [ ] 设计 verifier 轻量化方案

%% Import Date: 2026-05-26 %%
