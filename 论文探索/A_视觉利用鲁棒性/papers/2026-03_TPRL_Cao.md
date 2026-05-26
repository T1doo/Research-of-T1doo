---
title: "Language-Guided Token Compression with Reinforcement Learning in Large Vision-Language Models"
authors: "Sihan Cao, Jianwei Zhang, Pengcheng Zheng, Jiaxin Yan, Caiyan Qin, Yalan Ye, Wei Dong, Peng Wang, Yang Yang, Chaoning Zhang"
year: "2026"
venue: "arXiv 2603"
arxiv: "2603.13394"
status: "已精读"
tier: "⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, token-pruning, RL]
---

# TPRL (Language-Guided Token Compression with RL)

> [!info] 元信息
> - **作者**：Sihan Cao, Jianwei Zhang, Pengcheng Zheng, Jiaxin Yan, Caiyan Qin, Yalan Ye, Wei Dong, Peng Wang, Yang Yang, Chaoning Zhang
> - **arXiv**：[2603.13394](https://arxiv.org/abs/2603.13394)
> - **代码**：GitHub 公开

## 📄 TL;DR

TPRL 把 VLM/VLA 的视觉 token 剪枝形式化为 **sequential decision-making problem**，用 RL 学剪枝轨迹。框架三阶段：(1) 自监督 autoencoder 把视觉 token 压缩成紧凑状态表征，(2) 从 demonstration 初始化 policy（imitation pretraining），(3) PPO 微调平衡 accuracy 与 efficiency。剪掉 66.7% 视觉 token、FLOPs 降 54.2%，平均精度仅掉 0.7%（near-lossless）。

## 🧠 我的思考

### 核心观点

1. **Pruning as sequential decision making 是范式跃迁**：传统剪枝都是 one-shot scoring + top-K，缺乏全局规划。RL 把剪枝看作"按 token 顺序决定保留与否"的 MDP，policy 可以根据已保留 token 集合的状态动态调整下一步决策——这种 stateful 选择能避免冗余（如已保留中心物体后，不必再保留同区域 token）。
2. **Language-guided reward 是关键**：仅以 task accuracy 为 reward 会让 RL 倾向保留过多 token（reward 难微分），TPRL 通过 language 信号引导每步决策，相当于在剪枝过程中引入指令对齐先验。
3. **PPO + 模仿初始化的工程稳定性**：纯 RL 在剪枝这种高维离散动作空间很难收敛，先用 demonstration（可能是某 baseline 剪枝方法的输出）初始化 policy 是必要的工程稳定化手段。

### 方法论关键

- **State**：当前已保留的 visual token 集合的 latent encoding，来自自监督 autoencoder。
- **Action**：保留 / 丢弃下一个 token（或一组 token）。
- **Reward**：task performance + token budget penalty + language alignment 项。
- **Policy network**：可能是小型 transformer 或 MLP，take state + candidate token feature → action prob。
- **训练 pipeline**：autoencoder 预训练 → BC 初始化 policy → PPO fine-tune。

### 关键实验数据

| Setting | Token Removed | FLOPs Reduction | Accuracy Drop |
|---|---|---|---|
| Baseline | 0% | 0% | 0 |
| **TPRL** | **66.7%** | **54.2%** | **0.7%** |

未在 VLA 专属 benchmark（LIBERO 等）上验证，主要是 general VLM 任务。

### 与曦源（FocusVLA 主线）的关联

TPRL 与 FocusVLA 在「token 选择哲学」上是**反向互补**的：
- FocusVLA：硬约束架构（Focus Attention 模块），top-K 不可学但可微梯度回传影响 attention 学习。
- TPRL：把 token 选择本身做成 learnable policy，可显式优化"剪哪些"。

**潜在融合**：FocusVLA 的 top-K 选择目前是 attention-based scoring + topK，可以把这个 scoring 替换成 TPRL-style RL policy——但工程复杂度大幅上升。更轻量的方式是用 TPRL 思路做 post-hoc analysis：训练 FocusVLA 后，用 RL policy 评估其 attention 选择是否最优，发现可改进的 token 模式。

**对 VLA 的适配挑战**：TPRL 训练是 VQA-style 任务（reward 明确），VLA 任务的 reward 是 trajectory 成功率，sparse 且 delayed。把 TPRL 移植到 VLA 需要 dense reward shaping（如每步 action-token 对齐度），工程量极大。

**研究价值**：可以借鉴 TPRL 的「pruning as MDP」框架，但用更轻量的离线训练替代 PPO（如 DPO-style preference learning），让 token 选择 learnable 但不至于训练崩塌。

### 待解问题

- TPRL 在 VLA / 机器人任务上是否能直接迁移？
- 与简单 attention-based top-K 的 wall-clock 训练成本对比？
- Action space 设计：one-token-at-a-time 太慢，group-level 决策如何设计？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2025-11_VLA-Pruner_Liu]]
- [[2025-11_Compressor-VLA_Gao]]
- [[2025-12_LUVC_Zheng]]
