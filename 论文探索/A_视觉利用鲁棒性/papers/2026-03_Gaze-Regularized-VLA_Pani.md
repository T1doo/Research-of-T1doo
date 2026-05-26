---
title: "Gaze-Regularized Vision-Language-Action Models for Robotic Manipulation"
authors: "Anupam Pani, Yanchao Yang"
year: "2026"
venue: "arXiv 2603"
arxiv: "2603.23202"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, attention-guidance, gaze]
---

# Gaze-Regularized VLA

> [!info] 元信息
> - **作者**：Anupam Pani, Yanchao Yang
> - **arXiv**：[2603.23202](https://arxiv.org/abs/2603.23202)

## 📄 TL;DR

把**人眼 gaze**当作 VLA 的 attention 监督信号。不改架构，不加 eye-tracker——直接把已有数据集中的 gaze heatmap 投影到 patch-level 分布，与 transformer 内部 attention 用 KL 散度对齐。实现 task-relevant feature bias 同时保持部署效率。多个 manipulation benchmark 上 **4-12% 提升**，更少训练步达到同等性能，对光照变化和传感器噪声鲁棒，attention 可视化与人类策略一致。

## 🧠 我的思考

### 核心观点

1. **人类 gaze 是"免费"的 attention oracle**：人类执行同样操作时眼睛看哪里，蕴含了 task-relevant region 的天然标注。这种监督比 saliency / segmentation mask 更精准——它捕捉的是「执行时刻」的注意力，与动作时序对齐。
2. **KL regularization 而非 hard mask**：用 KL 让模型 attention 接近 gaze 分布，但不强制完全匹配——保留模型自由学到更细的差异（如机器人 vs 人手视角差异），既约束又灵活。
3. **不改架构、不加硬件、可复用既有数据**：这三条让方法的工程门槛极低，是 attention guidance 路线中 **adoption cost 最低**的方案。

### 方法论关键

- **Gaze heatmap → patch distribution**：把 gaze 的连续 2D 热图按 patch grid 离散化、归一化为 categorical distribution $p_{gaze}$。
- **Attention extraction**：从 transformer 某层（或多层平均）提取 attention map，token-wise 求和或取 attention 头平均得到 patch-level attention distribution $p_{attn}$。
- **Loss**：$\mathcal{L}_{total} = \mathcal{L}_{BC} + \lambda \cdot D_{KL}(p_{gaze} || p_{attn})$。
- **关键设计选择**：从哪一层取 attention？多头如何聚合？这些 ablation 决定方法是否 work。

### 关键实验数据

| Benchmark | Baseline | + Gaze Reg | Δ |
|---|---|---|---|
| Manipulation tasks (avg) | base | **+4 ~ +12%** | ↑ |
| 光照扰动 / sensor noise | base | 显著鲁棒 | ↑ |
| 训练步数达到同性能 | full | fewer | speedup |

具体表格需读正文，abstract 给的是范围。

### 与曦源（FocusVLA 主线）的关联

Gaze-Regularized VLA 与 FocusVLA 是**两种 attention guidance 哲学**的对照：
- FocusVLA：**self-induced** attention focusing，靠架构（top-K + gate）让模型自己学聚焦。
- Gaze-Regularized：**externally supervised** attention focusing，靠人类 gaze 当老师。

两者**强互补**：
1. **FocusVLA + gaze regularization 组合**：可以在 FocusVLA 的 Focus Attention 输出上加一个 KL gaze loss，让 patch-level top-K 选择不仅自洽，还匹配人类先验——理论上能显著提升对 distractor 的抑制能力。
2. **Gaze 数据可作为 evaluation oracle**：即便不用 gaze 监督训练，也可以用 gaze 数据评估 FocusVLA 学到的 attention 与人类聚焦区域的对齐度，做 interpretability 量化。
3. **解决 FocusVLA 留的空白**：FocusVLA 自述「鲁棒性对背景+纹理同时变化仍有限」，gaze regularization 直接强化对 task-relevant 区域的偏置，可能是补救手段。

**研究空白**：Gaze-Regularized VLA 没在 long-horizon 任务上深度验证，gaze 是按瞬时帧标注的，长序列任务中 gaze 转移如何对齐 attention 转移？这是值得研究的 temporal alignment 问题。

### 待解问题

- 哪一层 transformer attention 与 gaze 对齐效果最好？
- 没有 gaze 数据的任务上，能否用 [[2025-11_AutoFocus-IL_Gong]] 的 VLM saliency 替代？
- KL 系数 λ 的选择是否敏感？过强会让 attention 过拟合到 gaze 而损失 BC 性能？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2025-11_AutoFocus-IL_Gong]]
- [[2025-09_AttentionVoxel_Yurchyk]]
- [[2026-05_GuidedVLA_Jia]]
