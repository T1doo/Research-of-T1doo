---
title: "InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning"
authors: "Ji Zhang, Shihan Wu, Xu Luo, Hao Wu, Lianli Gao, Heng Tao Shen, Jingkuan Song"
year: "2025"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2505.13888"
arxiv: "2505.13888"
venue: "arXiv 2025-05-20 (rev 2025-09-29)"
citekey: "zhangInSpire2025"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, shortcut learning, spatial reasoning, plugin, A6]
---

# InSpire — Intrinsic Spatial Reasoning VLA 精读笔记

> [!info] 元信息
> - **作者**：Ji Zhang, Shihan Wu, Xu Luo, Hao Wu, Lianli Gao, Heng Tao Shen, Jingkuan Song
> - **机构**：UESTC（与 ShortcutLearning 同实验室）
> - **日期**：2025-05-20 (arXiv, rev 09-29)
> - **arXiv**：[2505.13888](https://arxiv.org/abs/2505.13888)
> - **定位**：A6 shortcut 抑制 — 无需新数据的 *plugin*，给 autoregressive VLA 注入 spatial reasoning

## 📄 TL;DR

InSpire 解决"VLA 把 task-irrelevant 视觉特征与 action 错误绑定"的 spurious correlation 问题。**核心机制**：作为 autoregressive VLA 的 plugin，**无需额外训练数据**，只在 prompt 前面加方向性问题（"In which direction is the [object] relative to the robot?"），并将 VLA 预测的方向答案（"right/left/up/down/front/back/grasped"）与 ground-truth action 对齐。这相当于强制 VLA *显式输出空间关系再输出 action*。在 sim + real 上验证泛化提升。本质是「**Chain-of-thought 提示 + 方向标签 align = 廉价的 spatial grounding 强化**」。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「Plugin、零数据成本」让 InSpire 极有工程吸引力**：不改架构、不加数据、只改 prompt + 加一个分类 loss → 可以无缝叠到 OpenVLA / SmolVLA / π₀ 上。这是曦源 "工程友好" 路线的典范。
2. **「Prompt 强制 CoT = 训练时显式 grounding」**：让模型必须先输出空间关系再输出 action，把 *隐式 grounding* 强制 *显式化*，类似 CoT 在 LLM 上的成功机制。
3. **方向词作为离散监督**：6 个方向词（+grasped）是个极简的 spatial vocabulary，但已足够约束 action 方向。这种 *离散符号 grounding* 是 ReKep / VoxPoser 的简化版。

### 方法论

- **Prompt 改造**：原任务前置 "In which direction is the [target] relative to the robot?"
- **预测对齐**：VLA 在 action 前先生成方向词 ∈ {right, left, up, down, front, back, grasped}
- **损失**：方向词 cross-entropy + action regression 联合
- **训练**：用原 demo 数据自动从 robot pose 与 object pose 推导 GT 方向词，无需新标注
- **plugin 性质**：不改 backbone，只加 head + loss

### 与曦源关联（FocusVLA / RobustVLA）

- **与 FocusVLA 方法学完全互补**：FocusVLA 在 attention 层强制看图；InSpire 在 output 层强制 reason。两者完全正交 — 可以无成本叠加（FocusVLA + InSpire prompt）。
- **与 RobustVLA 组合**：扰动训练时 *方向词应不变*（光照变了但物体相对位置不变）→ 可作为 RobustVLA 的 *额外 consistency 标签*。
- **暴露 FocusVLA 未涉及的子问题**：FocusVLA 不评估 *推理过程*；InSpire 给了一个把 spatial reasoning 显式化的工具，可以验证 *FocusVLA attention 是否对应正确的 spatial reasoning*。
- **本科生子模块可行性**：⭐⭐⭐⭐⭐ — 改 prompt + 加 head，最低工程成本，最高研究弹性。**强烈推荐作为第一个本科生实验项目**。

### 待解问题

1. **方向词的 GT 自动推导**：从 robot pose 和 object pose 推导方向词的准确度？是否对 pose 噪声敏感？
2. **能否扩展到更细粒度 spatial vocabulary**：6 方向太粗，"前左方 30°" 等是否需要？
3. **与端到端 VLA 的 CoT 干扰**：先输出方向词增加 latency，能否平衡？

%% end my-thoughts %%

## 🔗 关联笔记

- **同实验室同主题**：[[2025-08_ShortcutLearning_Xing]]（同 Lianli Gao / J. Song 团队）
- **互补的方法学路径**：[[2026-03_FocusVLA_Zhang]]（attention）、[[2026-03_SG-VLA_Tu]]（auxiliary）
- **CoT 增强 VLA 思路**：[[2025-12_CF-VLA_Peng]]
- **数据侧组合**：[[2025-10_RobustVLA]]
- **主线 A 总结**：[[A6_shortcut_learning_总结]]

## 📌 Action Items

- [ ] **首要实验**：FocusVLA + InSpire prompt → LIBERO 上 OOD 鲁棒性增益（4-6 周可完成）
- [ ] 评估：方向词预测准确率与 action 成功率的相关性
- [ ] 思考：能否把方向词作为 FocusVLA attention pattern 的弱监督（"看右上方"应有右上 attention）

%% Import Date: 2026-05-26 %%
