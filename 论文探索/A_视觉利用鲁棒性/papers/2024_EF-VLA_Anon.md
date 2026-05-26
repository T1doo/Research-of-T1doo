---
title: "EF-VLA: Early Fusion CLIP with Softmax Patch Selection for Vision-Language-Action"
authors: "Anon (待手动补充)"
year: "2024"
venue: "OpenReview / 待定"
arxiv: "待手动补充"
status: "已精读 (占位)"
tier: "⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, patch-selection, early-fusion, 待补充]
---

# EF-VLA

> [!warning] 需手动补充
> 此论文的 arxiv ID / OpenReview ID 在自动抓取中未能定位。
> - OpenReview URL `https://openreview.net/forum?id=EF-VLA` 返回 404
> - 需要进一步核实正确的 paper ID 或 venue（可能为 ICLR / CoRL / NeurIPS submission name）
> - 建议手动搜索：`EF-VLA early fusion CLIP softmax patch selection vision language action`

> [!info] 元信息
> - **作者**：（待补充）
> - **机构**：（待补充）
> - **链接**：（待补充）

## 📄 TL;DR（基于任务描述推测）

EF-VLA 主张在 VLA 的 **vision-language fusion** 阶段做 early fusion——不是分别 encode 视觉与语言再拼接，而是让 CLIP-like 模型在 patch 级别就把语言信号注入视觉特征。配合 **softmax patch selection**（可微的 soft top-K 替代 hard top-K），实现 task-relevant patch 的连续可学习选择。

## 🧠 我的思考（待论文正文确认后补充细节）

### 核心观点（推测）

1. **Early fusion vs late fusion 的范式选择**：late fusion（先 encode 各模态再 cross-attention）是 OpenVLA / VLA-Adapter 主流。Early fusion 让语言早期参与视觉编码，理论上能让视觉特征本身就 task-aware，避免后期 cross-attention 的对齐误差。
2. **Softmax patch selection 的可微性**：相比 hard top-K（不可微），softmax 选择保留了所有 patch 的连续权重，梯度回传更自然——可能比 FocusVLA 的 hard top-K + softmax 包装更简洁。
3. **CLIP-based fusion 的迁移性**：用 CLIP 作 backbone 意味着可以利用 image-text 对齐预训练带来的 task-relevant 偏置，是个 effective inductive bias。

### 方法论关键（待补）

- Early fusion 具体实现：cross-attention from text → vision 在哪一层？
- Softmax patch selection 的 temperature 如何调？
- 与 FocusVLA Focus Attention 的形式化对比？

### 与曦源（FocusVLA 主线）的关联

EF-VLA 是 FocusVLA 在 abstract / related work 中提及的**同代 patch selection 工作**。两者关系：
- **共同点**：都做 patch-level 重要度选择，都强调 task-relevance。
- **差异**：FocusVLA 在 **action head** 内部做 selection（latent space），EF-VLA 在 **vision encoder** 处做 early fusion（feature space）。
- **可对比研究**：在同一 backbone + LIBERO 上比较"early-fusion soft selection"vs"late-fusion hard top-K"，看哪种 patch 选择范式更鲁棒。

**研究空白**：EF-VLA 的 softmax selection 是 soft 的，是否真的"选择"还是"加权平均"？hard top-K 才真正减少 token 数（加速推理），soft selection 仍要全 forward——这是 efficiency vs differentiability 的 trade-off，值得在曦源框架下系统比较。

### 待解问题

- 找到正确 paper ID 后，补充 abstract、method 详细、benchmark 数据
- 是否做过 LIBERO / RoboTwin 等标准 benchmark？
- 代码开源情况？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2026-05_GuidedVLA_Jia]]
- [[2026-05_SpatialVLA_Qu]]
- [[2024-12_TraceVLA_Zheng]]

## 📌 Action
- [ ] **手动检索补充正确 paper 链接与详细内容**
- [ ] 确认是否为 ICLR 2024 / CoRL 2024 submission
