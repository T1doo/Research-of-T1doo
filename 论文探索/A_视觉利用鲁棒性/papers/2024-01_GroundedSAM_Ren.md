---
title: "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks"
authors: "Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, Lei Zhang"
year: "2024"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2401.14159"
arxiv: "2401.14159"
venue: "arXiv 2024-01-25 (IDEA Research technical report)"
citekey: "renGroundedSAM2024"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, 视觉grounding, segmentation, open-set, 工具基础, A4]
---

# Grounded SAM 精读笔记

> [!info] 元信息
> - **作者**：Tianhe Ren et al.（IDEA Research，17 人团队）
> - **机构**：IDEA Research / HKUST / Tsinghua / SJTU
> - **日期**：2024-01-25 (arXiv)
> - **arXiv**：[2401.14159](https://arxiv.org/abs/2401.14159)
> - **代码**：github.com/IDEA-Research/Grounded-Segment-Anything (15k+ stars)
> - **定位**：A4 grounding 的 *工具基础设施* — open-vocabulary detect + segment 的标准管线

## 📄 TL;DR

Grounded SAM = **GroundingDINO（开放集检测器） + SAM（任意分割模型）**。给定文本提示，GroundingDINO 输出 bounding box，SAM 在 box 内生成精确 mask。模块化设计，可以再串联 BLIP（caption）、Stable Diffusion（编辑）、OSX（3D 人体）。SegInW zero-shot 上 48.7 mean AP。本质是「**开放词汇 grounding 的事实标准管线**」，是后续 ReKep / VoxPoser / 大量 VLA 工作的 *上游视觉工具*。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「Grounding 工具的工程标准化」是 2024 年视觉机器人最大的基础设施收益**。在 GroundingDINO + SAM 出现前，"找到桌子上的红杯子并分割"需要专用训练；现在一行 prompt 解决。这把"上游视觉"从研究问题变成 *已解决问题*，让下游可以集中精力做 policy。
2. **模块化 = 可替换性**：每个组件独立 SOTA，可以根据场景换组件。比如 SAM2（视频版）出来后，Grounded SAM2 立刻可用。这种解耦设计是 *foundation model 时代的工程范式*。
3. **作为 baseline / 工具，而非研究方法**：本文不是新方法论，是把已有 model 工程化组合，意义在生态。

### 方法论

- **GroundingDINO**：BERT 文本编码 + DINO 视觉编码 + cross-attention 融合 → 文本 query 的 detection
- **SAM**：promptable segmentation（box / point / mask 作为 prompt）
- **管线**：text → GroundingDINO box → SAM mask；可继续输入 BLIP 做 caption / SD 做 inpainting
- **零样本**：SegInW 48.7 mean AP，无需 task-specific 微调

### 与曦源关联（FocusVLA / RobustVLA）

- **与 FocusVLA 互补**：FocusVLA 学 *attention 应聚焦哪*；Grounded SAM 直接给 *像素级 grounding 答案*。可以把 SAM mask 当作 attention 的 *弱监督 / 评估锚点*：「FocusVLA attention 与 SAM mask 的 IoU」是个直接的 grounding 质量指标。
- **与 RobustVLA 组合**：扰动训练时，SAM mask 可以作为 *几何不变量*（扰动不改变物体的物理形状 → mask 应不变），加 mask consistency loss。
- **暴露 FocusVLA 未涉及的子问题**：FocusVLA 没回答"如果 attention 与 SAM mask 高度重合，是否就鲁棒"。这是一个可量化的研究问题。
- **本科生子模块可行性**：⭐⭐⭐⭐ — Grounded SAM 是工程级工具，开箱即用；可作为 *评估管线* 嵌入所有视觉 grounding 实验。**强烈建议作为标配工具**。

### 待解问题

1. **零样本极限**：长尾物体（特殊工业件、医疗器械）的检测精度如何？
2. **mask 边界对接触操作的精度**：操作任务对 mask 精度要求高（mm 级），48.7 mean AP 不直接说明边界质量。
3. **实时性**：GroundingDINO + SAM 一帧约 0.5-1s，难做闭环；蒸馏版（如 FastSAM）牺牲精度换速度。

%% end my-thoughts %%

## 🔗 关联笔记

- **下游应用**：[[2024-09_ReKep_Huang]]（用 SAM 做 keypoint 区域）、[[2023-07_VoxPoser_Huang]]（用 OWL-ViT，可替换为 GroundedSAM）
- **互补 grounding 方法**：[[2024-02_DINOBot_DiPalo]]（dense feature retrieval）
- **VLA attention 评估**：[[2026-03_FocusVLA_Zhang]] 的 attention IoU 可对比 Grounded SAM mask
- **诊断侧**：[[2025-08_ShortcutLearning_Xing]]
- **主线 A 总结**：[[A4_visual_grounding_总结]]

## 📌 Action Items

- [ ] 工具化：在 `_utils/` 目录下封装 Grounded SAM 管线，作为所有实验的 grounding 评估器
- [ ] 评估：FocusVLA attention heatmap vs Grounded SAM mask 的 IoU 在 LIBERO 上量化
- [ ] 探索：用 Grounded SAM mask 作为弱监督 fine-tune FocusVLA attention

%% Import Date: 2026-05-26 %%
