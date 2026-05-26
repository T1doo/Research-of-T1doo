---
title: "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model"
authors: "Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, Xuelong Li"
year: "2025"
venue: "RSS 2025"
arxiv: "2501.15830"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, patch-selection, 3D, spatial-aware, foundation-model]
---

# SpatialVLA

> [!info] 元信息
> - **作者**：Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye 等
> - **机构**：上海人工智能实验室、东方理工等
> - **arXiv**：[2501.15830](https://arxiv.org/abs/2501.15830)
> - **会议**：RSS 2025

## 📄 TL;DR

SpatialVLA 主张"**spatial understanding 是机器人操作的关键**"。提出两大组件：(1) **Ego3D Position Encoding** 把 3D 空间信息（深度、相机外参）注入 vision-language input；(2) **Adaptive Action Grids** 用离散化空间网格表征机器人动作，使知识可跨不同机器人系统迁移。在 1.1M 真实机器人 episode 上预训练，多任务多环境强 zero-shot 性能，分布内分布外都泛化良好；适配新机器人通过重新离散化 grid 即可 fine-tune。

## 🧠 我的思考

### 核心观点

1. **2D VLA 缺少"机器人是空间智能体"的归纳偏置**：当前主流 VLA（OpenVLA、π0）都把图像当作 2D pixel grid 处理，对 3D 空间关系靠 implicit learning。SpatialVLA 用 Ego3D PE 显式注入空间先验，类似 NeRF / 3D detection 的做法引入到 VLA。
2. **Adaptive Action Grids 是跨 embodiment 迁移的工程钥匙**：不同机器人（UR5 vs Franka vs xArm）的动作空间不同（关节角 / 末端 pose / 速度），让 policy 在统一的空间 grid 上输出动作再 decode 到具体 embodiment——这是 1.1M episode 跨多机器人预训练能成立的基础。
3. **1.1M episode 预训练 + 强 zero-shot 是大力出奇迹**：与 OpenVLA / RT-2 风格类似，数据量碾压。但 SpatialVLA 强调空间表征质量是其优势——同样数据量下，3D-aware 表征应该更易泛化。

### 方法论关键

- **Ego3D Position Encoding**：每个 patch 不仅有 2D position emb，还附加 3D world-coord（从深度+相机外参反投影）。可能与 patch feature concat 或 add。
- **Adaptive Action Grid**：把工作空间离散化为 grid cells（如 100×100×100），机器人动作映射到 cell index + offset。policy 输出是 cell logit + offset regression。
- **预训练 corpus**：1.1M episode 来自 OXE / DROID / 等开源数据集汇总。
- **Fine-tune protocol**：新机器人只需用其工作空间重新离散化 grid，policy head 微调。

### 关键实验数据

| Setting | Performance |
|---|---|
| Zero-shot multi-task | strong |
| Cross-embodiment fine-tune | effective via re-discretization |
| Distribution shift | robust generalization |

abstract 强调"exceptional generalization"，具体数字需对照 RT-2 / OpenVLA 在同 benchmark。

### 与曦源（FocusVLA 主线）的关联

SpatialVLA 与 FocusVLA 是**互补维度**——一个解决"看什么 patch"（2D 选择），一个解决"patch 在哪儿"（3D 坐标）。组合潜力：

1. **3D 感知 + Focus Attention 的天然组合**：FocusVLA 现在的 patch-level top-K 是 2D 平面操作，把 SpatialVLA 的 Ego3D PE 加入后，top-K 可以基于"3D 空间距离 + task relevance"联合评分——对接触敏感任务（peg-in-hole）应该有显著提升。
2. **Adaptive Action Grids 为曦源的跨 embodiment 评测开门**：FocusVLA 目前只在 LIBERO 单一 simulator + xArm 真机评测，如果借鉴 SpatialVLA 的 grid 抽象，可以扩展到 RoboTwin / OXE 数据上做更广泛的 robustness benchmark。
3. **SpatialVLA 是导师推「视觉利用」路线之外的另一条主线（空间利用）的标志性工作**：可以作为「VLA 鲁棒性多维度评测」中的 spatial 维度 baseline。

**潜在弱点**：SpatialVLA 依赖深度图——在缺少深度传感器的场景（pure RGB camera）下退化为 monocular depth estimation，引入新的不确定性源。这种"3D prior 引入新失败模式"是值得做的实证研究。

**研究空白**：SpatialVLA 没在 visual perturbation（光照、纹理）下深度评测——3D PE 是否对 2D 视觉扰动鲁棒？或是反而更脆弱（depth estimation 失效会污染 PE）？

### 待解问题

- Ego3D PE 在 monocular setting 下如何处理（用 MiDaS / Depth Anything 估 depth）？
- Adaptive Action Grid 的 cell 大小如何选？是否需要任务自适应？
- 与 [[2026-03_ST-VLA_Wu]] 的 4D spatiotemporal mask 路线对比？
- 与 FocusVLA 的 patch 选择融合是否实证可行？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2026-03_ST-VLA_Wu]]
- [[2024-12_TraceVLA_Zheng]]
- [[2025-09_AttentionVoxel_Yurchyk]]
- [[2026-05_GuidedVLA_Jia]]
