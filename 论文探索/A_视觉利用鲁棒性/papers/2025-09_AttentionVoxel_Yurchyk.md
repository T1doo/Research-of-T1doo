---
title: "Large Pre-Trained Models for Bimanual Manipulation in 3D"
authors: "Hanna Yurchyk, Wei-Di Chang, Gregory Dudek, David Meger"
year: "2025"
venue: "Humanoids 2025 / arXiv 2509"
arxiv: "2509.20579"
status: "已精读"
tier: "⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, attention-guidance, voxel, bimanual, DINOv2]
---

# Attention-Guided Voxel Bimanual

> [!info] 元信息
> - **作者**：Hanna Yurchyk, Wei-Di Chang, Gregory Dudek, David Meger
> - **机构**：McGill University 等
> - **arXiv**：[2509.20579](https://arxiv.org/abs/2509.20579)
> - **会议**：IEEE Humanoids 2025

## 📄 TL;DR

用 **DINOv2 的自监督 attention map** 作为 RGB 像素级 saliency，**lift 到 3D voxel grid**，作为语义特征注入到双臂操作的 voxel-based policy 中。在 RLBench bimanual benchmark 上平均**绝对提升 8.2%、相对提升 21.9%**。核心思想：DINOv2 attention 已经隐式编码了"object salience"，无需额外训练即可用作 voxel-level semantic cue。

## 🧠 我的思考

### 核心观点

1. **DINOv2 attention = 免费的 voxel saliency oracle**：DINOv2 自监督训练得到的 attention map 与物体边界 / 显著区域高度对齐（DINOv2 paper 已展示），这是非常 robust 的 zero-shot saliency 源——比 SAM 轻、比手工标注便宜。
2. **2D attention → 3D voxel lifting 是关键创新**：双臂操作（bimanual）天然是 3D 任务，2D saliency 无法直接服务。通过相机内外参 + 深度图把 2D attention back-project 到 voxel，每个 voxel 得到 attention 加权特征。这把现成的 2D 模型成果引入到 3D policy。
3. **不动 policy 主体，只换 feature**：是个 "drop-in" 工作——把 voxel feature 从普通 RGB encoding 换成 DINOv2-attention-augmented voxel encoding，policy 架构不变。adoption cost 极低。

### 方法论关键

- **DINOv2 attention extraction**：取 CLS token 对 patch token 的 attention，softmax 归一化得到每个 patch 的 saliency score。
- **2D → 3D lifting**：对每个 voxel $v_{xyz}$，找其投影到所有 camera 视图的 patch，加权聚合 saliency。$s(v) = \sum_c w_c \cdot s_c(\pi_c(v))$，$\pi_c$ 是 camera c 的投影函数。
- **Voxel feature**：原 RGB feature × attention saliency，或拼接为额外通道。
- **Policy backbone**：未改，沿用 SOTA voxel-based policy（如 PerAct / Act3D 风格）。

### 关键实验数据

| Setting | RLBench Bimanual |
|---|---|
| Baseline voxel policy | base |
| **+ DINOv2 attention voxel** | **+8.2% absolute, +21.9% relative** |

任务覆盖多个双臂操作，跨任务统计稳定提升。

### 与曦源（FocusVLA 主线）的关联

这篇工作与 FocusVLA 在**信息域**上不同（3D voxel vs 2D patch）但在**核心思想**上一致——用 self-attention 自动识别 task-relevant region。与曦源关联：

1. **DINOv2 attention 作为 FocusVLA 的额外信号**：FocusVLA 现已使用 DINOv2 + SigLIP + VGGT 多 encoder，但是用作 feature。可以额外用 DINOv2 的 CLS attention 作为 patch importance prior，注入到 Focus Attention 的 top-K 选择中（hybrid attention scoring）。
2. **3D voxel 扩展是 FocusVLA 的潜在方向**：FocusVLA 只在 2D image 域做 token 选择，对于真正的 3D 操作（拼装、装配）这种 2D-only 表达可能不够。可以借鉴本工作的 lifting 思路，让 FocusVLA 在 voxel 域做 attention，与 SpatialVLA / ST-VLA 形成更直接对比。
3. **Bimanual 任务的视觉鲁棒性是个被忽视的子方向**：曦源主要在单臂 LIBERO 评测，本工作展示双臂任务对视觉表征要求更高（多视角融合、左右臂遮挡）。可以扩展评测到 RoboTwin 双臂 benchmark，看 FocusVLA 是否需要 lifting-style 增强。

**研究空白**：本工作的 attention 是 **DINOv2 自监督**得到的，未引入任务指令。同一物体在不同指令下 saliency 应该不同（拿杯子 vs 不拿杯子），需要 instruction-conditioned voxel saliency——可结合 [[2025-11_AutoFocus-IL_Gong]] 思路改进。

### 待解问题

- 多相机视图的 attention 融合权重 $w_c$ 怎么定？固定还是 learned？
- 在 single-arm LIBERO 上是否仍 work？
- 与 [[2026-05_SpatialVLA_Qu]] / [[2026-03_ST-VLA_Wu]] 的 3D representation 路线如何对比？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2026-03_Gaze-Regularized-VLA_Pani]]
- [[2025-11_AutoFocus-IL_Gong]]
- [[2026-05_SpatialVLA_Qu]]
- [[2026-03_ST-VLA_Wu]]
