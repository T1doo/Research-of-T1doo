---
title: "Action Images: End-to-End Policy Learning via Multiview Video Generation"
authors: "Haoyu Zhen, Zixian Gao, Qiao Sun, Yilin Zhao, Yuncong Yang, Yilun Du, Pengsheng Guo, Tsun-Hsuan Wang, Yi-Ling Qiao, Chuang Gan"
year: "2026"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2604.06168"
arxiv: "2604.06168"
venue: "arXiv 2026-04-07 (rev 2026-04-15)"
citekey: "zhenActionImages2026"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, 多视角融合, video generation, pixel-grounded, A5]
---

# Action Images — Multiview Video Generation 精读笔记

> [!info] 元信息
> - **作者**：Haoyu Zhen et al.（MIT / UMass Amherst / Apple，C. Gan / Y. Du 团队）
> - **机构**：UMass Amherst CSAIL、MIT、Apple
> - **日期**：2026-04-07 (arXiv, rev 04-15)
> - **arXiv**：[2604.06168](https://arxiv.org/abs/2604.06168)
> - **定位**：A5 多视角融合 — 把 7-DoF action 编码成 *多视角 pixel-grounded action images*，video backbone 直接当 policy

## 📄 TL;DR

Action Images 提出统一的 **world-action model**：把 7-DoF action 转换为 *interpretable action images* — 多视角动作视频，明确把机器人臂运动 *pixel-grounded* 标在 2D 像素空间。这种像素级表示让 video backbone 本身 *无需独立 policy module* 就能作为零样本 policy；同时支持多种下游任务（action-conditioned 生成、action labeling）共享同一表示。RLBench 与真机操作上 zero-shot 表现超越现有 video-space world model。本质是「**Action 不再是低维向量，而是 2D 像素轨迹图像**」。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「Action 的 modality 不必是 vector」是个范式级洞察**：把 7-DoF 数值转成 *机器人臂在多视角像素上的轨迹图*，能直接被 video transformer / diffusion 处理。这彻底重新定义了 action representation 空间。
2. **「Pixel-grounded = 几何与视觉对齐的天然保证」**：动作图就是动作在像素空间的真实投影，与场景视觉天然对齐；不存在 "动作向量 vs 视觉特征" 的 modality gap。
3. **多视角是为了 3D 反推唯一**：和 MV-VDP 同样的思路 — 单视角 2D 像素轨迹歧义大，多视角联合可反推 3D 动作。

### 方法论

- **Action → Image 编码**：7-DoF action sequence → 多视角下机器人末端的 *像素轨迹 mask + RGB 渲染*
- **统一表示**：video backbone 输入 (current obs + task) → 输出 action image 视频
- **解码**：从 action image 提取像素轨迹 → 反投影到 3D action
- **下游兼容**：action labeling、action-conditioned video generation 共享同一表示

### 与曦源关联（FocusVLA / RobustVLA）

- **与 FocusVLA 方法学对照**：FocusVLA 改 attention（输入空间）；Action Images 改 *action 表示*（输出空间）。两条路改动维度不同 — 可以叠加：FocusVLA attention 改造 + action image 输出。
- **与 RobustVLA 组合**：扰动后 *物理动作不变 → action image 应不变*，可直接做 image-level consistency loss（比向量级更直观，可视化 friendly）。
- **暴露 FocusVLA 未涉及的子问题**：FocusVLA 假设 action 输出空间不变；Action Images 挑战这个假设 — 改变 action representation 本身能否提升鲁棒性？这是个全新维度。
- **本科生子模块可行性**：⭐⭐ — action image 渲染管线复杂，需要相机参数 / 机器人 forward kinematics；但作为 baseline 评估和理论思考素材很有价值。

### 待解问题

1. **Pixel 精度对动作精度的瓶颈**：1 pixel 对应几 mm？长程任务的累积误差？
2. **action image 的反投影成功率**：遮挡 / 视角失败时怎么办？
3. **大数据预训练依赖**：需要 video pretrain？还是机器人 demo 就够？

%% end my-thoughts %%

## 🔗 关联笔记

- **同源 video 范式**：[[2026-04_MV-VDP_Li]]（heatmap 路线，互为对照）
- **3D 表示竞品**：[[2025-09_GP3_Qian]]、[[2025-12_VIPA-VLA_Feng]]
- **互补的 attention 路径**：[[2026-03_FocusVLA_Zhang]]
- **像素 grounding 思想**：[[2025_PointVLA_Anon]]、[[2024-01_GroundedSAM_Ren]]
- **主线 A 总结**：[[A5_多视角融合_总结]]

## 📌 Action Items

- [ ] 思考：能否在 FocusVLA 上加 action image 辅助输出 head（不改主架构）
- [ ] 调研：action image 渲染所需的相机 / 运动学管线，评估复现门槛
- [ ] 对照：MV-VDP heatmap vs Action Images pixel trajectory，哪个对扰动更鲁棒

%% Import Date: 2026-05-26 %%
