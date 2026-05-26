---
title: "SG-VLA: Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation"
authors: "Ruisen Tu, Arth Shukla, Sohyun Yoo, Xuanlin Li, Junxi Li, Jianwen Xie, Hao Su, Zhuowen Tu"
year: "2026"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2603.22760"
arxiv: "2603.22760"
venue: "arXiv 2026-03-24"
citekey: "tuSGVLA2026"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, 视觉grounding, mobile manip, auxiliary supervision, A4]
---

# SG-VLA — Spatial Grounding for Mobile Manipulation 精读笔记

> [!info] 元信息
> - **作者**：Ruisen Tu, Arth Shukla, Sohyun Yoo, Xuanlin Li, Junxi Li, Jianwen Xie, Hao Su, Zhuowen Tu
> - **机构**：UCSD（Hao Su / Z. Tu lab）
> - **日期**：2026-03-24 (arXiv)
> - **arXiv**：[2603.22760](https://arxiv.org/abs/2603.22760)
> - **定位**：A4 grounding — 用 **辅助解码器 (auxiliary decoders)** 强制 VLA backbone 学习 spatially-grounded 表示

## 📄 TL;DR

SG-VLA 解决移动操作的 13-DoF 控制问题（移动底盘 + 机械臂 + 夹爪联合）。核心机制：**多视角 RGB + 深度 + 短时序历史** 作为输入，**co-training 多个辅助解码器** 在 backbone 上同时重建 (1) 机器人位置、(2) 关节构型、(3) 抓取 affordance、(4) 目标物体相对位姿、(5) 分割 mask。这些 dense supervision 强迫 backbone 学到 *spatially-grounded、manipulation-aware* 的潜变量。在家庭场景的拾取/放置/开关任务上显著优于直接 imitation learning。本质是「**多任务监督 = 视觉表示的隐式约束**」。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「辅助解码器 = 廉价的视觉 grounding 监督」**。SG-VLA 不改 backbone 架构，只在训练时加一堆 *任务无关的几何/语义重建头*；推理时丢掉。这是把 *spatial grounding 信号* 注入 backbone 的低成本方法，比对抗训练简单、比显式约束求解灵活。
2. **移动操作的 13-DoF 是个独特测试床**。移动 + 臂 + 夹爪联合让"视觉 → 动作"的映射更稀疏，更需要好的视觉表示来支撑；这也是 LIBERO 单臂场景看不到的失败模式。
3. **与 FocusVLA 的本质区别**：FocusVLA 在 *attention 计算图* 里改造；SG-VLA 在 *loss 函数* 里加监督。前者改架构，后者改优化目标 — 两条独立路径都能促 grounding。

### 方法论

- **输入**：多视角 RGB（头+腕+底盘）+ 深度 + N 帧短历史
- **辅助 heads**（co-training）：
  - 机器人 base pose 回归
  - 关节角回归
  - 抓取 affordance heatmap
  - 目标物体相对 pose 回归
  - 物体 mask 分割
- **主 head**：13-DoF action 回归（移动底盘 SE(2) + 臂 7-DoF + 夹爪开合）
- **训练**：多任务 loss 加权，推理时只用 action head

### 与曦源关联（FocusVLA / RobustVLA）

- **与 FocusVLA 方法学互补**：FocusVLA 改 attention（架构层），SG-VLA 加 auxiliary loss（优化目标层）。两者可以叠加 — 在 FocusVLA 之上加 grounding 监督，预期获得更稳的 attention pattern。
- **与 RobustVLA 组合可能**：RobustVLA 的扰动训练 + SG-VLA 的几何重建一致性 = "扰动下重建结果应不变" 的 consistency loss。这是一个具体可做的联合方案。
- **暴露 FocusVLA 未涉及的子问题**：FocusVLA 只在 LIBERO 单臂上验证；SG-VLA 暴露移动操作下视觉 grounding 的额外挑战（基座移动、视角剧烈变化）。FocusVLA 在移动场景下是否仍 work？是个开放问题。
- **本科生子模块可行性**：⭐⭐⭐ — auxiliary head 加 loss 是工程友好的修改；可在 LIBERO 上做"加几个 grounding head 是否提升 FocusVLA 鲁棒性"的小实验（4-6 周）。

### 待解问题

1. **辅助 head 之间的梯度冲突**：5 个不同任务的梯度方向不一定一致，论文有没有做 GradNorm 类平衡？
2. **辅助监督需要标注**：SG-VLA 假设有 GT pose / mask；如何在真机数据上获取？是否需要仿真预训练？
3. **推理时辅助 head 真的没用吗**？可不可以用辅助 head 输出做 self-consistency check（OOD 检测）？

%% end my-thoughts %%

## 🔗 关联笔记

- **方法学竞品**：[[2026-03_FocusVLA_Zhang]]（attention 改造路径）
- **辅助监督前驱**：[[2024-02_DINOBot_DiPalo]]（dense retrieval 用作隐式监督）
- **shortcut 诊断**：[[2025-08_ShortcutLearning_Xing]]、[[2025-05_InSpire_Zhang]]
- **移动操作 baseline**：[[2024_MobileALOHA]]
- **主线 A 总结**：[[A4_visual_grounding_总结]]

## 📌 Action Items

- [ ] 实验：在 FocusVLA 上加 SG-VLA 风格 auxiliary head，测 LIBERO + LIBERO-Plus 鲁棒性增益
- [ ] 思考：哪些辅助 head 对 grounding 增益最大？做消融
- [ ] 对照：单臂 LIBERO 上的 grounding 监督收益是否远小于 mobile manip？

%% Import Date: 2026-05-26 %%
