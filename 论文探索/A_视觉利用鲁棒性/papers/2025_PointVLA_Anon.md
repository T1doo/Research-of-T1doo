---
title: "Point-VLA: Pixel-Level Grounding for Vision-Language-Action Models"
authors: "Anonymous (OpenReview submission)"
year: "2025"
journal: "OpenReview submission"
doi: ""
arxiv: ""
venue: "OpenReview (链接未定位到具体论文，候选清单标注为 OpenReview)"
citekey: "anonPointVLA2025"
itemType: "preprint"
status: "信息缺失（仅根据候选清单标题与定位推断）"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, 视觉grounding, pixel-level, A4, 信息缺失]
---

# Point-VLA 精读笔记（信息缺失 / 占位）

> [!warning] 元信息缺失
> - **候选清单标注**：OpenReview submission，pixel-level grounding for VLA
> - **WebFetch 状态**：OpenReview 上未找到名为 "Point-VLA" 的 note；可能已被替换为正式名 / 已撤回 / 在 anonymous 阶段
> - **下一步**：需要导师 / 同行确认具体链接；或者根据"pixel-level grounding for VLA"关键词重新检索
> - **本笔记**：基于"pixel-level grounding" 关键词的方法学推断，**待论文具体内容确认后重写**

## 📄 TL;DR（基于关键词推断）

Point-VLA 的方法学假设：在 VLA 训练中加入 **像素级 grounding 监督**（每个 action 对应一个像素 / 点位置），让模型显式输出 "下一步要操作的像素点"。这是介于 *attention 改造（FocusVLA）* 与 *mask 监督（SG-VLA）* 之间的中等粒度 grounding 信号。可能的实现：在 VLA 输出端加一个 point head，预测 next-step grasp point / target pixel，与 action 联合训练。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（推断 / 待校验）

1. **像素点是最经济的 grounding 中间表示**：比 mask 维度低、比 bbox 信息密度高、比 keypoint 数量灵活；天然适配 transformer 输出 token。
2. **「Action 与 grounding point 联合训练」可能是隐性 attention 监督**：如果 next-action 与 point 通过 cross-attention 关联，attention 就被 *任务真实关注点* 监督。
3. **与 RT-2 / OpenVLA 同范式**：把 grounding 也变成 *输出 token 序列*（"pixel coordinate" token），可保留 VLA autoregressive 架构。

### 方法论（推断）

- VLA backbone 输出 [pixel point token, action token] 双流
- 训练时用 demo 标注的 grasp / contact point 作 GT
- 推理时可视化 point → 解释性

### 与曦源关联

- **与 FocusVLA 方法学差异**：FocusVLA 隐式（attention 权重）；Point-VLA 显式（输出 point 坐标）。两者可叠加：用 Point-VLA 的 GT point 作 FocusVLA attention 的监督。
- **与 RobustVLA 组合**：扰动后 *物理上* point 不应变，可加 point consistency loss。
- **本科生可行性**：⭐⭐⭐ — 但前提是 *拿到论文具体方法*，否则方法学只能凭空猜。

### 待解问题（首要）

1. **必须确认论文真实链接和内容**：当前笔记基于关键词推断，方法细节、benchmark、对比 baseline 全部缺失。
2. **如确认是"输出 point token" 范式**：与 *Show-o*/*EmbodiedGPT* 类工作的关系是什么？
3. **如确认是"point 监督训练"范式**：与 SG-VLA 的 auxiliary head 区别？

%% end my-thoughts %%

## 🔗 关联笔记

- **方法学竞品**：[[2026-03_FocusVLA_Zhang]]（attention）、[[2026-03_SG-VLA_Tu]]（auxiliary）
- **mask 级 grounding**：[[2024-01_GroundedSAM_Ren]]
- **关键点 grounding**：[[2024-09_ReKep_Huang]]
- **主线 A 总结**：[[A4_visual_grounding_总结]]

## 📌 Action Items

- [ ] **首要**：确认 Point-VLA 具体 arXiv / OpenReview ID（需要导师确认或 Google Scholar 搜索）
- [ ] 论文确认后：重写本笔记 TL;DR / 方法 / 关联 部分
- [ ] 思考：「pixel-level grounding」这类中等粒度信号在曦源是否值得作为单独 ablation 维度

%% Import Date: 2026-05-26 %%
