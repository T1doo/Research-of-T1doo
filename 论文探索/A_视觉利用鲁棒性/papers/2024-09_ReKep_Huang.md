---
title: "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation"
authors: "Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, Li Fei-Fei"
year: "2024"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2409.01652"
arxiv: "2409.01652"
venue: "arXiv 2024-09-03 (CoRL 2024 workshop, project page: rekep-robot.github.io)"
citekey: "huangReKep2024"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, 视觉grounding, keypoint, LLM约束, A4]
---

# ReKep — Relational Keypoint Constraints 精读笔记

> [!info] 元信息
> - **作者**：Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, Li Fei-Fei
> - **机构**：Stanford / Columbia
> - **日期**：2024-09-03 (arXiv)
> - **arXiv**：[2409.01652](https://arxiv.org/abs/2409.01652)
> - **项目页**：rekep-robot.github.io
> - **定位**：A4 视觉 grounding 经典 — 把 LLM 推理与几何约束求解器联起来的工程模板

## 📄 TL;DR

ReKep 把机器人操作任务编码成一组 **关系型关键点约束（Relational Keypoint Constraints）**，每个约束是一个 Python 函数，输入 3D 关键点集合、输出标量代价。LLM/VLM 从自然语言 + RGB-D 自动生成 ReKep；下游用层级优化求解器（hierarchical optimization）实时反解出 SE(3) 末端轨迹。论文证明：单臂 / 双臂 / 多阶段 / 反应式 (reactive) 操作任务都可由这套表示统一处理，且不需要任务专属数据或环境模型。本质是「**符号约束求解 + 视觉关键点 + LLM 生成约束**」的混合范式，与端到端 VLA 形成方法学对照。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「约束 (constraint) 是比 action 更稳的中间表示」**。端到端 VLA 直接学 action 分布，对扰动敏感；ReKep 把任务编码成关键点之间的几何约束 + 优化器，扰动只要不改变 *约束的几何意义*，求解器就能补偿。这是一种 **结构化先验** 的鲁棒性来源。
2. **关键点是 grounding 的最小单元**。比 mask / box 维度更低，但保留了 3D 拓扑信息；VLM 标 keypoint 比标 mask 更稳，且关键点天然支持多阶段（每阶段一组关键点 + 一组约束）。
3. **「Python 函数 = 可微 cost」的设计**：约束写成 Python 函数后，可被 SciPy 类求解器直接调用，绕开了"LLM 写代码 → 调 API"的脆弱链路，让 LLM 只负责生成 *函数体*，不负责执行。

### 方法论

- **关键点提取**：DINOv2 特征 + K-Means 聚类 → 物体表面的语义聚类中心作为关键点候选；VLM 再筛选 task-relevant subset
- **约束生成**：GPT-4V 把语言任务 + 关键点编号图像 → 一组 stage-wise Python 约束函数（sub-goal 约束 + path 约束）
- **求解器**：层级优化 — 上层选 stage、中层在 stage 内做 sub-goal 优化、下层做 path tracking；保持 ~10 Hz 闭环
- **跟踪**：用 CoTracker / 帧间光流维持关键点 ID 跨时间一致

### 与曦源关联（FocusVLA / RobustVLA）

- **与 FocusVLA 方法学差异**：FocusVLA 是端到端 attention 改造（隐式 grounding）；ReKep 是 **显式 grounding** — 关键点是离散符号，attention 是连续权重。两者可以互补 — ReKep 的关键点可以作为 FocusVLA 的 attention 监督信号（即 keypoint heatmap 作为 attention prior）。
- **与 RobustVLA 组合可能**：RobustVLA 在数据侧做对抗扰动；ReKep 在 *推理* 侧做约束求解。可以做 **ReKep-guided RobustVLA**：用 ReKep 关键点作为不变量，对抗训练时约束 perturbed image 的关键点位置不变。
- **暴露 FocusVLA 未涉及的子问题**：FocusVLA 没回答 "attention 到底应该聚焦哪里" — ReKep 给出一个 *外部参考答案*（VLM 选出的 task-relevant keypoint），可作为 FocusVLA attention pattern 的 ground-truth 评估锚点。
- **本科生子模块可行性**：✅✅✅ — 关键点提取 + 约束求解器都是标准技术栈（DINOv2 + SciPy），一年内可在 LIBERO 上跑通 ReKep baseline，作为 grounding 侧的对照实验。

### 待解问题

1. **约束的可微化与梯度回传**：当前求解器是 black-box 优化，无法回传梯度给视觉模块。如果想把 ReKep 嵌入 VLA 训练循环，需要让约束 layer 可微（参考 differentiable optimization 文献）。
2. **关键点跟踪失败模式**：遮挡 / 大幅相机移动 → 关键点 ID 丢失 → 求解器失效。论文没量化这个失败率。
3. **LLM 生成代码的鲁棒性**：GPT-4V 写约束函数的成功率？论文没给硬数字。

%% end my-thoughts %%

## 🔗 关联笔记

- **同代竞品**：[[2023-07_VoxPoser_Huang]]（同作者，value map 路线）
- **下游 grounding 工具**：[[2024-01_GroundedSAM_Ren]]（mask 级 grounding）、[[2024-02_DINOBot_DiPalo]]（dense feature 级 grounding）
- **互补的隐式 grounding**：[[2026-03_FocusVLA_Zhang]]（attention 改造）
- **诊断侧**：[[2025-05_InSpire_Zhang]]（spatial reasoning 的另一路径）
- **主线 A 总结**：[[A4_visual_grounding_总结]]

## 📌 Action Items

- [ ] 评估：把 ReKep 关键点作为 attention prior 注入 FocusVLA，看是否进一步提升 grounding 质量
- [ ] 对照：在 LIBERO-Spatial 上跑 ReKep vs FocusVLA，比较 spatial generalization 增益
- [ ] 思考：ReKep 的「stage-wise constraint」是否可以指导 VLA 的 sub-task decomposition

%% Import Date: 2026-05-26 %%
