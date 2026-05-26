---
title: "Multi-View Video Diffusion Policy: A 3D Spatio-Temporal-Aware Video Action Model"
authors: "Peiyan Li, Yixiang Chen, Yuan Xu, Jiabing Yang, Xiangnan Wu, Jun Guo, Nan Sun, Long Qian, Xinghang Li, Xin Xiao, Jing Liu, Nianfeng Liu, Tao Kong, Yan Huang, Liang Wang, Tieniu Tan"
year: "2026"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2604.03181"
arxiv: "2604.03181"
venue: "arXiv 2026-04-03"
citekey: "liMVVDP2026"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, 多视角融合, video diffusion, heatmap, A5]
---

# MV-VDP — Multi-View Video Diffusion Policy 精读笔记

> [!info] 元信息
> - **作者**：Peiyan Li et al.（16 人，CASIA / NLPR / 字节 / Tsinghua 联合）
> - **机构**：CASIA NLPR、ByteDance、清华
> - **日期**：2026-04-03 (arXiv)
> - **arXiv**：[2604.03181](https://arxiv.org/abs/2604.03181)
> - **定位**：A5 多视角融合 — 把 action 当 "未来视频生成" 来做，多视角 heatmap 监督

## 📄 TL;DR

MV-VDP 把机器人 policy 表述为 **多视角 (heatmap + RGB) video 联合预测** 任务。模型同时生成：(1) 未来多视角 RGB 视频（环境如何演化），(2) 多视角 heatmap 视频（机器人/物体应在哪里）。这种"视频预训练格式 + action fine-tune"对齐方式，让 *世界模型预训练成果* 能直接迁移到 action policy。Meta-World + 真机实验上仅需 10 条 demo 即可良好泛化，证明数据效率。本质是「**video generation = world model = policy 的统一表示**」。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「Video diffusion 既是 world model 又是 policy」** — 这是当前最性感的范式之一（Sora、Veo 给的启发）。MV-VDP 把这个范式落地到机器人，关键是 *heatmap 通道* — 它让"环境视频"与"动作"通过几何对齐建立联系。
2. **「Heatmap 是 action 的 2D pixel 投影」**：把 7-DoF 动作编码成像素 heatmap，自然适配 video diffusion 架构；推理时反投影回 6-DoF action。这绕开了 *如何把低维动作塞进高维 video 模型* 的难题。
3. **多视角是为了减少 heatmap 的歧义性**：单视角下 2D heatmap 对应多种 3D 动作；多视角联合约束 → 反投影唯一。

### 方法论

- **输入**：多视角当前帧 + 任务描述
- **输出**：未来 K 帧的多视角 (RGB + heatmap) 视频
- **训练**：video diffusion + heatmap 监督
- **推理**：生成 heatmap → 反投影到 3D action
- **预训练**：可利用大规模视频数据预训练 video backbone

### 与曦源关联（FocusVLA / RobustVLA）

- **与 FocusVLA 方法学差异巨大**：FocusVLA 是 *动作 query 看图*（attention）；MV-VDP 是 *生成动作图*（diffusion）。两者完全不同范式 — diffusion 训练昂贵但泛化强，attention 训练便宜但需要好架构。
- **与 RobustVLA 组合**：扰动训练时，扰动后生成的视频 / heatmap 应一致；这是 diffusion 路线的 *自然 consistency 监督*。
- **暴露 FocusVLA 未涉及的子问题**：FocusVLA 无 *未来预测*；MV-VDP 暴露"预测未来视频"对 VLA 鲁棒性的影响 — 预测好了等于有了 model-based 优势。
- **本科生子模块可行性**：⭐ — diffusion 训练 + 多视角视频数据要求很高，单机难复现；但 *把 MV-VDP 当 baseline 评估*是可行的。

### 待解问题

1. **推理速度**：video diffusion 一次推理 N 步 denoising，10+ Hz 闭环现实吗？
2. **heatmap 反投影精度**：反投影对相机标定敏感，真机部署的鲁棒性如何？
3. **数据效率的真实性**：10 demo 看起来很少，但 video 预训练用了多少数据？

%% end my-thoughts %%

## 🔗 关联笔记

- **同主题视频范式**：[[2026-04_ActionImages_Zhen]]（同样把 action 当视频）
- **3D 表示竞品**：[[2025-09_GP3_Qian]]、[[2025-12_VIPA-VLA_Feng]]
- **互补 2D attention**：[[2026-03_FocusVLA_Zhang]]
- **数据效率对照**：[[2024_MobileALOHA]]
- **主线 A 总结**：[[A5_多视角融合_总结]]

## 📌 Action Items

- [ ] 调研：MV-VDP 的 video backbone 与算力开销（是否要 A100×8）
- [ ] 思考：FocusVLA + MV-VDP 的预测式 attention — 把"未来 attention pattern" 也作为预测目标
- [ ] 评估：MV-VDP 在 LIBERO 上 vs FocusVLA 在 OOD 物体的鲁棒性差距

%% Import Date: 2026-05-26 %%
