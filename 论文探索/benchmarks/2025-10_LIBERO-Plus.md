---
title: "LIBERO-Plus: In-Depth Robustness Analysis of Vision-Language-Action Models"
authors: "Senyu Fei et al. (13 authors)"
year: "2025"
venue: "arXiv preprint (2025-10)"
arxiv: "2510.13626"
status: "已精读"
tier: "⭐⭐⭐ 推荐曦源 TOP-1"
tags: [literature, T1D, benchmark, robustness-eval, LIBERO-family, perturbation]
---

# LIBERO-Plus — 7 维扰动鲁棒性专测

> [!info] 元信息
> - **作者**：Senyu Fei 等 13 人
> - **arXiv**：[2510.13626](https://arxiv.org/abs/2510.13626)
> - **数据集规模**：~10,030 perturbed tasks，覆盖原 LIBERO 全部 4 suites
> - **rolling leaderboard**：无固定 leaderboard，但论文+HF 持续更新

## 📄 这个 benchmark 是做什么的

在 LIBERO 4 suites（Spatial/Object/Goal/Long）基础上，沿 **7 个扰动维度** 系统化生成扰动版评测集，揭示主流 VLA「benchmark 高分但实际脆弱」的现象。

七维扰动：
1. **Objects layout** — 物体布局
2. **Camera viewpoints** — 摄像头视角
3. **Robot initial states** — 机器人初始姿态
4. **Language instructions** — 语言指令改写
5. **Light conditions** — 光照
6. **Background textures** — 背景纹理
7. **Sensor noise** — 传感器噪声

每个生成维度产出 ~500 instance × 4 sub-task × 7 dim = 14,000 候选 → 过滤天花板效应后保留 ~10,030 tasks，约 2,500/suite。

## 🧠 我的评估

### 评测维度（关键！）
- **7 维扰动**：视觉侧（layout / viewpoint / light / texture / noise）+ 状态（init state）+ 语言（instruction）
- **与 VLA-Risk 关系**：
  - **覆盖更广**（7 dim vs VLA-Risk 6 dim），但 VLA-Risk 的"infeasible instruction"（attr/action/space impossibility）维度 Plus **没有**
  - VLA-Risk 测「攻击成功率 ASR」，Plus 测「绝对 SR 退化」——视角不同
  - **互补**：Plus 测全 dim 平均退化，VLA-Risk 测最坏 case 攻击面

### 任务规模
- ~10,030 perturbed tasks
- 覆盖原 LIBERO 全部 130 base tasks × 7 维扰动 × 多 instance
- 无难度分级（但视角/初始状态被论文标为最难）

### 评估 baseline 模型
**10 个主流 VLA**（极强）：
- **OpenVLA** + **OpenVLA-OFT** 变体
- **π0** + **π0-fast**
- **Nora**, **WorldVLA**, **UniVLA**, **RIPT-VLA**

所有模型都有 HuggingFace checkpoint，零门槛复现。

### 关键 finding
- **视角 + 初始状态最致命**：SR 从 95% → 30% 以下（断崖式 60+ pp 退化）
- **语言扰动几乎无影响**：揭示「VLA 实际上**忽略**语言指令」——只靠视觉做模式匹配
- **「高 LIBERO 分数 ≠ 真实能力」**——直接戳穿 SOTA 论文的水分

### 复现成本
- **GPU**：推理只需 1×24GB（OpenVLA 7B）；训练 8×A100
- **仿真器**：基于原 LIBERO 的 robosuite + MuJoCo
- **评测时间**：10K tasks × 1 episode ≈ 24-48 h on 1 GPU；可子集化（每 dim 500 tasks ≈ 2 h）
- **开源**：✅ 已 release（HuggingFace + GitHub）

### 是否适合曦源课题
- ✅✅✅ **强烈推荐 TOP-1**
- 理由：
  1. **直接对接 FocusVLA**——可以一行命令测视觉聚焦在 7 dim 上的鲁棒增益
  2. **显存 46GB 完全够用**（推理）
  3. **10 个 baseline 已公开**——无需自跑对比，直接拼成完整表格
  4. **故事最强**：「FocusVLA 视觉聚焦能否抗 viewpoint / texture 扰动」就是一个完整实验
  5. 子集化灵活，本科生时间预算可控（2 h 跑 single-dim 调试）

## 🔗 关联
- [[VLA-Risk_ICLR2026]]：维度互补（Plus 测视觉 + 普通指令；VLA-Risk 测 infeasible 指令）
- [[2023-06_LIBERO_Original]]：基础任务来源
- [[2025-10_LIBERO-PRO]]：同月 sibling，PRO 更激进（90%→0%）
- [[2026-03_LIBERO-Para]]：Plus 的语言维度被认为太弱，Para 补齐
