---
title: 主线 B · 数据 & 训练侧鲁棒性 综述
status: 框架 · 待精读笔记回流后补全
target_length: 5500 字
anchor: RobustVLA (2510.00037)
---

# 主线 B · 数据 & 训练侧鲁棒性 综述

## B.0 一句话定位

通过**数据生成 / 增广 / 对抗训练 / worst-case 优化**获得显式鲁棒性——核心锚点 [[2026_RobustVLA_Guo]] 把这一切统一在多臂 bandit 框架下。

## B.1 问题域与 RobustVLA 的破局

RobustVLA 的三个划时代发现重塑了主线 B：
1. **Action 是最脆弱模态**（不是视觉！）—— π0 在 0.05 幅度 action 噪声下 SR 从 96% 跌到 52.4%，0.1 幅度下完全归零
2. **视觉鲁棒不迁移**：BYOVLA 在 action / instruction / environment 模态改进 **0.0%**
3. **多模态 worst-case 优化几乎零代价**：clean 95.5% vs 96.0%，0.5% 代价换 14% 鲁棒提升

这三个发现意味着：**单做视觉增广走不通**，必须做**多模态联合鲁棒训练**。

## B.2 七个子主题分布（25 篇精读，待回流后补）

### B1 · 多模态对抗训练 on VLA
[占位 — Eva-VLA / VLA-Fool / AdvVLA Vuln / Model-Agnostic AdvDef]

### B2 · 视觉数据增广 for Manipulation
[占位 — BYOVLA / GreenAug / RoboEngine / RoboVIP / RoboAug / RESample]

代表工作的演化线：
- 2022: CACTI（多场景采集）
- 2023: ROSIE（diffusion inpainting）
- 2024-07: GreenAug（绿屏抠图）
- 2024-10: BYOVLA（运行时干预）
- 2025-03: RoboEngine（语义分割）
- 2026-01: RoboVIP（multi-view video gen）
- 2026-02: RoboAug（region-contrastive）

### B3 · 物理域随机化
[占位 — RoboCat / DR Survey / Understanding DR]

### B4 · 合成数据 / Sim-to-Real
[占位 — RoboGen / Gen2Sim / Genesis / UniSim / Holodeck / RoboCasa / MimicGen]

### B5 · 噪声 demo 鲁棒训练
[占位 — Robust IL from Noisy Demos]

### B6 · Worst-case action / DRO
[占位 — ReMix / DRO Library / DRO Survey]

### B7 · Cross-Embodiment Generalization
[占位 — Open X-Embodiment / CrossFormer / HPT / OpenVLA / X-VLA]

代表工作的时间线：
- 2023-10: Open X-Embodiment（1M+ 跨机体数据集）
- 2024-06: OpenVLA（7B 开源标杆，970K demos）
- 2024-12: π0（diffusion / flow matching，跨机体）
- 2024-08: CrossFormer（30 具化 × 900K traj）
- 2024-09: HPT（异构 tokenizer + 共享 trunk）
- 2025-10: X-VLA（软提示 transformer）

## B.3 与曦源立项的关系（已能写）

主线 B 的 RobustVLA 是**最直接对应原 EfVLA 鲁棒性方向**的工作，但被导师认为 AAC × 鲁棒性耦合难度大。**现在收窄到鲁棒性主线后，主线 B 的策略选择有四种**：

1. **直接复现 RobustVLA + 改进 bandit 算法**（创新性低，可达性高）
2. **把 RobustVLA 应用到 SmolVLA**（轻量化对照实验，novelty 中等）
3. **RobustVLA + Synthetic data 联合**（用 RoboGen / MimicGen 生成训练数据，与 worst-case 优化叠加）
4. **指令侧扰动训练**（RobustVLA 主要在 visual + action，instruction 维度未深入）

## B.4 与主线 A/C 的接口

- **B × A**：RobustVLA worst-case δ 训练 + FocusVLA Cascaded Attention 是"训练 + 架构"双重防御（**方案候选**）
- **B × C-iii**：Recovery 训练数据如何采集？可借鉴 RobustVLA worst-case δ 思路生成 failure scenarios
- **B × C-ix**：训练时 worst-case + 测试时 adaptation 是完整的"训练 + 推理"双侧鲁棒框架

[详细子主题综述、关键论文对比、未解问题汇总 — 待精读笔记完成]
