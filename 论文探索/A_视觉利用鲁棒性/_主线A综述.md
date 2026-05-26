---
title: 主线 A · 视觉利用鲁棒性 综述
status: 框架 · 待精读笔记回流后补全
target_length: 4500 字
anchor: FocusVLA (2603.28740)
---

# 主线 A · 视觉利用鲁棒性 综述

## A.0 一句话定位

通过**改进 VLA 对视觉信息的利用机制**（而非增加视觉编码器质量）来获得**隐性鲁棒性**——这条路线由 FocusVLA 2026-03-30 定义。

## A.1 问题域

视觉是 VLA 最大的输入信号源，但当前主流 VLA 存在三个结构性缺陷（FocusVLA 总结）：
1. **架构 shortcut**：mixed attention 让 action query 绕过视觉细节
2. **token 过载**：512+ visual token 稀释 attention 焦点
3. **任务无关噪声**：背景 / distractor / 纹理变化引入扰动

主线 A 的所有工作都在试图解决这三个问题之一或多个。

## A.2 七个子主题分布（28 篇精读，待回流后补）

### A1 · Visual token pruning / compression
[占位 — 等 P3 agent 完成 VLA-Pruner / VLA-IAP / LUVC / TPRL / Compressor-VLA 笔记后填充]

代表工作时间线（前述）：
- 2024-10: ZipVL (layer-wise adaptive)
- 2025-02: VIST (slow-fast)
- 2025-09: Action-aware Dynamic Pruning, Differentiable Token Pruning
- 2025-11: VLA-Pruner (dual-objective), Compressor-VLA (instruction-conditioned)
- 2025-12: LUVC (lossless 2x), VLA-IAP (training-free)
- 2026-03: TPRL (RL-based)

### A2 · Attention guidance / saliency 注入
[占位 — Gaze-Reg VLA / AutoFocus-IL / Attention-Guided Voxel]

### A3 · Task-relevant patch selection（与 FocusVLA 同代）
[占位 — EF-VLA / GuidedVLA / SpatialVLA / ST-VLA]

### A4 · Visual grounding for manipulation
[占位 — ReKep / VoxPoser / SG-VLA / Grounded SAM]

### A5 · 多视角 / 多分辨率融合
[占位 — GP3 / MV-VDP / Action Images / VIPA-VLA]

### A6 · Shortcut learning / spurious correlation 诊断
[占位 — Shortcut Learning Diagnosis / InSpire / CF-VLA / CAST]

### A7 · Encoder 选择对鲁棒性的影响
[占位 — SemanticVLA / Cobra / DINOBot]

## A.3 与曦源立项的关系（已能写）

主线 A 是导师推荐的导航星路线。**核心立项假设**：「让 attention 真正看任务相关区域 → 自然抑制背景扰动 / 视觉噪声 → 鲁棒性提升」。

但 FocusVLA 自己留下了三个未解空白：
1. 「鲁棒性对**背景 + 纹理同时变化**仍有限」（作者自述）
2. 「对**初始状态敏感**」（机器人臂不在标准位置就失败）
3. 「**VLM 内部的视觉利用**没涉及」（只改了 action head attention）

这三个空白都能成为曦源课题方案。

## A.4 与主线 B/C 的接口

- **A × B**：FocusVLA 的 Cascaded Attention 与 RobustVLA 的 worst-case δ 训练正交，可叠加（架构 + 训练损失双重防御）
- **A × C-iii**：视觉聚焦失败时如何触发恢复？需要 attention entropy 作为失败信号
- **A × C-ix**：测试时如何动态调整 patch selection 的 top-K？

[详细子主题综述、关键论文对比、未解问题汇总 — 待精读笔记完成]
