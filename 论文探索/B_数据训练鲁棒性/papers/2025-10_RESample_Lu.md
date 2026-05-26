---
title: "RESample: OOD State Augmentation with Adaptive Noise for Robust VLA"
authors: "Lu et al."
year: "2025"
venue: "arXiv 2510.17640"
arxiv: "2510.17640"
status: "已精读"
tier: "⭐⭐⭐ 必读 · OOD 状态增广"
tags: [literature, T1D, 主线B, 数据增广, OOD状态, 自适应噪声, RESample]
---

# RESample — OOD 状态增广 + 自适应噪声

> [!info] 元信息
> - **作者**：Lu et al.
> - **arXiv**：[2510.17640](https://arxiv.org/abs/2510.17640)
> - **日期**：2025-10
> - **核心定位**：从「视觉增广」转向「**状态空间增广**」——在 BC 训练中主动构造 OOD 状态（机械臂在非演示位置）
> - **本地 PDF**：未缓存

## 📄 TL;DR

RESample 揭示了模仿学习 (BC) 的一个长期痛点：**演示轨迹是单一连续路径，VLA 看到的全是 "在路径上" 的状态**，一旦推理时偏离路径（执行误差 / 扰动）就陷入 OOD 失败。RESample 在训练时主动注入**自适应噪声** 让机械臂偏离演示轨迹，同时**重新生成相应的 corrective action**（用 inverse dynamics model 或 retargeting），创造 "OOD 状态 → 回归路径" 的训练样本。在 LIBERO-Long 上：RESample 把 OpenVLA SR 从 76% → 87%，且对**初始位置偏移**鲁棒性大幅提升（偏移 20cm SR 从 23% → 71%）。论文核心思想：**鲁棒 BC = 大量 corrective behavior in training data**。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「视觉鲁棒 ≠ 状态鲁棒」是这条工作的核心洞察**：GreenAug / RoboEngine / RoboVIP 增广视觉（背景/纹理/视角），但机械臂状态轨迹不变；推理时如果机械臂被推到非演示位置（哪怕视觉完全 ID），VLA 仍失败。RESample 直接攻击这个 **state distribution shift**。

2. **「corrective action 数据」是 BC 鲁棒性的根本**：DAgger (Ross 2011) 已证明 corrective demo 对 BC 至关重要。RESample 把 DAgger 的人工 corrective 换成 **自动合成的 corrective**——用 inverse dynamics / retargeting。**这是 DAgger 在 VLA 时代的合成版本**。

3. **「自适应噪声」是工程关键**：均匀加噪可能破坏任务（如把抓手推到目标外几米）；RESample 根据当前任务进度自适应调整噪声幅度——任务初期可大、临近成功可小。

### 方法论

- **三步增广**：
  1. **State perturbation**：在演示轨迹的状态 s_t 上加自适应噪声 → s_t + ε_t
     - ε_t 幅度由「距离目标」自适应：远离目标允许大噪声
  2. **Corrective action synthesis**：用 inverse dynamics 求"从 s_t + ε_t 回到 s_t+1 的动作" a_corr
  3. **训练数据**：(s_t + ε_t, l, a_corr) 加入原始 BC 数据
- **训练**：标准 OpenVLA BC，无修改
- **任务**：LIBERO-Spatial / -Long / -Object / -Goal

### 实验关键数据

| Setting | LIBERO-Long SR | OpenVLA + 20cm 初始位移 SR |
|---|---|---|
| Baseline | 76% | 23% |
| GreenAug (视觉) | 78% | 25% |
| RoboEngine (视觉) | 81% | 28% |
| **RESample (状态)** | **87%** | **71%** |
| RESample + RoboEngine | **91%** | **74%** |

→ **RESample 对初始位置扰动有 3x 提升**，远超视觉增广方法。

### 与 [[2024_BYOVLA]] 对照（B2 类必答）

| 维度 | RESample | BYOVLA |
|---|---|---|
| 攻击维度 | 状态分布 (action / proprioception) | 视觉分布 |
| 路径 | 训练时增广 | 推理时清洗 |
| 工具 | Inverse dynamics + adaptive noise | VLM + SAM + LaMa |
| 鲁棒方向 | OOD 状态 | OOD 视觉 |
| 互补 | **完全正交** —— 攻击不同模态 |

### 与 [[2026_RobustVLA_Guo]] 的关系

RobustVLA 的 output-robust 模块也在 action space 加 worst-case 噪声，但**只是在 loss 上加扰动**，不真生成新训练样本；RESample **真生成 (state+ε, corrective action) 配对**。两者思想互补：
- RobustVLA output-robust = **在 loss 上**对抗 action 噪声
- RESample = **在数据上**注入 corrective behavior

**结合方案**：用 RESample 生成的 corrective trajectory 作为 RobustVLA 训练数据，再叠加 RobustVLA 的 worst-case loss。

### 能否与 RobustVLA 的 input-robust 模块结合（B2 类必答）

**可以但定位不同**：
- RobustVLA input-robust 处理 **观察空间** ω*(o) 扰动
- RESample 处理 **状态空间** s + ε 扰动
- **结合方案**：RESample 提供状态扰动样本 → 这些样本本身也有视觉差异（机械臂在不同位置）→ 可作为 RobustVLA input-robust 训练样本
- **优势**：RESample 自动覆盖了 RobustVLA 17 扰动中缺失的"初始位置偏移"维度

### 与曦源关联

1. **「初始位置鲁棒性」是 FocusVLA 论文 callout 的 limitation**：FocusVLA 作者承认"对初始状态敏感"，RESample 恰好解决此痛点。**FocusVLA + RESample 是天然组合**。
2. **Inverse dynamics 工具链可重用**：用 MuJoCo / Isaac 的 IK 求 corrective action，曦源已有此栈。
3. **真机难复现**：RESample 在 simulator 里加状态噪声容易，真机需要安全机制（机械臂不能随便加噪到非法位置）。
4. **EfVLA 的「数据效率」卖点**：RESample 用同样数据合成更多有效样本，节省真机采集成本。

### 待解问题

1. **Inverse dynamics 在复杂任务上的精度**：deformable / contact-rich 任务的 IK 不准，corrective action 错→训练崩。
2. **自适应噪声 schedule 的调参**：噪声幅度对效果影响大，论文未充分调参分析。
3. **与 RobustVLA output-robust 的实证对比**：两者都是 action-side 鲁棒化，谁更优？

%% end my-thoughts %%

## 🔗 关联笔记

- **直接对比**：[[2024_BYOVLA]]（OOD 状态 vs OOD 视觉，正交维度）
- **天然组合**：[[2026_FocusVLA_Zhang]]（解 FocusVLA 的初始位置敏感问题）
- **可堆叠**：[[2026_RobustVLA_Guo]]（output-robust loss + RESample 数据）
- **同类视觉增广（正交）**：[[2024_GreenAug]]、[[2025_RoboEngine]]、[[2026_RoboAug]]
- **理论根源**：[[2011_DAgger_Ross]]（corrective demo 在 BC 中的作用）
- **被增广对象**：[[2024_OpenVLA]]、[[2025_π0]]
- **评测平台**：[[2025_Eva-VLA]]（物理 pose 扰动维度）
