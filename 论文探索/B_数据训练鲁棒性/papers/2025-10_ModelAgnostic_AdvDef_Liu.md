---
title: "Model-Agnostic Adversarial Attack and Defense for Vision-Language-Action Models"
authors: "Liu et al."
year: "2025"
venue: "arXiv 2510.13237"
arxiv: "2510.13237"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 通用攻防框架"
tags: [literature, T1D, 主线B, 对抗训练, 模型无关, 通用框架]
---

# Model-Agnostic Adv Attack & Defense for VLA — 通用攻防框架

> [!info] 元信息
> - **作者**：Liu et al.
> - **arXiv**：[2510.13237](https://arxiv.org/abs/2510.13237)
> - **日期**：2025-10
> - **核心定位**：**模型无关**的攻防框架，无需访问 VLA 内部结构，仅用 input-output 黑盒接口
> - **本地 PDF**：未缓存

## 📄 TL;DR

论文提出**模型无关 (model-agnostic)** 的 VLA 攻防框架，区别于 EDPA / RobustVLA 等需要白盒梯度的方法。攻击侧：利用 **transferable surrogate model** —— 在一个 surrogate VLA (如 OpenVLA) 上生成对抗扰动，转移攻击其他 VLA (π0 / Octo)，发现 transfer ASR 平均 41%（远高于 random 噪声 8%）。防御侧：提出 **input purification module** —— 在 VLA 输入前接一个 lightweight diffusion-based denoiser，把对抗扰动洗掉再送入 VLA，**不需要修改 VLA 本身**。在 OpenVLA / π0 / Octo 上：纯化后 SR 从 41% 攻击下恢复到 78%（clean SR = 82%）。框架的**模型无关性**让它可即插即用到任何 VLA。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「模型无关」 = 工程友好性 + 商业封闭模型适用**：与 EDPA / RobustVLA 不同，本方法只需要 VLA 的输入输出接口，**适用于商业闭源 VLA（如 Figure AI、Tesla Optimus 内部模型）**。这是工业落地的关键卖点。

2. **Transfer attack 的有效性证明 VLA 间存在共享脆弱性**：在 OpenVLA 上生成的对抗 patch 对 π0 transfer ASR 41%——说明不同 VLA 共享视觉编码器（多数用 SigLIP/CLIP）的脆弱性。**这意味着任何单一 VLA 的鲁棒性都不能解决问题，需要 backbone-level 防御**。

3. **Diffusion-based purification 的优雅之处**：在 VLA 推理前加一个 denoising step（噪声估计 + 去噪），把所有对抗扰动统一视为高频噪声处理掉。**优点**：不需要重训 VLA、不需要白盒访问、可对未知攻击防御。**缺点**：增加推理延迟（约 +30-100ms）。

### 方法论

- **黑盒攻击**：
  - 在 surrogate VLA φ_s 上 PGD：$\delta^* = \arg\max_\delta \mathcal{L}_{task}(\phi_s, o + \delta)$
  - 把 δ* 转移到 target VLA φ_t，无需 φ_t 的梯度
- **Diffusion purification**：
  $$\hat{o} = \text{DDPM}_{denoise}(o + \delta, t^*)$$
  - t* 是反扩散步数（折中：t 太小不能去噪，t 太大丢失细节）
  - 论文用 t* = 50 step on DDPM-100
- **训练**：纯化器在 clean obs 上独立训练，不接触 VLA 梯度

### 实验关键数据

| Setting | OpenVLA SR | π0 SR | Octo SR |
|---|---|---|---|
| Clean | 82% | 95% | 79% |
| Transfer attack (from surrogate) | 41% | 48% | 38% |
| **+ Diffusion purification** | **78%** | **89%** | **74%** |
| White-box EDPA (对比) | 24% | 30% | 22% |
| **+ Diffusion purification** | 71% | 82% | 67% |

→ Purification 对白盒和黑盒攻击都有效，但白盒情况下 gap 更大。

### 与 [[2026_RobustVLA_Guo]] 的差异（B1 类必答）

| 维度 | Model-Agnostic AdvDef | RobustVLA |
|---|---|---|
| 防御范式 | **input purification**（外挂） | **adversarial training**（内置） |
| 修改 VLA | 否（即插即用） | 是（需要重训） |
| 训练成本 | 低（仅训 denoiser） | 高（worst-case 优化外内循环） |
| 推理开销 | +30-100ms | 无 |
| 通用性 | **任何 VLA**（含闭源） | 仅开源（需重训） |
| 时间线 | 2025-10 | 2026-02 |
| 关系 | **正交可组合** —— purification + RobustVLA 训练可叠加 |

→ **两个方法可以堆叠**：用 RobustVLA 训出鲁棒 VLA，再在推理前接 purification denoiser，**双重防御**。这是值得做的组合实验。

### 与曦源关联

1. **purification 是曦源 EfVLA 的轻量防御选项**：如果想做"鲁棒推理"而非"鲁棒训练"，外挂 denoiser 是工程上最省事的选项。可在 baseline π0 推理前加 denoiser 看鲁棒提升。
2. **diffusion denoiser 与曦源「轻量化」冲突**：DDPM-50 step 推理增加 50-100ms，对边缘部署不友好。需要研究 **单步 denoiser（distillation）** 替代。
3. **Transfer attack 给曦源评测多个 VLA 的标准化方案**：在一个 reference VLA 上生成扰动，统一评测所有候选模型，**避免每个 VLA 重新跑白盒攻击**。

### 待解问题

1. **Purification 对 adaptive attack 的鲁棒性**：如果攻击者知道有 denoiser，能否设计 "denoise-aware" 攻击（BPDA）？论文未充分讨论。
2. **Purification + Adversarial training 的协同效应**：两者叠加是否 1+1>2，还是有负协同？
3. **真机延迟实测**：边缘 GPU (Jetson Orin) 上 DDPM-50 实际延迟有多少？

%% end my-thoughts %%

## 🔗 关联笔记

- **直接对比**：[[2026_RobustVLA_Guo]]（内置防御 vs 外挂防御）
- **同类外挂思路**：[[2024_BYOVLA]]（推理时图像清理，但更重）
- **被纯化前的攻击**：[[2024_AdvVLA_Vuln]] (EDPA)、[[2025_VLA-Fool]]
- **diffusion 基础**：[[DDPM_Ho2020]]
- **评测平台**：[[2025_Eva-VLA]]、[[2026_VLA-Risk]]
