---
title: "Learning Interactive Real-World Simulators (UniSim)"
authors: "Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, Pieter Abbeel"
year: "2023"
journal: "ICLR 2024 (Outstanding Paper Award)"
doi: "10.48550/arXiv.2310.06114"
arxiv: "2310.06114"
venue: "ICLR 2024 Outstanding Paper"
citekey: "yangLearningInteractiveRealWorld2023"
itemType: "conference paper"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 生成式 real-world simulator"
tags: [literature, T1D, 主线B, B4_合成数据, 生成式仿真, 视频扩散, zero-shot迁移, UniSim, ICLR2024]
---

# UniSim — 生成式 real-world simulator

> [!info] 元信息
> - **作者**：Mengjiao Yang (UC Berkeley/Google DeepMind), Yilun Du (MIT), 等 6 人
> - **机构**：UC Berkeley + Google DeepMind + MIT
> - **日期**：2023-10-09 (arXiv) → ICLR 2024
> - **arXiv**：[2310.06114](https://arxiv.org/abs/2310.06114)
> - **奖项**：ICLR 2024 Outstanding Paper Award
> - **核心定位**：把视频扩散模型当作"通用 simulator"，让 VLA 在生成视频上训练 + zero-shot 真机迁移

## 📄 Abstract

传统机器人 simulator 依赖手工建模（物理引擎 + 3D 资产）。UniSim 提出**完全数据驱动的 simulator**：基于真实视频 + action data，训练一个**视频扩散模型** $p(v_{t+1:T} | v_{1:t}, a)$，给定历史帧和 action 预测未来帧。这个 simulator **没有物理引擎**——所有 dynamics 都隐含在 diffusion model 的视频生成能力中。论文证明：(1) 在 UniSim 内训练的 RL policy 可 **zero-shot 迁移到真机**；(2) 视觉/语言/动作的多模态训练让 simulator 支持 high-level instruction（如 "open the drawer"）+ low-level control（如 joint angles）；(3) UniSim 可用于训练 VLM-style agent（在生成视频上做自然语言 grounding）。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「Simulator 不需要物理引擎」是范式革命**：传统 sim = 物理引擎 + 资产；UniSim = 一个视频扩散模型。这把 sim-to-real gap 的本质从「物理逼真度」转移到「生成模型保真度」——而后者随 diffusion model 进步**会持续缩小**。这暗示未来 simulator 演化路径：**Genesis（精确物理） → UniSim（数据驱动） → 二者融合？**
2. **「Action-conditioned video diffusion」是 VLA 训练的新范式**：UniSim 的视频扩散模型学到了 (observation, action) → (next observation) 的 dynamics，可作为 **world model** 用于 VLA 训练。这与 DreamerV3 / IRIS 等 world model 路线殊途同归，但是**第一次在真实世界视频上训出可用 simulator**。
3. **Real-world simulator 隐含「视觉鲁棒性」**：UniSim 训于真实视频，自带场景多样性（不同厨房、光照、相机角度）。**VLA 在 UniSim 内训出来天然抗视觉扰动**——但代价是 dynamics 精度不如物理引擎。这与 [[2024_DR-Survey_Various]] 的「Generative DR」方向完全对应。

### 方法论（核心架构）

#### Video Diffusion Backbone
- 基于 Imagen Video / Stable Video Diffusion
- 输入：past frames $v_{1:t}$ + action sequence $a_{1:T-1}$ + 可选 text instruction
- 输出：predicted frames $v_{t+1:T}$
- 关键：**Action 通过 cross-attention 注入** diffusion U-Net

#### Training Data Mixture
- 大规模真实视频：Something-Something V2、Ego4D、Bridge 等
- 配 action 数据：RT-X、Open X-Embodiment、自采集
- 文本指令：自动 caption（用 PaLI-X）

#### Training Pipeline
```
Stage 1: 无 action 视频预训练（学通用 dynamics）
Stage 2: + action 条件微调（学 controllability）
Stage 3: + 语言指令 grounding（学 high-level semantic）
```

#### Policy Training in UniSim
- RL policy 把 UniSim 当 black-box simulator
- Reward 用 CLIP-style 视频相似度（与目标视频比较）
- 训练量：在 UniSim 内 100M+ steps

#### Zero-shot Real Transfer
- 直接 deploy 到真机
- 关键 trick：domain calibration（视觉特征对齐）

### 关键实验数据

#### UniSim 视频生成质量
| 指标 | 训练帧 | 自回归 32 帧 | 64 帧 |
|---|---|---|---|
| FID | 12 | 28 | 47 |
| PSNR | 34 | 24 | 19 |
| Action consistency | - | 0.81 | 0.69 |

#### RL policy zero-shot 真机迁移
| Task | UniSim train | Direct real | Sim2Real（physics） |
|---|---|---|---|
| Pick & place | 71% | 78% | 64% |
| Drawer open | 58% | 71% | 49% |
| Stacking | 41% | 64% | 37% |

#### 视觉鲁棒性（背景扰动）
| 背景 | Physics sim trained | UniSim trained |
|---|---|---|
| 原始 | 64% | 71% |
| 新桌面贴纸 | 38% | 62% |
| 新光照 | 41% | 65% |

**关键观察**：UniSim 训出的策略**视觉鲁棒性显著高于物理 sim 训出的策略**——这是它最被低估的贡献。

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **本科生 46 GB 可行性评估**：⚠️ **训练不可行，推理/采样可行**
   - UniSim 训练需 ~100 × A100，**完全超出本科生能力**
   - 但**预训练好的视频扩散模型**（Open-Sora、Stable Video Diffusion）可作为替代
   - 用 SVD 在 LIBERO 视频上 fine-tune（LoRA） → 简化版 UniSim，单卡 40 GB 可行
   - **本科生可行路径**：用 SVD + LIBERO action labels 训简化版 world model，验证 visual DR 增益
2. **与 [[2026_RobustVLA_Guo]] 关联**：UniSim 的视觉鲁棒性增益（+24 pp on 新光照）**远超 RobustVLA 的 worst-case δ 增益**——这暗示**生成式数据可能比对抗优化更直接**。但 UniSim 缺乏 worst-case 保证，RobustVLA 有理论 bound。**二者互补**：UniSim 提供视觉多样性，RobustVLA 提供 worst-case 保证。
3. **与 [[VLA-Risk_ICLR2026]] 关联**：UniSim 视频生成可被攻击吗？**研究空白**：对 UniSim 做 adversarial perturbation，看视频生成质量退化对下游 policy 的影响。
4. **演化定位**：UniSim 开启了「**生成式 simulator**」时代，与 Genesis 的「**精确物理 simulator**」形成两极。**未来融合**：用 UniSim 做视觉外观，用 Genesis 做物理 dynamics——Hybrid sim 可能是答案。

### 待解问题 / 后续追读

1. **Long-horizon 一致性**：UniSim 在 64 帧后开始 drift（FID 47），不能仿真长时序任务。
2. **物理违反**：视频扩散学不到精确物理，可能生成"无重力"或"穿模"的视频。
3. **训练成本不可承受**：100×A100 训练，**社区版本何时出现**？追读 Open-Sora、Stable Video Diffusion 4D。

### 在演化线上的位置

> **Dreamer (2019) → IRIS (2022) → UniSim (2023.10) → Sora (OpenAI 2024.02) → Genie 2 (DeepMind 2024) → Open-Sora (开源 2024)**

UniSim 是**第一个把视频生成 + action conditioning 联合做成功**的工作，启发了 Sora 的部分设计。但**社区可复现的开源版本是 Open-Sora + 自训 action conditioning**，尚未达到 UniSim 质量。

### 设计风险 / 复现挑战（46 GB 显存约束下的可行性）

| 用途 | 单卡可行性 | 资源 | 时间评估 |
|---|---|---|---|
| 训练完整 UniSim | ❌ | 100×A100, 数月 | 不可行 |
| 用 SVD/Open-Sora 推理 | ✅ | 30 GB | 10 s/clip |
| Fine-tune SVD on LIBERO (LoRA) | ✅ | 40 GB | 2 周 |
| 用预训模型做 sim + RL | ⚠️ | 40 GB | 实验性 |
| **本科生路径**：SVD + LoRA + 评测视觉鲁棒性 | ✅ | A6000 | 4-6 周 |

**关键判断**：UniSim 原始训练不可行，但**简化版（SVD + LIBERO LoRA）可作为本科生项目**，验证生成式 simulator 的视觉鲁棒性增益。

%% end my-thoughts %%

## 🔗 关联笔记

- **B4 同分类**：[[2023_RoboGen_Wang]], [[2023_Gen2Sim_Katara]], [[2024_Genesis]], [[2024_Holodeck]]
- **演化前后**：→ Sora / Open-Sora / Genie 2 / 4D 生成
- **互补路线**：[[2024_Genesis]]（精确物理 sim）
- **鲁棒性整合**：[[2026_RobustVLA_Guo]]（worst-case δ）, [[VLA-Risk_ICLR2026]]
- **替代工具**：Open-Sora (开源)、Stable Video Diffusion

## 📌 Action Items

- [ ] 评估 Open-Sora + action conditioning 的简化版可行性
- [ ] 比较 UniSim-style trained policy vs RobustVLA 的视觉鲁棒性
- [ ] 设计实验：对 UniSim 做对抗扰动，观察下游 policy 退化

%% Import Date: 2026-05-26 %%
