---
title: "RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation"
authors: "Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, Chuang Gan"
year: "2023"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2311.01455"
arxiv: "2311.01455"
venue: "arXiv 2023-11 → ICML 2024"
citekey: "wangRoboGenInfiniteData2023"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 生成式合成数据开山"
tags: [literature, T1D, 主线B, B4_合成数据, LLM驱动, 自动任务生成, RoboGen, 演化起点]
---

# RoboGen — LLM 驱动的生成式机器人学习管线

> [!info] 元信息
> - **作者**：Yufei Wang (CMU), Zhou Xian (CMU/MIT-IBM), 等 9 人
> - **机构**：CMU + MIT-IBM Watson Lab
> - **日期**：2023-11-02 (arXiv) → ICML 2024
> - **arXiv**：[2311.01455](https://arxiv.org/abs/2311.01455)
> - **项目主页**：robogen-ai.github.io
> - **代码**：开源（CMU）
> - **演化定位**：「RoboGen → Gen2Sim → GenSim2」生成式数据线路的开山之作

## 📄 Abstract

机器人学习长期受限于"任务数据稀缺"。RoboGen 提出**全自动 LLM-driven 任务生成 pipeline**：(1) **Task Proposal**——GPT-4 提议合理的机器人技能任务；(2) **Scene Generation**——LLM 生成 URDF + 物体布局；(3) **Training Supervision Generation**——LLM 自动写 reward function / sub-goal 序列；(4) **Skill Learning**——RL / motion planning 求解。整个 pipeline 无需人工干预，可在 100 任务/day 速度持续产生**多样化合成数据**。论文展示在 100+ 自动生成任务上 PPO + RL 训练出可执行 skill，并 zero-shot 迁移到部分真机。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「LLM 是合成数据的 universal generator」**——RoboGen 的核心断言。传统合成数据 pipeline 需要工程师手写任务/场景/奖励，每个任务几天到几周。RoboGen 用 GPT-4 把这三个全部自动化，**单卡 + LLM API 可日产 100 任务**。这是合成数据从「手工资产」到「LLM 资产」的范式转移。
2. **生成式 vs 程序化的根本差异**：传统 procedural generation（如 ProcGen / RoboCasa）依赖预定义模板，**任务多样性受模板限制**；RoboGen 让 LLM 自由提议，理论上**任务空间无界**。但代价是质量参差（有的任务物理不合理、有的 reward 写错）。
3. **「LLM 写 reward function」是关键技巧**：这是 RoboGen 最被低估的贡献。Reward shaping 是 RL 的传统难题，而 LLM 把它转化为「**让 GPT-4 根据任务描述生成 Python 函数**」的问题。**这一思想后来被 Eureka (NVIDIA 2024) 单独拿出来做成 SOTA**——RoboGen 算 Eureka 的先行版。

### 方法论（5 步 pipeline）

#### Step 1: Task Proposal
- LLM prompt：「你是机器人专家，给定 Franka 手臂和厨房环境，提议 10 个常见技能任务」
- 输出：任务列表（如 "open microwave", "stack two cups", "wipe table"）

#### Step 2: Scene Generation
- LLM 输出 URDF 描述（物体类型、位置、关节）
- 关键：用 **PartNet-Mobility** 数据集 + LLM 选合适物体
- 物理性检查：用 PyBullet 跑 5 秒模拟，filter 掉穿模/重力不稳定的场景

#### Step 3: Training Supervision Generation
- **Sub-goal decomposition**：LLM 把任务拆为 sub-goal 序列
  - e.g. "open microwave" → [grasp handle, pull handle, release]
- **Reward function**：LLM 写 Python 函数计算每个 sub-goal 的 reward
  - 通常是距离/角度/速度的加权和

#### Step 4: Skill Learning
- **Motion planning** 优先（OMPL 求解 grasp/place）
- **RL fallback**（contact-rich tasks 用 PPO，1M-10M env steps）
- **Diffusion policy**（部分任务用）

#### Step 5: Real-World Transfer
- 视觉 DR：纹理/光照随机
- Sim-to-real：直接 deploy，部分任务 50% 真机成功率（无 fine-tune）

### 关键实验数据

#### 自动生成质量
| 阶段 | 数量 | LLM 成功率（人工 sanity check） |
|---|---|---|
| Task proposal | 100/day | 78% 合理 |
| Scene generation | 100/day | 65% 物理合理 |
| Reward function | 100/day | 52% 一次可用，30% 需修正 |
| Skill learning | 100 tasks | 41% 训出可用 policy |

#### Skill 质量（vs 人工写 reward）
- 自动 reward 训出 policy 平均 SR: 64%
- 人工 reward 训出 policy 平均 SR: 71%
- **gap ~7 pp**，但 LLM 路径速度 ~100×

#### Real-world transfer
- 选 5 个"看起来稳"的任务做真机
- 平均 SR：~42%（远低于 sim 的 64%）
- 主要失败：grasp 偏差、物理参数不一致

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **合成数据可作 RobustVLA 训练源**：RoboGen 产生的多样化任务可以作为 [[2026_RobustVLA_Guo]] 的训练 base，再用 worst-case δ 增广。两者**完全正交**：RoboGen 解决"任务多样性"，RobustVLA 解决"扰动多样性"。
2. **本科生 46 GB 可行的关键判断**：
   - **LLM API 部分**：✅ 完全可行（GPT-4 / Claude 3 Sonnet 几美元/天）
   - **Scene generation**：✅ 可行（PyBullet/Sapien 单卡）
   - **RL skill learning**：⚠️ 慢（PPO 单任务 ~6h on A6000）
   - **完整 pipeline**：✅ 但每天产能从 100 task 降到 20 task
3. **与 [[VLA-Risk_ICLR2026]] 关联**：RoboGen 完全没考虑指令侧扰动鲁棒性，所有合成数据都是"clean instruction"。**这是 RoboGen 的关键缺陷**——可设计「Adversarial RoboGen」让 LLM 同时生成指令扰动。
4. **方法论灵感**：让 LLM 生成 reward 的思路可应用到 **「让 LLM 生成 worst-case δ 的物理合理范围」**——既保证扰动的物理合法性，又最大化对抗性。

### 待解问题 / 后续追读

1. **LLM 幻觉导致的物理不合理 scene**：35% 的 scene 不合理，需要 PyBullet 二次过滤。是否有更聪明的 filter？
2. **Reward function 质量瓶颈**：30% reward 需要人工修正——这是 LLM-only pipeline 的瓶颈。Eureka 2024 用 RL feedback loop 自动改善 reward，这是直接后继。
3. **没考虑指令-动作语义对齐**：所有 task 都是单步指令，没有复杂的多步推理任务。**Gen2Sim 部分补足**。

### 在演化线上的位置

> **RoboGen（2023.11）→ Gen2Sim（2023.10，并行）→ Eureka（NVIDIA 2024）→ GenSim2（2024.10）→ DexMimicGen（2024.10）→ RoboCasa（2024）**

RoboGen 是**「LLM 驱动合成数据」的开山之作**，与 Gen2Sim 几乎同时发表（前后差几周），形成两条平行路线：
- **RoboGen**：从 LLM 任务提议出发，端到端生成
- **Gen2Sim**：从 2D 图像/视频出发，3D lifting + URDF 生成

后续 GenSim2 = RoboGen 思想 + 强化迭代；Eureka = RoboGen reward 模块 + RL feedback loop。

### 设计风险 / 复现挑战（46 GB 显存约束下的可行性）

| 模块 | 单卡可行性 | 资源需求 | 时间 |
|---|---|---|---|
| LLM API（GPT-4/Claude） | ✅ 不依赖本地 GPU | ~$10/day | 实时 |
| URDF generation + scene | ✅ CPU/PyBullet | 4 GB RAM | 实时 |
| RL training (PPO) | ✅ 单卡（10 任务/day） | 24-40 GB GPU | 6 h/任务 |
| Motion planning | ✅ CPU | - | 1 min/任务 |
| Diffusion policy training | ✅ 可行 | 30 GB GPU | 4-8 h/任务 |
| **完整端到端 pipeline** | ✅ 完全可行 | 1× A6000 + LLM API | 20 任务/day |
| **本科生一年期评估** | ✅ **强烈推荐** | 这是 B4 最适合本科生的入门点 |

**核心可行性结论**：RoboGen 是 12 篇里**本科生最容易上手的合成数据 pipeline 之一**。代码完全开源，单卡 A6000 即可端到端跑通，LLM API 成本可控。

%% end my-thoughts %%

## 🔗 关联笔记

- **B4 同分类（生成式合成数据）**：[[2023_Gen2Sim_Katara]], [[2024_Genesis]], [[2024_Holodeck]], [[2024_RoboCasa]]
- **演化前后**：→ [[2024_Eureka_NVIDIA]]（reward 路线）→ [[2024_GenSim2_Hua]]（迭代版）
- **互补路线**：[[2024_MimicGen_Mandlekar]]（数据增广，非生成）
- **鲁棒性整合**：[[2026_RobustVLA_Guo]], [[VLA-Risk_ICLR2026]]
- **B3 关联**：[[2024_DR-Survey_Various]]（合成数据本身就是 DR 的极端化）

## 📌 Action Items

- [ ] 复现 RoboGen 在 LIBERO 上的简化版（用 Claude API 替 GPT-4）
- [ ] 设计「Adversarial RoboGen」：LLM 同时生成扰动指令
- [ ] 评估 RoboGen 合成任务对 RobustVLA 训练的增益
- [ ] 在 VLA-Risk 上测 RoboGen-trained VLA 的鲁棒性

%% Import Date: 2026-05-26 %%
