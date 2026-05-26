---
title: "RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation"
authors: "Konstantinos Bousmalis, Giulia Vezzani, Dushyant Rao, Coline Devin, Alex X. Lee, Maria Bauza, et al. (DeepMind)"
year: "2023"
journal: "arXiv preprint / TMLR 2024"
doi: "10.48550/arXiv.2306.11706"
arxiv: "2306.11706"
venue: "arXiv 2023-06 → TMLR 2024"
citekey: "bousmalisRoboCatSelfImproving2023"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读 · B3 域随机化 + self-improvement 锚点"
tags: [literature, T1D, 主线B, B3_域随机化, 视觉DR, self-improvement, multi-embodiment, RoboCat, DeepMind]
---

# RoboCat — DeepMind 自改进通用机器人代理

> [!info] 元信息
> - **作者**：Konstantinos Bousmalis 等 30 余人（Google DeepMind 主力 + Intrinsic + RWTH Aachen）
> - **日期**：2023-06-20 (arXiv) → TMLR 2024
> - **arXiv**：[2306.11706](https://arxiv.org/abs/2306.11706)
> - **机构**：Google DeepMind
> - **基座模型**：Gato 1.18B（decoder-only transformer + VQ-GAN tokenization）
> - **训练数据规模**：跨 7 种 embodiment + 36 真实/19 仿真任务，**百万级 episode**

## 📄 Abstract

RoboCat 是 DeepMind 首个**自改进**的通用机器人 VLA 基础模型。它基于 Gato 架构，统一处理图像、本体感知（proprioception）、动作 token，跨 **4 种机械臂家族**（Sawyer/Panda/KUKA/xArm）、**6 类末端执行器**（两指、三指、平行夹爪）做 imitation learning。关键创新是**自改进循环**（self-improvement loop）：用基模型在 100-1000 个真机 demo 上 fine-tune 出 specialist → 部署 specialist 自动产生 10000+ 新轨迹 → 把这些数据回灌进下一代基模型。论文证明**3 次 self-improvement 迭代后，新任务的 zero-shot 成功率从 36% → 74%**。视觉域随机化（DR）与跨 embodiment 训练是其鲁棒性来源。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「VLA 的鲁棒性来自规模 × 多样性 × 自改进闭环」**——RoboCat 在 zero-shot 新任务、新 embodiment 上的鲁棒性不是来自显式对抗训练，而是来自**百万级跨实体跨场景数据 + 视觉 DR 自然涌现**。这与 [[2026_RobustVLA_Guo]] 显式 worst-case δ 优化形成正交路线：**RoboCat = 「数据规模派」，RobustVLA = 「优化目标派」**。对本科生而言，RoboCat 路线完全不可复现（需 TPU pod + 真机集群），但其**自改进 loop 的小规模实验是可借鉴的设计模式**。

2. **Self-improvement 是 sim-to-real 的隐形通道**：传统 sim-to-real 依赖手工 DR，而 RoboCat 让模型在真机上自己生成数据自己学，本质是用**学习信号替代手工随机化**。3 次迭代后，新任务 success rate 提升 ~2 倍。这给出一个关键洞察：**真机数据自循环 > 仿真随机化**，因为真机自带物理一致性。

3. **跨 embodiment 不需要 explicit alignment**：RoboCat 没用任何对齐网络（不像 CrossFormer 的 share encoder），就直接把 7 种臂的 action 全 tokenize 到同一 vocabulary 让 transformer 学。这暗示**transformer 的 in-context generalization 能力可以隐式处理 embodiment 差异**，前提是数据量足够大。对比 X-VLA / OpenX 的复杂 embodiment encoding 设计，RoboCat 是「简单暴力派」。

### 方法论（要重现的关键技术细节）

#### 模型架构（Gato backbone）
- 1.18B 参数 decoder-only transformer
- VQ-GAN tokenize 图像（256 tokens / 图像）
- proprioception + action 直接量化为 256 levels 的离散 token
- 输入序列：`[task_id, img_1, ..., img_T, proprio_1, ..., proprio_T, action_1, ..., action_T-1]`
- 自回归预测下一个 action token

#### 自改进 Loop（最值得借鉴的设计）
```
Step 1: 用 RoboCat-base 在 100-1000 真机 demo 上 fine-tune → specialist
Step 2: 部署 specialist 自动收集 10k-50k 新 episode（含失败案例）
Step 3: 把新数据按成功标签筛选 → 加入下一代训练集
Step 4: 重新预训练 RoboCat-base → 得到 RoboCat-v2
迭代 N 次
```
关键点：**失败 episode 也有价值**（作为 "negative" 反向监督），但论文主要用成功 ep。

#### 视觉 DR 配置（B3 重点）
- 真机端：背景随机化（多种桌面贴纸）、光照变化（4 种 LED 配置）、相机视角微扰（±5° yaw/pitch）
- 仿真端：物理参数随机化（friction ∈ [0.2, 0.8]、object mass ±20%、object color HSV 抖动 ±0.2）
- **DR 强度远低于 OpenAI Rubik's Cube 论文**（那个用极度 DR），原因是 RoboCat 靠数据量补足

### 关键实验数据

#### 新任务 zero-shot 成功率（n-shot 在 specialist 上 fine-tune）
| 任务难度 | RoboCat-base | RoboCat-v1 (1 iter) | RoboCat-v2 (2 iter) | RoboCat-v3 (3 iter) |
|---|---|---|---|---|
| Easy (stacking) | 52% | 64% | 71% | 78% |
| Medium (insertion) | 36% | 50% | 62% | 74% |
| Hard (cable manipulation) | 18% | 32% | 41% | 58% |

#### 跨 embodiment 迁移（Sawyer → KUKA, zero-shot）
| 任务 | KUKA only baseline | RoboCat (mix-trained) | 增益 |
|---|---|---|---|
| Cube stacking | 41% | 67% | +26 pp |
| Vegetable lifting | 33% | 58% | +25 pp |

#### 视觉鲁棒性（背景/光照变化下的 SR）
- 原始背景：82% → 新桌面贴纸：76% → 强光下：71%
- 退化幅度小（~11 pp），但**未在 LIBERO-Plus / VLA-Risk 等标准 benchmark 上量化**

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **完全不可复现，但提供方法论锚点**：1.18B 模型 + TPU pod + 7 种机械臂集群，远超 46 GB 单卡能力。但 **self-improvement loop 的设计哲学**可以缩小到 LIBERO 范围实验：用 SmolVLA / FocusVLA 0.5B 模型，在仿真里跑 self-improvement，验证 1-2 轮迭代的效果。
2. **与 [[2026_RobustVLA_Guo]] 互补**：RoboCat 数据驱动，RobustVLA 优化驱动。**可设计联合实验**：在 RoboCat 风格的 self-improved 数据上跑 RobustVLA 的 worst-case δ，看是否进一步提升鲁棒性。
3. **VLA-Risk benchmark 缺位**：RoboCat 没测过指令侧扰动（[[VLA-Risk_ICLR2026]] 揭示指令攻击 ASR 比视觉攻击高 25 pp+）。**这是评测空白**：RoboCat 风格的 data scaling 能否抵抗指令攻击？
4. **真机自循环的成本警示**：作者报告每个 specialist 收集 10k 真机 episode 需 **3 周 × 8 真机**。本科生一年期完全不可行，但**仿真版的自改进 loop 在 LIBERO 上可能 2-4 周完成**。

### 待解问题 / 后续追读

1. **Self-improvement 的失败模式 unknown**：论文只报告成功提升，但什么时候 self-improvement 会**导致 distribution shift 失控**（模型越学越偏）？这是 DRO 视角的关键问题，跟 [[2024_ReMix]] 数据混合策略相关。
2. **VQ-GAN tokenization 的瓶颈**：image → 256 discrete tokens 会丢失细节，是否限制了精细操作？后续 GenSim2 / Diffusion-based VLA 抛弃了这个设计。
3. **跨 embodiment 训练曲线 unknown**：什么时候加入新 embodiment 是 marginal cost 最低的？这是 curriculum learning 视角的开放问题。

### 在演化线上的位置

> **RoboCat（2023.06）→ Open X-Embodiment（2023.10）→ RT-X（2024）→ π0（2024）→ X-VLA（2025）**

RoboCat 是 **数据规模 × 多 embodiment** 路线的开山之作（DeepMind 内部探索），但**未开源**，所以社区影响力不如随后开放数据集的 Open X-Embodiment（[[2024_OpenXEmbodiment]]）。RoboCat 验证了「**规模化 cross-embodiment imitation 可行**」的核心命题，为 RT-X / π0 铺路。

### 设计风险 / 复现挑战（46 GB 显存约束下的可行性评估）

| 模块 | 单卡可行性 | 备注 |
|---|---|---|
| Gato 1.18B 推理 | ✅ FP16 ~2.4 GB | 但推理速度慢 |
| Gato 1.18B fine-tune | ❌ 需 100+ GB | 不可行 |
| Self-improvement loop | ❌ 真机依赖 | 仿真版可行 |
| 视觉 DR pipeline | ✅ 完全可行 | 数据预处理，纯 CPU |
| **替代方案**：SmolVLA + self-improvement (LIBERO) | ✅ 可行 | 本科生一年期可做 |

%% end my-thoughts %%

## 🔗 关联笔记

- **B3 同分类**：[[2024_DomainRandomization_Survey]], [[2021_Understanding_DR_SimToReal]]
- **演化前后**：→ [[2024_OpenXEmbodiment]] → [[2024_CrossFormer]] → [[2024_pi0]] → [[2025_X-VLA]]
- **互补路线**：[[2026_RobustVLA_Guo]]（worst-case δ 优化派）
- **评测空白**：[[VLA-Risk_ICLR2026]], [[2026_LIBERO-Plus]]
- **替代实验路径**：[[2025_SmolVLA]] + self-improvement on LIBERO

## 📌 Action Items

- [ ] 设计 mini self-improvement loop 实验：SmolVLA 在 LIBERO-10 上跑 2 轮自改进
- [ ] 对比 RoboCat 风格数据 scaling vs RobustVLA worst-case δ 的鲁棒性增益
- [ ] 在 VLA-Risk 上评估「大数据 scaling」对指令侧扰动的抵抗力

%% Import Date: 2026-05-26 %%
