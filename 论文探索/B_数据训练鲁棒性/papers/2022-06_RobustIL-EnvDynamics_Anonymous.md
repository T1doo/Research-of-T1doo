---
title: "Robust Imitation Learning against Variations in Environment Dynamics"
authors: "Jongseong Chae, Seungyul Han, Whiyoung Jung, Myungsik Cho, Sungho Choi, Youngchul Sung"
year: "2022"
journal: "ICML 2022 / arXiv"
doi: "10.48550/arXiv.2206.09314"
arxiv: "2206.09314"
venue: "ICML 2022"
citekey: "chaeRobustImitationLearning2022"
itemType: "conference paper"
status: "精读完成"
tier: "⭐⭐ B5/B6 桥梁 · DRO 环境变化"
tags: [literature, T1D, 主线B, B5, B6, DRO, worst-case环境, sim2real, RIME]
---

# Robust IL against Env Dynamics Variations — 精读笔记

> [!info] 元信息
> - **作者**：Jongseong Chae 等（KAIST 团队）
> - **arXiv**：[2206.09314](https://arxiv.org/abs/2206.09314)
> - **B5/B6 桥梁定位**：把 DRO 的 worst-case 优化思想用到 **环境 dynamics 上**（不是 action / demo），处理 sim-to-real 时仿真器物理参数误估问题

## 📄 Abstract

IL 训练时的环境 dynamics（仿真器物理参数）与真实部署的 dynamics 通常存在差距（mass、friction、damping 等）。本文提出 **RIME (Robust IL against env Dynamics)**：
1. 把环境扰动建模为 **可能环境集合 $\mathcal{P}$**
2. 学习一个对 $\mathcal{P}$ 内**最坏环境**仍可执行的 policy
3. 通过 **AIL（Adversarial IL）框架**实现：adversary 选 worst-case env，IL agent 优化 policy
4. 与 GAIL 比较：在 MuJoCo + 多 dynamics setting 下，policy transfer success rate 提升 30-50%

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「环境扰动 vs 演示扰动 vs 动作扰动」的三分法**：
   - **演示扰动**：demo 含噪（B5 RIL-Co 解决）
   - **动作扰动**：execution 时 action 被攻击（B1 RobustVLA、B6 worst-case action）
   - **环境扰动**：仿真→真机的 dynamics gap（本文 RIME 解决）
   - 这三类问题在 VLA 工程中**同时存在**，但目前没工作把它们统一处理

2. **DRO 的「环境分布」建模视角**：
   - 标准 DRO：$\min_\theta \max_{P \in \mathcal{P}} \mathbb{E}_P[\ell]$
   - 本文：$\mathcal{P}$ 就是物理参数的扰动集合（KL ball / Wasserstein ball）
   - 与 [[2024_ReMix_DRO]]（数据分布扰动）形成对比：本文是 **physics-level DRO**

3. **AIL framework 复用**：
   - GAN-style：discriminator $D$ 判 expert vs policy rollout
   - 额外加一个 adversary $A$ 调环境参数，让 $D$ 更难判
   - **三方博弈**：generator π vs discriminator D vs env adversary A

### 方法论

#### Worst-case env 公式
$$
\min_\pi \max_{P \in \mathcal{P}} J(\pi, P) = \mathbb{E}_{\tau \sim \pi, P}\left[\sum_t r(s_t, a_t)\right]
$$
其中 $\mathcal{P} = \{P : D_{KL}(P \| P_0) \le \epsilon\}$。

#### 优化算法
1. 固定 $\pi$，PPO-style 优化 env adversary（梯度上升 KL 约束）
2. 固定 env，AIL 更新 $\pi$
3. 交替收敛

### 实验结果

#### MuJoCo Hopper（dynamics 扰动：mass × [0.5, 2.0]）
| Method | Avg Return | Worst-case Return |
|---|---|---|
| BC | 1200 | 200 |
| GAIL | 2400 | 500 |
| Domain Randomization | 2800 | 1100 |
| **RIME** | **2900** | **1800** |

#### Sim-to-Real（HalfCheetah → 真实参数偏移 30%）
- BC 失败率 60%
- RIME 失败率 18%

### 与曦源的关联

1. **直接补全 RobustVLA 的盲点**：
   - RobustVLA 只对 action 做 worst-case；现实中**动作-环境联合扰动**才是真实风险
   - 可以扩展 RobustVLA 的 perturbation set，包含 env dynamics 维度
   
2. **对 sim-to-real 链路的影响**：
   - 当前 LIBERO 训练 → 真机部署的常见失败模式之一就是 dynamics gap
   - RIME 可以作为 VLA 训练时的 **物理参数随机化 + worst-case 选择** 二合一方案
   - 与 [[B3_域随机化]] 的 Domain Randomization 思想互补：DR 是随机采样，RIME 是 adversarial 选取
   
3. **对 cross-embodiment 的潜在意义**（接 B7）：
   - 不同 embodiment 本质就是 dynamics 不同
   - 如果把 "embodiment" 看作 env 的一个维度，RIME 思想可以让 policy 对**任意机体形态**鲁棒
   - **这是 X-VLA / HPT 的理论延伸**

### 待解问题

1. **可行扰动集合 $\mathcal{P}$ 的设定**：KL ball 半径 $\epsilon$ 难定，太小无意义、太大过拟合 worst case
2. **三方博弈的收敛稳定性**：实践中训练不稳，需要大量 trick
3. **未在 manipulation task 上验证**：本文 MuJoCo locomotion，机械臂 manipulation 的扰动空间更复杂

### 复现挑战

- **三个网络 + adversary 训练**：资源开销 3-4× GAIL
- **仿真器需要可调 physics 参数**：MuJoCo OK，Isaac Gym 也行，但 LeRobot 默认仿真不直接支持

%% end my-thoughts %%

## 🔗 关联笔记

- **B5 同系列**：[[2020-10_RobustIL-NoisyDemos_Tangkaratt]]
- **B6 DRO 系列**：[[2024_ReMix_DRO]], [[2025_DRO_Library]], [[2024_DRO_Theory_Survey]]
- **B3 域随机化**：[[2023_RoboCat]], [[2024_DR_Survey]]
- **VLA 潜在结合**：[[2026_RobustVLA_Guo]]（扩展扰动维度）
- **B7 cross-embodiment 理论联系**：embodiment = dynamics 的一种

## 📌 Action Items

- [ ] 思考：能否把 embodiment 编码为 dynamics 扰动维度
- [ ] 实验：MuJoCo Hopper 复现 RIME，体会三方博弈训练 dynamics
- [ ] 集成：RobustVLA + RIME 联合 worst-case action + worst-case env

%% Import Date: 2026-05-26 %%
