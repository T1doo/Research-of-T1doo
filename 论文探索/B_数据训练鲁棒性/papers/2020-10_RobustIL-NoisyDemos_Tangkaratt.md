---
title: "Robust Imitation Learning from Noisy Demonstrations"
authors: "Voot Tangkaratt, Nontawat Charoenphakdee, Masashi Sugiyama"
year: "2021"
journal: "AISTATS 2021 / arXiv preprint"
doi: "10.48550/arXiv.2010.10181"
arxiv: "2010.10181"
venue: "AISTATS 2021"
citekey: "tangkarattRobustImitationNoisy2021"
itemType: "conference paper"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B5 噪声 demo 鲁棒训练基石"
tags: [literature, T1D, 主线B, B5, 噪声标签, 对称损失, 伪标记, RobustIL]
---

# Robust IL from Noisy Demonstrations — 精读笔记

> [!info] 元信息
> - **作者**：Voot Tangkaratt (RIKEN AIP), Nontawat Charoenphakdee (东京大学), Masashi Sugiyama (RIKEN AIP / 东京大学)
> - **arXiv**：[2010.10181](https://arxiv.org/abs/2010.10181)
> - **代码**：https://github.com/voot-t/ril-co
> - **B5 价值定位**：第一个**系统性证明**从含噪 demo 中学习可以收敛到 expert policy（PAC-style 保证）的工作；与 Symmetric Loss 学习理论紧密相连

## 📄 Abstract

实际收集的 demo 通常含噪——人类专家偶尔出错、亚专家混入数据池、传感器抖动。传统 BC / GAIL 假设 demo 是干净的，遇到噪声会**学到平均策略而非最优策略**。本文提出 **RIL-Co (Robust IL with Co-pseudo-labeling)**：
1. 用 **Symmetric Loss**（AUC-loss）作为分类器损失，证明其对类条件噪声鲁棒
2. 用 **co-pseudo-labeling** 让两个 classifier 互相打标签筛选可信样本
3. 收敛保证：即使噪声率 < 50%，仍能收敛到 expert policy

实验在 MuJoCo Walker / HalfCheetah / Hopper 上：当 demo 中 30-40% 是次优策略时，BC / GAIL 性能崩溃 30-50%，RIL-Co 仅下降 5-10%。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「对称损失天然抗噪」是关键理论桥梁**：
   - 普通 cross-entropy 在标签噪声下偏向多数类，过度自信
   - Symmetric Loss（即 AUC loss / unhinged loss）满足 $\ell(f, +1) + \ell(f, -1) = \text{const}$
   - 这意味着即使部分标签翻转，**期望梯度方向不变**——天然鲁棒
   - 用到 IL 上：把 "expert vs noisy" 看作二分类，用 AUC loss 训练判别器，避免 GAIL 判别器被噪声样本误导

2. **Co-pseudo-labeling 是 small-loss trick 的双网络版**：
   - 训练两个独立 classifier $f_1, f_2$
   - 每个 epoch：$f_1$ 给 $f_2$ 标可信样本（loss 小的），反之亦然
   - 关键好处：避免 self-confirmation bias（单网络会强化自己的偏见）
   - **这与噪声标签领域的 Co-teaching (Han et al. 2018) 是同一思路**——把 NLP / CV 的噪声学习技术搬到 IL

3. **对 VLA 的启示：BC pipeline 全链路含噪**：
   - Demo 含噪（人类失误）
   - 标注含噪（语言标签拼错、跨任务复用）
   - 仿真→真机迁移含噪（动作执行误差）
   - 当前 OpenVLA / π0 假设 OXE 970k demos 全部干净，**这是一个被回避的强假设**

### 方法论（要点）

#### Symmetric Loss 的形式化条件
对于二分类 loss $\ell(f, y)$，若满足
$$
\ell(f(x), +1) + \ell(f(x), -1) = K, \quad \forall x
$$
则该 loss 对**类条件对称噪声**（symmetric label noise）鲁棒。常见例子：
- Unhinged loss：$\ell(z, y) = 1 - yz$
- Sigmoid loss：$\ell(z, y) = \sigma(-yz)$
- Ramp loss

#### RIL-Co 流程
```
1. 初始化两个判别器 f_1, f_2 和一个 policy π
2. 每轮：
   a. 用 π 与环境交互收集 rollout R_π
   b. 从 noisy demo D 和 R_π 联合训练 f_1, f_2（co-pseudo-label）
   c. 用 max(f_1, f_2) 给 R_π 打 reward
   d. PPO / TRPO 更新 π
3. 收敛后 π → expert policy（理论保证）
```

#### 收敛定理（Theorem 1）
设噪声率 $\eta < 0.5$，n 样本数足够大，则 $\hat{f}$ 收敛到 Bayes-optimal $f^*$，从而 RIL-Co 学到的 reward 等价于 expert reward。

### 实验关键数据

#### MuJoCo Walker2d（噪声率 = 0.4）
| Method | Return ↑ |
|---|---|
| Expert demo (clean) | 4500 |
| BC | 1800 (-60%) |
| GAIL | 2100 (-53%) |
| **RIL-Co** | **3900 (-13%)** |

#### Noise 容忍度（HalfCheetah）
- 噪声率 0.1：所有方法相近
- 噪声率 0.3：RIL-Co 显著领先
- 噪声率 0.45：仅 RIL-Co 仍可学习

### 与曦源（EfVLA-鲁棒性）的关联

1. **直接关联：VLA 训练数据天然含噪**
   - OXE 970k demos 跨 22 个机构收集，质量参差不齐
   - LeRobot / SmolVLA 用 community-contributed demos，子专家比例可能高达 20-30%
   - **可以把 RIL-Co 的 co-pseudo-label 思想移植到 VLA fine-tuning**：让两个 VLA 头互相筛选可信 demo
   
2. **与 RobustVLA 对比的价值**：
   - RobustVLA 假设 demo 干净，对 action 做 worst-case 扰动
   - 本文假设 demo 含噪，对 demo 做筛选
   - **两者正交可组合**：RIL-Co 筛 demo → RobustVLA 训 worst-case → 双重鲁棒
   
3. **与 [[2026_RobustVLA_Guo]] 的潜在协同**：
   - RobustVLA 多臂 bandit 决定哪些 demo 训哪种扰动
   - 可结合 RIL-Co：bandit 的 confidence 决定 demo 权重，低 confidence demo 不参与训练
   
4. **针对 SmolVLA 的实操性**：0.5B 模型，可以训两个独立 head 做 co-pseudo-label，显存开销可控

### 待解问题 / 后续追读

1. **从离散 noise 到连续 demo 误差**：本文是「该 demo 是否是 expert」的二分类。VLA 中更现实的是「该 timestep 的 action 偏离最优多少」（连续噪声），需要扩展
2. **OOD demo 检测**：本文假设 noisy demo 来自固定的次优策略；真实场景中 noisy demo 可能来自完全不同的任务（cross-task contamination）
3. **追读队列**：
   - Bayesian Robust Opt for IL (2007.12315)
   - Robust BC with Adv Demo Detection
   - Co-teaching (Han et al. 2018, NeurIPS) — 原 NLP/CV 工作

### 设计风险 / 复现挑战

- **双 discriminator + 双 policy 资源开销 2×**：对 7B OpenVLA 不可行，但 0.5B SmolVLA 可行
- **Co-pseudo-label 的 disagreement metric 需调参**：哪些样本算「可信」依赖 loss threshold，对 task 敏感
- **Symmetric loss 对 deep network 优化困难**：unhinged loss 没有上界，可能梯度爆炸，需配合 gradient clipping

%% end my-thoughts %%

## 🔗 关联笔记

- **B5 同系列**：[[2007.12315_BayesianRobustOpt_IL]], [[2101.01251_RobustMaxEnt_BC]]
- **B6 worst-case 对比**：[[2024_ReMix_DRO]], [[2025_DRO_Library]]
- **理论基础**：Symmetric Loss for Label Noise (Ghosh et al. 2017)
- **VLA 端应用候选**：[[2026_RobustVLA_Guo]]（B1 锚点）, [[2024_OpenVLA]]（潜在 noisy demo 受害者）
- **跨主题协同**：[[A6_shortcut_learning_总结]]（noisy demo 是 shortcut 的成因之一）

## 📌 Action Items

- [ ] 复现 RIL-Co 在 MuJoCo 上的结果（1 周）
- [ ] 在 LIBERO demo 上人工加入 20% 次优 demo，测试 OpenVLA / SmolVLA 性能下降
- [ ] 把 co-pseudo-label 移植到 VLA fine-tuning pipeline（4 周）
- [ ] 与 RobustVLA 联合实验：noisy demo + worst-case action 双重扰动

%% Import Date: 2026-05-26 %%
