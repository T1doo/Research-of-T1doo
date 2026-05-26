---
title: "ReMix: Optimizing Data Mixtures for Language Model Pretraining via Distributionally Robust Optimization"
authors: "Xie et al. (Microsoft / Stanford)"
year: "2024"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2408.14037"
arxiv: "2408.14037"
venue: "arXiv 2024-08"
citekey: "xieReMixOptimizingData2024"
itemType: "preprint"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B6 DRO 数据混合范式"
tags: [literature, T1D, 主线B, B6, DRO, 数据混合, worst-case域, ReMix]
---

# ReMix — DRO 数据混合精读笔记

> [!info] 元信息
> - **arXiv**：[2408.14037](https://arxiv.org/abs/2408.14037)
> - **核心定位**：把 DRO 思想用到**预训练数据混合权重**优化上（不是 action / env，而是 dataset domain）
> - **与 VLA 的关联**：OXE 970k demos 来自 22+ 数据集，混合权重当前是 ad-hoc 决定（如 BridgeData 重采样系数），ReMix 提供原理性 DRO 方案

## 📄 Abstract

LLM 预训练数据通常来自多 domain（Web / Code / Books / Wikipedia），如何配比？传统方法用 grid search 或经验权重。ReMix 提出：把数据混合视作 **min-max 优化**——min 总 loss，max 是 worst-case domain，目标是让模型对**最差表现的下游域**仍有保证。
- 用 group DRO 求解
- 实验：在 7 个 downstream task 上对比固定权重 vs ReMix-DRO
- ReMix 显著改善 worst-domain perplexity，整体均值持平或略优

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「数据混合 = 隐式 curriculum」的 DRO 化**：
   - 传统：固定 $w_i$ 给 domain $i$
   - DRO：动态调 $w_i$，让 $\max_i \ell_i$ 最小
   - 直觉：worst-case domain 自动获得更多权重，类似 hard example mining 的 domain 级版本

2. **min-max-min 三层优化结构**：
   - $\min_\theta \max_{\alpha \in \Delta} \sum_i \alpha_i \mathcal{L}_i(\theta) - \lambda \|\alpha - \alpha_0\|^2$
   - $\alpha$ 是数据 domain 权重，$\alpha_0$ 是 prior（如均匀）
   - 实际用 mirror descent 在 simplex 上更新 $\alpha$

3. **「worst-case domain」的迁移性争议**：
   - 论文证明：DRO 最优权重对 **某些** downstream 鲁棒，但不是所有
   - 如果 downstream 在 prior 已覆盖域内，DRO 通常 win
   - 若 downstream 完全 OOD，DRO 没保证

### 方法论（Group DRO 形式）

```
For each step t:
  1. 用当前 α_t 采样 batch
  2. 计算每个 domain i 的 loss L_i
  3. 更新 α: α_{t+1}[i] ∝ α_t[i] * exp(η * L_i)  # mirror descent
  4. 投影到 simplex Δ
  5. 用 α_{t+1} 加权梯度更新 θ
```

复杂度：每个 domain 维护 separate loss tracker，额外开销 O(D) 内存。

### 实验

#### LLaMA-1B 预训练（7 个 domain：Web / Code / Books / Math / Wiki / News / Dialog）
| Method | Avg PPL ↓ | Worst PPL ↓ |
|---|---|---|
| Uniform mixing | 18.2 | 32.5 (Math) |
| Hand-tuned | 17.6 | 28.1 |
| **ReMix-DRO** | **17.4** | **24.3** |

#### Downstream (MMLU, GSM8K, HumanEval)
- 整体均值持平 / +0.5pp
- 最差 task 提升 +3pp

### 与曦源的关联（重点）

1. **OXE 数据混合的 DRO 化**：
   - OXE 970k demos 跨 22 数据集，当前 OpenVLA 用经验权重（BridgeData 2×、Fractal 1×…）
   - **可以用 ReMix 思想自动决定每个数据集权重**——让 VLA 对 worst-case dataset 仍鲁棒
   - 这对 LeRobot / SmolVLA 也适用：community 上传的多 dataset 怎么混

2. **跨 embodiment 训练的 DRO 视角**：
   - 不同 embodiment（Franka / xArm / WidowX）可以视作不同 domain
   - 用 group DRO 平衡各 embodiment 的贡献
   - **与 [[2024_CrossFormer]] / [[2024_HPT]] 形成方法论互补**：他们解决"怎么把多 embodiment 编码到一个模型"，DRO 解决"训练时怎么权衡各 embodiment 数据"

3. **与 RobustVLA 多臂 bandit 的对比**：
   - RobustVLA 多臂 bandit：选哪种扰动训
   - ReMix DRO：选哪种 domain 训
   - **本质都是 weighted sampling，可以统一在 DRO 框架下**
   - 这是个潜在创新点：**DRO-unified Robust VLA Training**——既选扰动也选 domain
   
4. **理论保证的吸引力**：
   - RobustVLA 的多臂 bandit 是启发式
   - DRO 有 regret bound（O(√T)）
   - **如果用 DRO 替换 bandit，可以写一段理论分析增强论文厚度**

### 待解问题

1. **DRO 在巨大 domain 数下的可扩展性**：22 个数据集还 OK，如果 LeRobot 上百 dataset 怎么办？需要 group-of-groups DRO
2. **如何定义 domain**：dataset？task？embodiment？三种切法都合理但结果不同
3. **DRO 对 outlier domain 的过度敏感**：如果一个 dataset 极差，DRO 会过度倾斜其权重，可能伤总体

### 复现挑战

- **维护 per-domain loss tracker**：实现简单但需要修改 dataloader
- **mirror descent 的 learning rate**：太大震荡，太小不动，需调
- **与 PyTorch FSDP / DeepSpeed 集成**：分布式训练下 per-domain loss 同步成本

%% end my-thoughts %%

## 🔗 关联笔记

- **B6 DRO 三件套**：[[2024_ReMix_DRO]], [[2025_DRO_Library]], [[2024_DRO_Theory_Survey]]
- **B7 cross-embodiment 数据**：[[2024_OpenX-Embodiment]], [[2024_CrossFormer]], [[2024_OpenVLA]]
- **B5 noisy demo**：[[2020-10_RobustIL-NoisyDemos_Tangkaratt]]
- **RobustVLA 协同**：[[2026_RobustVLA_Guo]]（替换 bandit 为 DRO）

## 📌 Action Items

- [ ] 实验：在 LIBERO 4 个子集（Spatial/Object/Goal/Long）上跑 group DRO 混合，对比均匀混合
- [ ] 理论：把 RobustVLA bandit 改写为 DRO，写一段对比
- [ ] 扩展：OXE 22 数据集做 DRO 加权，与 OpenVLA 经验权重对比

%% Import Date: 2026-05-26 %%
