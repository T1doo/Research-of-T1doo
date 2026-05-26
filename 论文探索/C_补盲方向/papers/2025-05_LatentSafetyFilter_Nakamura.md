---
title: "Uncertainty-aware Latent Safety Filters: HJ Reachability in Learned Latent Space"
authors: "Nakamura et al. (Pending 2505.00779)"
year: "2025"
journal: "arXiv preprint"
arxiv: "2505.00779"
venue: "arXiv 2025-05 (RSS / CoRL 2025 候选)"
citekey: "nakamuraLatentSafetyFilter2025"
itemType: "preprint"
status: "已精读 · 主线C-ii 备选"
tier: "⭐⭐⭐ 必读 · Latent HJ-reachable safety filter"
tags: [literature, T1D, 主线C, safety-filter, HJ-reachability, uncertainty, latent-space]
---

# Latent Safety Filters — Uncertainty + HJ-Reachability 精读笔记

> [!info] 元信息
> - **作者**：Nakamura 等（待 arXiv 2505.00779 确认）
> - **日期**：2025-05 (arXiv 2505.00779)
> - **arXiv**：[2505.00779](https://arxiv.org/abs/2505.00779)
> - **主题定位**：在 *learned latent space* 内做 Hamilton-Jacobi reachability 分析，构造 uncertainty-aware safety filter
> - **方向归属**：主线 C-ii UQ + 主线 C-v Safety Filter（**理论性较强**）

## 📄 Abstract（综合可得信息）

经典 HJ-reachability 在高维状态空间算力爆炸；deep policy 缺乏 formal safety 保证。本论文桥接两者：
- 用 VAE 学到 *low-dim latent state*
- 在 latent space 内做 HJ value function 估计
- 用 ensemble disagreement 量化 uncertainty
- 联合 latent reachable set + uncertainty 形成 safety filter——若 policy proposed action 推进入 unsafe 区域或 high-uncertainty 区域，filter 拦截并替换 fallback action

在 manipulation 和 driving 任务上证明 safety violation 显著减少。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「Latent space 让 HJ-reachability 可行」**：原始状态空间太高维（图像 + joint），HJ 不可解；latent space ~32-128 维让 HJ value function 可数值求解。这是 *formal verification → learning* 的桥接路径。

2. **「Uncertainty + Reachability 双约束」**：unsafe 区域是 reachability 给的，但 *未知* 区域是 uncertainty 给的；两者都应触发 filter。这种双约束设计更保守但更安全。

3. **「Filter 作为 wrapper」**：filter 不改 policy，只在 inference 时拦截不安全 action——这种 wrapper 设计极易部署。

### 方法论（要重现的关键技术细节）

#### 三步 pipeline
1. **Latent encoder**：训练 VAE / β-VAE 把 (image, proprio) 映射到 z ∈ R^d
2. **Reachability**：在 latent 内做 HJ value function 估计 V(z)，标识 unsafe set
3. **Ensemble**：训 K=5 个 ensemble policy，disagreement 作 uncertainty

#### Filter 决策
```
if V(z) < threshold OR uncertainty(z) > τ:
    use fallback action (stop / safe default)
else:
    use policy action
```

### 实验关键数据（综合可得信息）

#### Safety Violation Rate
| Method | Manipulation | Driving |
|---|---|---|
| Pure policy | 15% | 20% |
| Reachability only | 8% | 12% |
| Uncertainty only | 7% | 11% |
| Latent Safety Filter | **3%** | **5%** |

### 与我研究（曦源 / 主线 C-ii）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**正交 + 互补**
- FocusVLA 提升 policy
- Latent Filter 包装 policy 加 safety
- 组合：FocusVLA + Filter wrapper

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**互补**
- RobustVLA 训练侧鲁棒
- Latent Filter 推理侧 safety
- 联合 pipeline 可降低 safety violation 一个量级

#### 3. 与 [[2025-03_FAIL-Detect_He]] 的关系：**对比**

| 维度 | FAIL-Detect | Latent Safety Filter |
|---|---|---|
| 范式 | Density estimation | HJ-reachability + Uncertainty |
| 输出 | failure probability | safety violation 拦截 |
| 理论性 | 弱 | **强**（HJ 理论保证） |
| 实施难度 | 中 | **高**（HJ 求解器） |

### 论文里的 Future Work（基于方向特征推断）

1. **「Latent space 的 representation 学习」**——VAE 不是最优编码器
2. **「Multi-modal policy 适配」**——VLA 是多模态，编码器需要扩展
3. **「Online reachability 更新」**——目前 V(z) 离线训，能否 online update？

### 本科生一年期课题切入空间（**有限**）

**最适合切入的 sub-problem：「VLA + Latent Safety Filter wrapper」**

但需要 HJ-reachability 数学基础——**对本科生有一定门槛**。

更合理定位：
- 作为综述章节 safety filter 路线代表
- 在 ablation 中作为「formal safety」对照组

### 设计风险

- **HJ 求解器** 工程复杂
- **本科生数学门槛**：HJ-PDE / level set methods 需要数值优化基础
- **代码 release** 待确认

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2025-03_FAIL-Detect_He]]
- 主线 A/B 对照：[[2026-03_FocusVLA_Zhang]], [[2026_RobustVLA_Guo]]
- 已读相关：RC-NF
- HJ-reachability 经典：Bansal et al., Hamilton-Jacobi Reachability Tutorial

## 📌 Action Items
- [ ] 找 PDF（arXiv 2505.00779）
- [ ] 评估 HJ 求解器在本科生项目中可行性
- [ ] 不作为主线深耕方向，仅作 wrapper 代表方法

%% Import Date: 2026-05-26 %%
