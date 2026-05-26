---
title: "Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models"
authors: "Wang et al."
year: "2024"
venue: "arXiv 2411.13587"
arxiv: "2411.13587"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 攻击开创"
tags: [literature, T1D, 主线B, 对抗训练, 嵌入扰动, EDPA, 早期工作]
---

# Exploring the Adversarial Vulnerabilities of VLAs — EDPA 嵌入扰动攻击

> [!info] 元信息
> - **作者**：Wang et al.（多机构）
> - **arXiv**：[2411.13587](https://arxiv.org/abs/2411.13587)
> - **日期**：2024-11
> - **核心定位**：**最早系统性研究 VLA 对抗脆弱性的论文之一**；提出 EDPA (Embedding-Disruption Patch Attack)
> - **本地 PDF**：未缓存

## 📄 TL;DR

论文首次系统揭示 VLA 在白盒对抗扰动下的脆弱性，提出 **EDPA (Embedding Disruption Patch Attack)** —— 不是直接最大化任务失败损失，而是 **最大化扰动样本与原始样本在 VLM 嵌入空间的距离**。这种"嵌入空间攻击"利用了 VLA backbone (CLIP / SigLIP) 已知的对抗脆弱性，比直接 task-loss 攻击更高效（节省 ~60% PGD 步数）。在 OpenVLA、RT-2 上：EDPA 平均 ASR 76% (ε=4/255)，task-loss PGD 仅 52%。同时给出基线防御：embedding-aware adversarial training，可把 ASR 压回 31%。论文是 VLA 安全方向的**早期开创性工作**，给后续 RobustVLA / VLA-Fool 提供了攻击工具箱。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「嵌入空间攻击」比「任务空间攻击」更高效**：EDPA 不优化「让 action 错」，而是优化「让 VLM embedding 漂移」。因为 action head 是 embedding 的下游，embedding 一漂 action 自然错。这种间接攻击更高效——**避开 VLA 黑盒 action 推理的梯度估计噪声**。这是 VLA 安全的一个根本洞察。

2. **VLA 的脆弱性继承自上游 VLM**：CLIP / SigLIP 已有大量对抗工作（Co et al. 2023、Mao et al. 2023），EDPA 证明这些 VLM 的脆弱性**直接传导到 action**。意味着 VLA 的对抗鲁棒性研究**必须从 VLM 鲁棒化开始**——这给"基础模型层" robustness 留了大空间。

3. **白盒假设的现实意义被低估**：作者论证白盒攻击在开源 VLA 时代（OpenVLA / Octo / π0 全部开源权重）是 **完全现实的**，攻击者可在自己 GPU 上 PGD，再把 patch 物理打印贴到环境里。

### 方法论

- **EDPA 形式化**：
  $$\max_{\delta} \|\phi_{VLM}(o + \delta) - \phi_{VLM}(o)\|_2 \quad \text{s.t. } \|\delta\|_\infty \le \epsilon$$
  - 其中 φ_VLM 是 backbone 视觉编码器输出
  - 用 PGD-10 step
- **Patch 攻击变体**：把 δ 限制到固定大小 patch (如 64×64) 而非全图，便于物理打印
- **基线防御**：embedding-aware adversarial training
  $$\min_\theta \max_\delta \mathcal{L}_{task}(\theta, o+\delta) + \lambda \|\phi(o+\delta) - \phi(o)\|^2$$
  - 防御目标：让 embedding 在扰动下保持稳定

### 实验关键数据

| 攻击 | OpenVLA ASR (ε=4/255) | RT-2 ASR | 计算开销 (PGD 步数) |
|---|---|---|---|
| Task-loss PGD | 51.8% | 47.3% | 50 step |
| **EDPA** | **76.4%** | **71.2%** | 20 step |
| EDPA patch (64×64) | 68.7% | 63.5% | 20 step |

防御后：embedding-aware adv training 将 EDPA ASR 从 76% → 31%，但 clean SR 下降 4-6 pp。

### 与 [[2026_RobustVLA_Guo]] 的差异（B1 类必答）

| 维度 | AdvVLA Vuln (EDPA) | RobustVLA |
|---|---|---|
| 时间线 | **2024-11（最早）** | 2026-02 |
| 攻击类型 | 单一 visual PGD / patch | 17 类多模态扰动 |
| 攻击新颖性 | **嵌入空间损失**首创 | 仍用 task-loss |
| 防御 | 基础 adversarial training | 多臂 bandit 调度 + worst-case action |
| 关系 | **RobustVLA 的早期前驱**——证明了 VLA 对抗脆弱性的存在 |

→ **EDPA 的嵌入空间损失是 RobustVLA 没涉及的攻击形式**。RobustVLA 的 input-robust 模块只对 task-loss-equivalent 扰动鲁棒，未必能防 EDPA 这种 embedding-level 攻击。**这是你研究的延伸点**：把 RobustVLA 的 worst-case δ 优化目标从 task-loss 替换 / 增强为 embedding distance，可能进一步提升鲁棒性。

### 与曦源关联

1. **EDPA 给曦源的「攻击工具箱」**：评测 SmolVLA / EfVLA 的鲁棒性时，EDPA 应作为标配攻击（与 PGD 一并报）。比单独 PGD 更能暴露视觉编码器的脆弱性。
2. **小模型可能对 embedding-level 攻击更脆弱**：SmolVLA 用 SmolVLM 做 backbone，其鲁棒性远不如大 CLIP；曦源应专门测 EDPA 对 SmolVLA 的破坏力。
3. **embedding-aware defense 思想可复用**：曦源若设计鲁棒训练 loss，应在 task loss 外加 embedding 稳定 loss——这是从 EDPA 直接借来的设计点。

### 待解问题

1. **EDPA 在 patch 物理打印后实际效果**：仿真 ASR 76% 转换到真机有多少 retention？论文未做。
2. **EDPA vs CTA (VLA-Fool)**：embedding-level 单模态攻击 vs cross-modal 联合攻击，哪个对部署威胁更大？
3. **代码 / 攻击 patch 是否开源**：如开源可直接接入曦源评测。

%% end my-thoughts %%

## 🔗 关联笔记

- **直接对比**：[[2026_RobustVLA_Guo]]（被 EDPA 攻击的对象）
- **后继扩展**：[[2025_VLA-Fool]]（cross-modal 攻击）、[[2025_Model-Agnostic_AdvDef]]
- **同类早期攻击**：[[2024_DPA_Diffusion_Policy_Attacker]]、[[2024_PG-Attack]]
- **基础脆弱性**：CLIP / SigLIP 对抗工作（Co 2023、Mao 2023）
- **被攻击对象**：[[2024_OpenVLA]]、[[2023_RT-2]]
- **评测平台**：[[2025_Eva-VLA]]
