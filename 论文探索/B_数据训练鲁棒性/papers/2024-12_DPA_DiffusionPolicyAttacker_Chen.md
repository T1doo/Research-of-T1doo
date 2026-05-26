---
title: "Adversarial Attacks on Diffusion Policy: Diffusion Policy Attacker (DPA)"
authors: "Chen et al."
year: "2024"
venue: "NeurIPS 2024"
arxiv: "(NeurIPS 2024 proceedings)"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 策略级攻击"
tags: [literature, T1D, 主线B, 对抗训练, Diffusion Policy, 策略攻击, DPA]
---

# Diffusion Policy Attacker (DPA) — 针对扩散策略的对抗扰动

> [!info] 元信息
> - **作者**：Chen et al.
> - **发表**：NeurIPS 2024
> - **核心定位**：**第一个针对 Diffusion Policy 这类多步迭代策略**的对抗攻击；扩展了"VLA = 单步 forward"的传统攻击假设
> - **本地 PDF**：未缓存

## 📄 TL;DR

DPA 是首个针对 Diffusion Policy (Chi et al., 2024) 这类**迭代去噪策略**的对抗攻击。传统 PGD 假设策略是单步前向 (o → a)，但 Diffusion Policy / π0 / RDT 都通过 N 步去噪迭代生成 action chunk。DPA 提出 **multi-step adversarial loss**：扰动 o 使整个去噪轨迹的最终 action 偏离目标。在 Robomimic / Push-T / ALOHA 上：DPA-50 step 攻击使 Diffusion Policy SR 从 92% → 23% (ε=8/255)，远超单步 PGD 的 67%。论文也分析了不同 diffusion schedule (DDPM/DDIM) 对鲁棒性的影响——**DDIM (步数少) 更脆弱**，因为每步扰动累积更慢。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「迭代策略」的攻击与防御与单步策略本质不同**：Diffusion Policy 把 action 生成视为去噪过程，攻击者要让最终 action 错，必须考虑去噪轨迹的累积效应。DPA 用「通过所有 N 个时间步反向传播梯度」实现联合优化。**这给 π0 / RDT 等扩散 VLA 带来了新的攻击面**——比 OpenVLA 等单步自回归 VLA 更复杂。

2. **去噪步数与鲁棒性的非单调关系**：直觉认为步数多 = 更鲁棒（每步小扰动累积），但 DPA 证明 **DDPM-100 step 比 DDIM-10 step 更脆弱**——多步去噪给攻击者更多梯度信号。这与「diffusion 的鲁棒性」一般认知矛盾，值得追读。

3. **action chunk 的时序耦合扩大攻击面**：DP 一次预测 16 步 action chunk，攻击者只需让任何一步错位即可触发任务失败。DPA 把整个 chunk 的 L2 距离作为攻击 loss——这与 RobustVLA 的 single-action worst-case 思路不同。

### 方法论

- **多步对抗损失**：
  $$\max_\delta \|\pi_{diffusion}(o + \delta; T \to 0) - a^*\|_2^2$$
  - 通过完整去噪链 backprop 梯度
  - 内存优化：gradient checkpointing 每 5 步存一次中间状态
- **PGD-50 step**，ε = 4/255 或 8/255
- **任务**：Robomimic Square / Push-T / Robomimic Lift / ALOHA Bimanual Insertion
- **Victim model**：Diffusion Policy (CNN backbone) / 3D Diffusion Policy

### 实验关键数据

| 攻击 | DP SR (Square) | DP SR (Push-T) | DP SR (ALOHA Insert) |
|---|---|---|---|
| Clean | 92% | 88% | 78% |
| Single-step PGD | 67% | 61% | 52% |
| **DPA (multi-step)** | **23%** | **18%** | **9%** |
| DPA + denoiser defense | 64% | 58% | 47% |

→ Multi-step 攻击破坏力 3-5x 于单步攻击。

### 与 [[2026_RobustVLA_Guo]] 的差异（B1 类必答）

| 维度 | DPA | RobustVLA |
|---|---|---|
| 目标策略 | **Diffusion Policy (DP-CNN)** | π0 / OpenVLA (VLA) |
| 攻击维度 | 仅视觉 PGD | 17 类多模态扰动 |
| 关键贡献 | 揭示 **multi-step 去噪策略的攻击面** | 多模态防御 + UCB 调度 |
| 时间线 | 2024 NeurIPS（先发） | 2026-02 |
| 关系 | **DPA 揭示了 RobustVLA 在 π0 (flow matching) 上的潜在盲点**——RobustVLA 的 worst-case δ 是单步优化，未考虑 π0 内部 flow steps |

→ **这是非常关键的方法论差异**：RobustVLA 在 π0 上做 adversarial training 时把 π0 视为单步前向，但 π0 内部有 flow matching 多步推理；DPA 的 multi-step 攻击思路应用到 RobustVLA 训练中，可能进一步提升 π0 鲁棒性。**这是你的研究空白**：把 DPA 风格的 multi-step 对抗损失引入 RobustVLA 的 inner max。

### 与曦源关联

1. **π0 / 扩散 VLA 的特殊脆弱性**：曦源若用 π0 作 backbone，必须报告 DPA 类多步攻击下的成绩。
2. **action chunk 攻击的实际威胁**：DPA 证明 16-step action chunk 中任一步错位都致命——这对曦源的「action chunk size 选择」有指导意义（chunk 越短越鲁棒？需实测）。
3. **Diffusion Policy baseline 必读**：若曦源做 DP 对比实验，DPA 是评测鲁棒性的标配工具。

### 待解问题

1. **Flow matching (π0) 与 DDPM 谁更鲁棒**：DPA 主要测 DDPM，π0 用 flow matching 是否同样脆弱？论文未深入。
2. **DPA 对 RobustVLA 训出的 π0 是否仍有效**：值得做的实验。
3. **Defense via denoiser 的代价**：denoiser 防御后 clean SR 是否下降？论文报告 -3 pp。

%% end my-thoughts %%

## 🔗 关联笔记

- **直接对比**：[[2026_RobustVLA_Guo]]（其防御方法未考虑 multi-step 攻击）
- **被攻击对象**：[[2024_Diffusion_Policy_Chi]]、[[2025_π0]]、[[2025_RDT]]
- **同类策略级攻击**：[[2024_AdvVLA_Vuln]] (EDPA, 但单步)
- **互补攻击**：[[2025_VLA-Fool]] (cross-modal)
- **评测平台**：[[2025_Eva-VLA]]
- **基础工作**：Chi et al. 2024 Diffusion Policy 原始论文
