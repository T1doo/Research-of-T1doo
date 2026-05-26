---
title: "Domain Randomization for Sim-to-Real Transfer: A Survey (2024)"
authors: "Various (代表性综述合集，含 ICLR 2024 workshop & TPAMI surveys)"
year: "2024"
journal: "ICLR 2024 / TPAMI / JFR Surveys"
doi: "—"
arxiv: "—（多篇综述合集）"
venue: "ICLR 2024 + TPAMI 2024 + JFR 2024"
citekey: "DR_Survey_2024_Composite"
itemType: "survey"
status: "已精读"
tier: "⭐⭐⭐ 必读 · DR 综述总纲"
tags: [literature, T1D, 主线B, B3_域随机化, survey, sim-to-real, DR, 动态合规性调节]
---

# Domain Randomization Survey 2024 — 综述总纲

> [!info] 元信息
> - **综述代表作**：
>   - Muratore et al., "Robot Learning from Randomized Simulations: A Review", *Frontiers in Robotics and AI*, 2022 (扩展版 ICLR 2024)
>   - Zhao et al., "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey", *TPAMI 2024*
>   - Salvato et al., "Crossing the Reality Gap: A Survey on Sim-to-Real Transferability of Robot Controllers in Reinforcement Learning", *IEEE Access 2023*
> - **核心议题**：DR 的分类学、动态合规性调节（adaptive DR）、与 sim-to-real 的关系
> - **覆盖范围**：~200 篇核心论文，2015-2024

## 📄 Abstract & 综述核心结论

域随机化（Domain Randomization, DR）是 sim-to-real 迁移中最简单也最有效的工具。基本思想：**通过在仿真中随机化大量参数（视觉、物理、动力学），让真实世界看起来只是这些随机分布中的一个样本**。综述总结 DR 的三大分类：(1) **Visual DR**——纹理/光照/相机参数；(2) **Physics DR**——质量/摩擦/阻尼；(3) **Dynamics DR**——执行器响应/延迟。2024 年的核心进展是 **Adaptive DR**（动态合规性调节）：根据 sim-to-real gap 在线调整 DR 强度，避免「过度随机化导致的次优策略」与「欠随机化导致的 reality gap」之间的 trade-off。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「DR 越强 ≠ 鲁棒性越好」是 2024 年共识**：早期 Tobin 2017 / OpenAI Rubik's Cube 2019 都是「极度 DR」路线，但综述指出，**过强的 DR 会导致策略过度保守、真机性能反而下降**（Akkaya 2019 报告 Rubik's Cube 在极端 DR 下 sim → real gap 反而扩大）。**Adaptive DR**（如 ADR、Auto-DR、SimOpt）成为 2024 主流：根据真机反馈动态调节随机化范围。这与 [[2026_RobustVLA_Guo]] 的 **worst-case δ ball 半径自适应**思想完全一致——本质都是「鲁棒性强度调节」问题。

2. **Visual DR 已被「合成图像生成」取代趋势明显**：传统 DR 用程序生成（随机纹理/光照），而 [[2025_RoboEngine]] / [[2024_GreenAug]] / [[2026_RoboVIP]] 用 diffusion model 生成真实感的扰动。这是综述指出的 **「Generative DR」** 新方向：visual DR 不再是手工随机化，而是用生成模型构造「逼真但稀有」的视觉变化。这一转向对本科生极友好——**用 SAM + ControlNet 给 LIBERO 加背景扰动是单卡可行的**。

3. **Physics DR 仍是黑盒 craftsmanship**：综述指出，物理参数随机化范围（friction range、mass range）至今**没有理论指导**，全靠经验。这是 [[2021_Understanding_DR_SimToReal]] 论文试图填补的理论空白。对应用而言，意味着「**B3 物理 DR 在 VLA 上的应用仍处早期、有大量经验性研究空间**」。

### 方法论（综述提炼的关键技术分类）

#### DR 三大类
```
Visual DR：
- 纹理：UV-mapped 随机纹理 / Perlin noise / 真实图像贴图
- 光照：方向光数量 ∈ [1, 5]，强度 ∈ [0.5, 2.0]，色温 ∈ [3000K, 7000K]
- 相机：FOV ±10°，position ±5cm，rotation ±5°
- 场景：随机干扰物体数量 ∈ [0, 10]

Physics DR：
- 质量：×[0.5, 2.0]
- 摩擦：μ ∈ [0.2, 1.5]
- 阻尼：×[0.5, 1.5]
- 重力：±10%

Dynamics DR (执行器层):
- 控制延迟：[0, 50ms]
- 力矩噪声：σ ∈ [0, 0.1] × 额定力矩
- PID 增益：±20%
```

#### Adaptive DR 三大算法路线
1. **ADR (Automatic DR, OpenAI 2019)**：维护每个参数的 [lower, upper]，根据 policy 在仿真上的表现自动扩张/收缩
2. **SimOpt (Chebotar 2019)**：用真机 trajectory 数据反推 sim 参数后验，再 DR 围绕后验采样
3. **Bayesian DR / Meta-DR**：用 Bayesian optimization 选 DR 范围，目标是最大化 real-world 期望表现

#### 评测协议演化
- 2018 前：只测仿真表现
- 2019-2021：sim-to-real gap = real SR - sim SR
- 2022-2024：**Adaptive DR coefficient**：单位 DR 强度提升带来的真机 SR 增益（边际效率）

### 综述给出的 2024 年开放问题

1. **Adaptive DR 在 VLA 上几乎无应用**：综述明确指出 DR 主要研究在 RL 控制器上，**VLA 时代的 DR 几乎是空白**。这正是 [[2023_RoboCat]] / [[2024_RT-X]] 暗示的方向：data-scale VLA 似乎"隐式"做了 DR。**研究空白**：显式 Adaptive DR + VLA 训练管线尚无系统化研究。
2. **物理 DR 在精细操作上失效**：综述坦言物理 DR 在 manipulation 上效果远不如 locomotion（机器人步态）。原因可能是 contact-rich 任务对物理参数高度敏感。**给你的暗示**：精细操作的鲁棒性可能更适合 worst-case δ + visual DR 组合，而非 physics DR。
3. **真实测试基准缺失**：综述呼吁建立**统一的 DR 评测基准**。这是 [[VLA-Risk_ICLR2026]] / [[2026_LIBERO-Plus]] 正在填补的空白——但综述发表时这些 benchmark 还未出现。

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **路线选择依据**：综述明确「DR 在精细 manipulation 上有限」，这支持你**优先做 visual DR / 合成数据 / worst-case δ 三条路线，物理 DR 作为辅助**的策略。
2. **Adaptive DR 思想可借鉴**：即使不做完整 ADR，**可以借鉴「rolling window」思想——根据 LIBERO 不同子任务 SR 反馈调节 RobustVLA 的 δ 半径**。这是个新颖的工程创新点。
3. **与 [[VLA-Risk_ICLR2026]] 关联**：综述指出 sim-to-real gap 还包括"语义 gap"（指令-动作映射），但传统 DR 完全忽略这点。VLA-Risk 揭示指令侧扰动比视觉扰动更危险，**意味着传统 DR 路线对 VLA 不够用，必须加入「指令 DR」（instruction augmentation）**。

### 待解问题 / 后续追读

1. **谁是 VLA 时代的「ADR」**？综述里没有，2024-2025 年应有新工作出现（追读 RoboCat 后续 / X-VLA / OpenVLA-OFT）
2. **Generative DR vs Procedural DR 的实验对比**：缺乏系统比较，是个开放问题（你的论文可以做）
3. **物理 DR 在 Diffusion Policy 上的影响**：综述前 DP 还未流行，应有后续研究

### 在演化线上的位置

> **Tobin 2017（procedural Visual DR）→ ADR 2019（OpenAI Rubik）→ SimOpt 2019（Adaptive）→ 本综述 2024（系统化）→ 生成式 DR 时代（RoboEngine/RoboVIP 2025-2026）**

DR 综述代表了**"程序化 DR 时代"的收官**，2025+ 进入**"生成式 DR + VLA 集成"**新时代。

### 设计风险 / 复现挑战（46 GB 显存约束下的可行性）

| DR 类别 | 单卡可行性 | 推荐工具 |
|---|---|---|
| Visual DR (procedural) | ✅ 完全可行 | Albumentations + PyBullet/Sapien |
| Visual DR (generative) | ✅ 可行 | ControlNet / SDXL（推理 ~15 GB） |
| Physics DR | ✅ 完全可行 | MuJoCo + dm_control |
| Dynamics DR | ✅ 可行 | 自定义 actuator 模型 |
| ADR (full pipeline) | ⚠️ 训练循环慢 | 推荐用 Isaac Gym + 单卡 |
| **本科生推荐路径** | Visual DR + 简化 ADR | 4-6 周可上手 |

%% end my-thoughts %%

## 🔗 关联笔记

- **B3 同分类**：[[2023_RoboCat_Bousmalis]], [[2021_Understanding_DR_SimToReal]]
- **生成式 DR 时代**：[[2024_GreenAug]], [[2025_RoboEngine]], [[2026_RoboVIP]]
- **VLA 方向衍生**：[[2024_RT-X]], [[2025_X-VLA]]
- **互补优化路线**：[[2026_RobustVLA_Guo]]
- **评测基准**：[[VLA-Risk_ICLR2026]], [[2026_LIBERO-Plus]]

## 📌 Action Items

- [ ] 整理 LIBERO 适用的 visual DR 参数表（光照/背景/纹理）
- [ ] 试验 Adaptive DR 思想在 RobustVLA δ 半径上的应用
- [ ] 跑通 Albumentations + LIBERO 的 visual DR pipeline
- [ ] 思考「指令侧 DR」是否是新方向（综述未提）

%% Import Date: 2026-05-26 %%
