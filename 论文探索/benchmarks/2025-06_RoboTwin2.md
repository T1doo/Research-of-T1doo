---
title: "RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Manipulation"
authors: "Tianxing Chen et al. (26 authors)"
year: "2025"
venue: "arXiv preprint (2025-06)"
arxiv: "2506.18088"
status: "已精读"
tier: "⭐⭐ 多机体 DR 平台"
tags: [literature, T1D, benchmark, robustness-eval, multi-embodiment, domain-randomization, bimanual]
---

# RoboTwin 2.0 — 5 机体 × 50 任务 × 5 维 DR

> [!info] 元信息
> - **作者**：Tianxing Chen 等 26 人
> - **arXiv**：[2506.18088](https://arxiv.org/abs/2506.18088)
> - **GitHub**：[robotwin-Platform/robotwin](https://github.com/robotwin-Platform/robotwin)（MIT，⭐2.4k）
> - **数据集规模**：731 个对象 × 147 类别 + 100K+ 预收集轨迹
> - **核心定位**：**双臂操控的 sim 平台 + DR benchmark**

## 📄 这个 benchmark 是做什么的

为双臂 VLA 训练 + 评测设计的 sim 平台。强调 **5 维 Domain Randomization (DR)** 和**多机体支持**，目标是「合成数据 + 10 个真演示 → 击败 10 演示 baseline 367% 相对增益」。

5 维 DR：
1. **Clutter** — 杂乱物体
2. **Lighting** — 光照
3. **Background** — 背景
4. **Tabletop height** — 桌面高度
5. **Language** — 指令变化

5 个 embodiments：论文未具名列表（AgileX 是已知合作方，应包含其双臂机器人系列）。

## 🧠 我的评估

### 评测维度
- **5 维 DR**——比 Colosseum 12 维少，但加入「**tabletop height**」（物理特有）和「**language**」（少见）
- 与 LIBERO-Plus 对比：维度覆盖相近（lighting/bg/lang）但 LIBERO-Plus 任务单臂，RoboTwin 双臂
- 与 VLA-Risk 对比：VLA-Risk 测攻击 ASR，RoboTwin 测合成数据训练增益

### 任务规模
- **50 tasks**（双臂操控）
- **731 instances / 147 categories** 对象库
- **100K+ 预收集轨迹**（HuggingFace）

### 评估 baseline 模型
- **DP, ACT, DP3, RDT, PI0** 以及多个 VLA 变体
- baseline 广泛但**非 OpenVLA 主流系**——双臂 VLA 较新

### 关键 finding
- **代码生成 SR +10.9%**
- **VLA 合成数据 + 10 真演示 vs 10 演示 baseline → 367% 相对增益**
- **零样本（仅合成）vs 10 演示 → 228% 增益**
- 含义：**合成数据 DR 是双臂 VLA 训练的高 ROI 路线**

### 复现成本
- **GPU**：双臂仿真比单臂重，估计 24-48GB 显存训练
- **仿真器**：自研双臂 sim（未明确底层物理引擎）
- **安装**：~20 min
- **评测时间**：50 tasks × 多机体，**全套估计 2-3 天/GPU**
- **开源**：✅ MIT，HuggingFace 提供数据

### 是否适合曦源课题
- ❌ **不推荐**
- 理由：
  1. **双臂任务**——曦源若做 FocusVLA 改进，FocusVLA 是单臂，不直接对接
  2. **5 个 embodiment 协调复杂**——增加变量难调试
  3. 100K+ 数据库**显存压力**——若做训练侧实验，46GB 紧张
  4. 主流单臂 VLA 在此 benchmark 上 baseline 不丰富
- ✅ 仅适合：**做 cross-embodiment 鲁棒性**专题 → 否则 LIBERO 谱系更合适

## 🔗 关联
- [[2024-02_Colosseum]]：两者都做 DR 评测，RoboTwin 多机体 / Colosseum 单机
- [[2025-10_RoboChallenge]]：RoboChallenge 真机 30 任务 / RoboTwin 仿真 50 任务
- [[VLA-Risk_ICLR2026]]：VLA-Risk 不含多机体
