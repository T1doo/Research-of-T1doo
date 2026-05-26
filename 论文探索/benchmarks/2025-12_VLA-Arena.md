---
title: "VLA-Arena: A Comprehensive Benchmark Platform for Vision-Language-Action Models"
authors: "Borong Zhang et al. (9 authors)"
year: "2025"
venue: "arXiv preprint (2025-12)"
arxiv: "2512.22539"
status: "已精读"
tier: "⭐⭐⭐ 综合平台 TOP-1 候选"
tags: [literature, T1D, benchmark, robustness-eval, arena, evaluation-platform]
---

# VLA-Arena — 综合评测平台（170 tasks × 3 难度 × 双扰动轴）

> [!info] 元信息
> - **作者**：Borong Zhang 等 9 人
> - **arXiv**：[2512.22539](https://arxiv.org/abs/2512.22539)
> - **数据集规模**：170 tasks × 3 难度 × W0-W4 / V0-V4
> - **rolling leaderboard**：✅ 论文承诺端到端工具链 + leaderboard
> - **协议**：CC BY-NC-SA 4.0

## 📄 这个 benchmark 是做什么的

定位为「VLA 评测的综合 arena」，按四类能力切分 170 项任务，每项设计三个难度，再叠加双轴扰动（语言 W0-W4，视觉 V0-V4），实现**解耦的鲁棒性分析**——单维度可控、多维度可叠。

四大能力维度：
1. **Safety** — 安全约束
2. **Distractor** — 干扰物
3. **Extrapolation** — 外推能力
4. **Long Horizon** — 长程任务

难度层（每任务 3 层）：
- **L0**：训练分布，用于 fine-tuning
- **L1 / L2**：渐进外推，测泛化

扰动轴：
- **W0-W4**：语言扰动 5 档（W0 干净 → W4 严重）
- **V0-V4**：视觉扰动 5 档（V0 干净 → V4 严重）

## 🧠 我的评估

### 评测维度（关键！）
- **三维评测**：能力 × 难度 × 双扰动轴——是目前**最系统化的综合平台**
- 与 VLA-Risk 对比：
  - VLA-Risk 6 维（vision × instruction × 3 task aspects）专攻**对抗扰动**
  - Arena 双轴 5 档专攻**渐进扰动曲线** + 多任务广覆盖
  - **完全互补**：VLA-Risk 测最坏 case，Arena 测大场覆盖
- 比 LIBERO 谱系**广度更大**（不限 LIBERO 任务，自己设计 170 个）

### 任务规模
- **170 tasks**（远超 LIBERO 130）
- 4 类能力分组
- 每任务 3 难度 → 实际有 510 个 (task, difficulty) 组合
- 加 W×V 双扰动 → 完整矩阵指数级

### 评估 baseline 模型
- 论文未具名清单（信息缺失）
- 但根据「VLA 倾向记忆而非泛化」「不对称鲁棒性」的描述，应至少包含 OpenVLA / π0 / OpenVLA-OFT

### 关键 finding
- 「**强烈倾向于记忆而非泛化**」
- **不对称鲁棒性**：W/V 两轴上崩塌速度差异大
- **缺乏安全约束意识**——给安全部署敲警钟
- 长程任务上「无法组合技能」

### 复现成本
- 论文承诺「端到端工具链 + VLA-Arena-S/M/L 数据集 + 模型 + leaderboard」——理论上完整
- 但 **2025-12 新出**，代码/leaderboard 是否真上线需验证
- 显存需求未明确，估计与 LIBERO 同级（24GB 推理）
- 评测时间**会很长**：170 × 3 × 25 (W×V) = 12,750 evaluation cells——子集化必要

### 是否适合曦源课题
- ✅✅ **强烈推荐 TOP-1 候选（综合平台）**
- 优点：
  1. **官方有 leaderboard** → 曦源结果可直接打榜
  2. **维度最全**——能力 × 难度 × 双扰动轴，写论文 story 最丰满
  3. **故事感最强**：「在 Arena 上做 holistic robustness 评测」
- ⚠️ 风险：
  1. **代码 release 状态待核实**（2025-12 太新）
  2. 完整矩阵评测时间长——必须子集化，但论文方提供了 S/M/L 三档数据集，可挑 S 跑
  3. baseline 模型清单不明——可能需要自跑

## 🔗 关联
- [[VLA-Risk_ICLR2026]]：同期 benchmark，VLA-Risk 攻击 / Arena 综合
- [[2025-10_LIBERO-Plus]]：Plus 单 dim 退化 / Arena 双轴渐进
- [[2025-10_RoboChallenge]]：Arena 仿真 / RoboChallenge 真机
