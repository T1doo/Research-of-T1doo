---
title: "THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation"
authors: "Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, Dieter Fox"
year: "2024"
venue: "RSS 2024"
arxiv: "2402.08191"
status: "已精读"
tier: "⭐⭐⭐ 经典 sim-to-real validated"
tags: [literature, T1D, benchmark, robustness-eval, RLBench, generalization]
---

# Colosseum — 12 维扰动 × 20 RLBench 任务 × 真机验证

> [!info] 元信息
> - **作者**：Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, Dieter Fox（UW · USC · NVIDIA）
> - **arXiv**：[2402.08191](https://arxiv.org/abs/2402.08191)
> - **官网**：[robot-colosseum.github.io](https://robot-colosseum.github.io)
> - **数据集规模**：20 RLBench 任务 × 12 维扰动
> - **真机验证**：✅ Franka Panda 4 个任务

## 📄 这个 benchmark 是做什么的

最早系统化做 **环境扰动鲁棒性** 的 benchmark 之一（2024-02 即 RSS-24）。基于 RLBench + PyRep + CoppeliaSim，在 20 个操控任务上沿 12 维（论文一直被引为 14 维，实际表格是 12 维）扰动，**唯一做了真机验证**的 sim-only 鲁棒性 benchmark（R² = 0.614）。

12 维扰动（3 类）：
1. **Manipulation Object (MO)**：color, texture, size
2. **Receiver Object (RO)**：color, texture, size
3. **Background**：light color, table color, table texture, distractor objects, background texture, camera pose

20 RLBench tasks：open drawer / slide block / basketball in hoop / meat on grill / close box / close laptop / empty dishwasher / reach and drag / get ice / hockey / put money in safe / place wine / move hanger / wipe desk / straighten rope / insert peg / stack cups / turn oven / setup chess / scoop with spatula.

## 🧠 我的评估

### 评测维度
- **12 维**（物体属性 × 接受物 × 背景三类）
- **与 LIBERO-Plus 对比**：维度高度重叠（color/texture/light/camera/distractor），但 Colosseum 多了「**物体大小**」「**接受物属性**」
- 与 VLA-Risk 对比：Colosseum 不测语言扰动，VLA-Risk 测；Colosseum 测真机相关性，VLA-Risk 不测

### 任务规模
- **20 tasks × 12 dim**——比 LIBERO-Plus 任务数少但维度精
- 真机仅 4 task 验证（insert peg / slide / scoop / chess）

### 评估 baseline 模型
- **4 个模型**（论文常被引为 5 个）：
  - 2D：**R3M-MLP**，**MVP-MLP**
  - 3D：**PerAct**，**RVT**
- **不含主流 VLA**（2024-02 太早，OpenVLA 尚未出）——是其最大短板

### 关键 finding
- **单扰动 30-50% SR 退化**
- **多扰动叠加 ≥75% 退化**
- 最致命因素：**distractor 数量 > 目标物颜色 > 光照**
- **真机相关 R² = 0.614**——sim 评测是 real-world 的合理代理（关键贡献）

### 复现成本
- **仿真器**：CoppeliaSim + PyRep（**比 MuJoCo 麻烦**——CoppeliaSim 是 GUI 程序，无显卡 headless 麻烦）
- **GPU**：训练 PerAct 需 32GB（A100），推理 24GB
- **3D 打印**：论文提供 3D 打印模型用于真机扰动复现——**真机门槛极低**
- **开源**：✅ Apache 2.0

### 是否适合曦源课题
- ⚠️ **不推荐为主测试场**
- 理由：
  1. **baseline 落伍**——4 个模型都不是 VLA，需要自己加 OpenVLA / π0
  2. **CoppeliaSim 依赖**——比 LIBERO 系列(MuJoCo)安装复杂
  3. **任务数较少**（20 vs LIBERO 130）
- ✅ **优点**：
  1. 真机相关性是其他 benchmark 都没有的硬通货
  2. 12 维扰动**与 LIBERO-Plus 互补**——若主测 Plus，可加 Colosseum 子集做 cross-validation
- **决策建议**：作为「辅助 benchmark」——主测 LIBERO-Plus，论文里加一段 Colosseum 子集结果证明跨 benchmark 一致性

## 🔗 关联
- [[VLA-Risk_ICLR2026]]：维度互补（Colosseum 视觉环境扰动 / VLA-Risk 含语言扰动）
- [[2025-10_LIBERO-Plus]]：12 维 vs 7 维，方法论相同
- [[2024-05_SimplerEnv]]：都做 sim-real 相关性，但 SimplerEnv 不测扰动
