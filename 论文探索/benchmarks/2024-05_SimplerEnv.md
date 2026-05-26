---
title: "Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv / SIMPLER)"
authors: "Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, Ted Xiao"
year: "2024"
venue: "CoRL 2024"
arxiv: "2405.05941"
status: "已精读"
tier: "⭐⭐⭐ Real-to-sim 标杆"
tags: [literature, T1D, benchmark, robustness-eval, real-to-sim, evaluation-infrastructure]
---

# SimplerEnv (SIMPLER) — Real-to-Sim 评测基础设施

> [!info] 元信息
> - **作者**：Xuanlin Li, Kyle Hsu, ..., Chelsea Finn, Hao Su, Quan Vuong, Ted Xiao（UCSD · Stanford · UCB · Google DeepMind）
> - **arXiv**：[2405.05941](https://arxiv.org/abs/2405.05941)
> - **官网**：[simpler-env.github.io](https://simpler-env.github.io) · [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv)
> - **核心定位**：**唯一**把 sim 评测与 real-world 评测做高相关性的 benchmark

## 📄 这个 benchmark 是做什么的

解决 real-world 评测「不 scalable、难复现」的痛点：构造一个 **simulator suite，让 sim 评测结果与 real-world 排序高度一致**（Pearson r ≈ 0.92 on Google Robot），从而**用 sim 替代 real-world 排行**。

两大技术：
1. **Control Gap 闭合（SysID）**：simulated annealing 在演示轨迹上拟合 PD 参数
2. **Visual Gap 闭合**：
   - **Green-screening**：分割 sim 中互动物体 + 真实背景覆盖
   - **Texture matching**：把真实物体纹理投影到 sim 资产

评估指标：
- **Pearson Correlation**：sim 与 real-world SR 线性一致性
- **MMRV (Mean Maximum Rank Violation)**：考虑评测噪声的排序错误代理

## 🧠 我的评估

### 评测维度
- **不是"扰动"benchmark**——是「让 sim 评测**变得可信**」的基础设施
- 评测的是**实际 SR 复现性**，不是 robustness 本身
- 但论文 demonstrate 了：sim 能预测 real-world 对 **lighting / backgrounds / textures / camera pose** 的鲁棒性（与 Colosseum / LIBERO-Plus 维度有交集）

### 任务规模
- 覆盖 **2 套 real-world 数据集对应的 sim**：
  - **Google Robot**（RT-series tasks）
  - **WidowX + BridgeData V2**
- 任务数：每套约 5-10 个核心任务
- 较小（远小于 LIBERO 130）

### 评估 baseline 模型
- **RT-1, RT-1-X, RT-2-X**（Google Robot 系）
- **Octo-Base, Octo-Small**（Berkeley 系）
- **OpenVLA**（社区扩展）
- 物理仿真器：**SAPIEN**（主）+ **Isaac Sim**（验证）

### 关键 finding
- **Pearson r ≈ 0.924 on Google Robot**——sim 评测可信度极高
- **MMRV**比 MSE 更鲁棒于评测噪声
- Visual matching > SysID 单独的效果——**视觉是 real-to-sim gap 主因**
- sim 能预测 real-world distribution shift（lighting/bg/texture/camera）

### 复现成本
- **GPU**：推理 24GB 足够；不训练
- **仿真器**：SAPIEN 开源、Python 友好
- **一行 import**：`import simpler_env` 后即可调 Gym API
- **评测时间**：单 model 全套约 2-4 h
- **开源**：✅ 完整 + Colab demo

### 是否适合曦源课题
- ⚠️✅ **作为「真机代理」推荐 TOP-3 候选**
- 优点：
  1. **跑 SimplerEnv = 准 real-world 评测**——一篇本科论文如果声称 robustness，加 SimplerEnv 数据就有 real-world 说服力
  2. **复现门槛最低**——一行代码即可，46GB 显存绰绰有余
  3. baseline 已固定（RT-X / Octo），可直接拼对比表
- ⚠️ 缺点：
  1. **任务数少**——只有 Google Robot / WidowX 两套，覆盖窄
  2. **不测扰动维度**——是基础设施，不是 robustness benchmark；需配合 LIBERO-Plus 等使用
  3. RT-1 / RT-2-X 不开源权重——只能跑 Octo / OpenVLA
- **决策建议**：作为 **「真机代理验证」** 加入——主测 LIBERO-Plus / VLA-Arena，最后用 SimplerEnv 跑一遍证明结果与 real-world 一致

## 🔗 关联
- [[2024-02_Colosseum]]：两者都做 sim-real correlation，但 Colosseum 实测真机，SimplerEnv 用相关性指标
- [[2025-10_RoboChallenge]]：RoboChallenge 是真·真机，SimplerEnv 是 real-to-sim 仿真
- [[VLA-Risk_ICLR2026]]：VLA-Risk 评测扰动 / SimplerEnv 评测 SR 复现性
