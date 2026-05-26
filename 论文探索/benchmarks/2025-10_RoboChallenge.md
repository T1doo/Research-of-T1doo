---
title: "RoboChallenge: Large-Scale Online Evaluation of Vision-Language-Action Models for Real-World Robotic Manipulation"
authors: "37 authors (Tiancai Wang corresponding)"
year: "2025"
venue: "arXiv preprint (2025-10)"
arxiv: "2510.17950"
status: "已精读"
tier: "⭐⭐⭐ 唯一真机 online platform"
tags: [literature, T1D, benchmark, robustness-eval, real-robot, online-evaluation]
---

# RoboChallenge — 真机 30 任务在线评测平台

> [!info] 元信息
> - **作者**：37 人联合（Tiancai Wang 通讯）
> - **arXiv**：[2510.17950](https://arxiv.org/abs/2510.17950)
> - **官网**：[robochallenge.ai](https://robochallenge.ai)
> - **核心定位**：**首个真机在线评测平台**——用户远程调 API，机器人在远程实验室真跑

## 📄 这个 benchmark 是做什么的

解决「论文都说自己 SR 90%，但谁也复现不了」的痛点：用户**不上传 model**，而是**在自己端运行推理 + 通过 API 控制远程真机器人**。30 个 table-top 任务（Table30），所有轨迹和视频公开。

任务覆盖：
- 「arrange flowers」「fold dishcloth」「wipe the table」等
- 测试 **3D 定位 / 多视角推理 / 时序依赖 / 长程规划 / 物体识别 / 双臂协调 / 软体材料**

## 🧠 我的评估

### 评测维度
- **真机 30 任务**——是所有 benchmark 里**最接近真实部署**的
- **不显式扰动**，但真机自带 noise（光照、物体放置、摩擦）——天然 robustness 测试
- 与 SimplerEnv 关系：SimplerEnv 是「real-to-sim 代理」，RoboChallenge 是「直接真机」
- 与 VLA-Risk 关系：VLA-Risk sim-only / RoboChallenge real-only

### 任务规模
- **30 tasks**（Table30）
- 任务设计覆盖多种能力（pick/place 简单到 deformable 难）

### 评估 baseline 模型
- **5 个家族**：
  - **π0, π0.5**（Physical Intelligence）
  - **CogACT**（Microsoft）
  - **OpenVLA + OFT** 变体
  - 各 model 跑 task-specific + generalist 两种模式

### 关键 finding
- **π0.5 显著领先**（其他被甩开）
- 简单 pick-and-place SR ~90%（接近饱和）
- **SOTA 平均 SR 仅 ~44%**——大量提升空间
- 时序推理 + 软体操控**仍是最大瓶颈**

### 复现成本（关键！）
- **提交模式**：用户端推理 → 调 API 控制远程机器人
  - 用户不寄送硬件
  - 用户不上传 model weights
  - **仅需稳定网络 + 自己 GPU 跑推理**
- **本科生可行性**：✅ **门槛极低**——只要能本地跑 OpenVLA 推理（24GB 显存），就能用 RoboChallenge 评测
- **leaderboard**：✅ 官网公开排名（SR + progress score）+ 所有 trajectory 视频
- **真机租用时间**：未明示，估计需排队/预约
- **开源**：评测协议开源，但实验室硬件是平台方提供

### 是否适合曦源课题
- ✅✅✅ **强烈推荐 TOP-3 候选（真机代理 / leaderboard story）**
- 优点：
  1. **本科生友好**——无需真机硬件，只需网络 + 1×24GB GPU
  2. **leaderboard 公开**——上榜本身就是曦源结题亮点
  3. **唯一真机数据**——曦源论文若有 RoboChallenge 数字，比纯 sim 高一个档次
  4. 5 个 baseline 已公开，对比表现成可用
- ⚠️ 风险：
  1. **真机评测有排队** —— 若做大量 ablation 不现实
  2. 提交 API 接入需开发——非纯 plug-and-play（但应当不复杂）
  3. 2025-10 平台，**稳定性 / 可用性需验证**（官网 / 论文未明说每周可提交几次）
- **决策建议**：作为 **「最后真机验证」** 步骤——前期主测 LIBERO-Plus（快速迭代），最后挑 SOTA 配置在 RoboChallenge 跑 5-10 任务证明真机能 work

## 🔗 关联
- [[2024-05_SimplerEnv]]：互补——SimplerEnv 代理真机，RoboChallenge 真·真机
- [[VLA-Risk_ICLR2026]]：VLA-Risk sim-only / RoboChallenge real-only
- [[2025-10_LIBERO-Plus]]：Plus sim 快速迭代 / RoboChallenge 真机最终验证
- [[2025-12_VLA-Arena]]：Arena 仿真综合 / RoboChallenge 真机门面
