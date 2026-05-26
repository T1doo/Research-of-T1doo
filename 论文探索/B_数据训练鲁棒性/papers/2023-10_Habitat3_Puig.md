---
title: "Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots"
authors: "Xavi Puig, Eric Undersander, Andrew Szot, Mikael Henaff, et al. (FAIR)"
year: "2023"
journal: "ICLR 2024"
doi: "10.48550/arXiv.2310.13724"
arxiv: "2310.13724"
venue: "ICLR 2024"
citekey: "puigHabitat3CoHabitat2023"
itemType: "conference paper"
status: "已精读"
tier: "⭐⭐ 必读 · 人机交互大规模仿真"
tags: [literature, T1D, 主线B, B4_合成数据, 人机交互, 高速仿真, Habitat3, FAIR, ICLR2024]
---

# Habitat 3.0 — 人-机交互的大规模仿真

> [!info] 元信息
> - **作者**：Xavi Puig (FAIR), Eric Undersander (FAIR), Andrew Szot (FAIR), 等 17 人
> - **机构**：Meta FAIR + Carnegie Mellon
> - **日期**：2023-10-19 (arXiv) → ICLR 2024
> - **arXiv**：[2310.13724](https://arxiv.org/abs/2310.13724)
> - **项目**：aihabitat.org
> - **GitHub**：开源（Meta FAIR）
> - **核心定位**：**第一个大规模人-机协作仿真平台**，>100 FPS 实时仿真

## 📄 Abstract

家庭机器人的关键挑战是**与人共存**——避让、协作、社交。Habitat 3.0 是 Habitat 系列第三代，主要扩展：(1) **Humanoid avatar**——真实的可控 3D 人形角色（含 SMPL-X 人体模型）；(2) **高速仿真**——>100 FPS 含人形 + 机器人 + 多物体场景；(3) **HSSD-200 数据集**——200 个高质量房屋扫描场景；(4) **Social rearrangement task**——人+机器人协作整理房间；(5) **HITL (human-in-the-loop)** 接口——真人 VR 控制 avatar 与 sim 机器人交互。论文展示训练的策略可在 HITL 与真人协作完成 social rearrangement 任务。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「机器人不是孤岛，必须考虑人」是 Habitat 3.0 的范式转移**：之前的 Habitat / AI2-THOR / RoboCasa 都假设场景中只有机器人。但真实世界家庭机器人必须**与人共存**——避让、传递、协作。Habitat 3.0 第一次把可控 humanoid 加入大规模仿真，开启「social robotics in simulation」时代。
2. **「>100 FPS 含 humanoid」是工程奇迹**：仿真 humanoid（含骨骼绑定 + 皮肤变形 + 实时碰撞）通常 <10 FPS。Habitat 3.0 通过：(a) SMPL-X 简化模型；(b) BVH-based collision；(c) GPU rendering 优化——做到 >100 FPS。这让**大规模 social rearrangement RL training 成为可能**。
3. **HITL 是 sim-to-real 的新接口**：传统 sim-to-real 是「sim 训→ real 测」。Habitat 3.0 的 HITL 让**真人 VR 进入 sim 与训练中的机器人交互**——这相当于**「半 sim 半 real」的数据采集模式**，更便宜更安全。可能是未来 sim-to-real 的关键中介。

### 方法论（核心系统）

#### Humanoid Avatar System
- **SMPL-X 人体模型**：可参数化（性别/体型/姿态）
- 控制：joint angle 直接控制 + IK 辅助
- 动画：从 BVH motion capture 库读取
- 物理：BVH-based capsule collision（高效）

#### HSSD-200 Dataset
- 200 个真实房屋的 3D 扫描
- 含家具、纹理、光照
- 物体 articulation 已标注（橱柜可开）

#### Social Rearrangement Task
- 任务：机器人 + 人协作整理房间
- 人 = AI policy 或真人（HITL）
- 评测：成功率 + 协作效率 + 社交合规性（不撞人）

#### High-Speed Simulation
- 主流配置：1 humanoid + 1 robot + 50 物体 = ~120 FPS（A100）
- 多 instance：256 并行环境 ~80 FPS each

#### HITL Interface
- VR 头戴（Quest 2/3）连入
- 真人通过 VR 控制 SMPL-X avatar
- 与 sim 中的机器人 policy 实时交互
- 用于：评测、demonstration collection

### 关键实验数据

#### 仿真性能
| 配置 | FPS |
|---|---|
| 1 humanoid + 1 robot + 简单 scene | 200+ |
| 1 humanoid + 1 robot + HSSD scene + 100 物体 | 120 |
| 1 humanoid + 1 robot + HSSD + 1000 物体 | 78 |
| 256 并行 envs（HSSD） | 80 FPS/env |

#### Social Rearrangement SR
| 方法 | 单 robot SR | 含人协作 SR |
|---|---|---|
| Hierarchical RL baseline | 41% | 28% |
| LLM planner | 52% | 39% |
| Multi-agent RL | 38% | 47% (社交合规性高) |

#### HITL 评测
- 用 20 名真人志愿者通过 VR 与机器人 policy 协作
- 平均任务完成时间：3.2 min
- 社交合规性评分（主观）：3.8/5

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **本科生 46 GB 可行性**：✅ **可行但小众**
   - Habitat 3.0 开源 + 文档好
   - 单卡 24-40 GB GPU 即可（仿真本身不耗 GPU 多）
   - HSSD-200 下载 ~50 GB
   - **但人机交互方向远离当前 VLA 主流**——本科生用 Habitat 3.0 风险高（论文方向偏门）
2. **演化定位**：Habitat 3.0 主要服务 **navigation / social robotics** 而非 manipulation。与 RoboCasa（manipulation）走两条不同路线。VLA 主流目前是 manipulation，Habitat 3.0 与 VLA 鲁棒性研究**关联较弱**。
3. **与 [[2026_RobustVLA_Guo]] 弱关联**：RobustVLA 关注 image-level 扰动，Habitat 3.0 提供大规模室内场景——可作为视觉多样性来源，但**主要用于 navigation 鲁棒性，非 manipulation**。
4. **与 [[VLA-Risk_ICLR2026]] 中等关联**：VLA-Risk 含 "Direct Manipulation"、"Semantic Reasoning"、"Autonomous Driving" 三类，**未含 "Social Interaction"**——Habitat 3.0 可启发**「Social Robustness」**作为新评测维度。
5. **方法论启发**：HITL 接口的思想可借鉴——**让真人测试者快速生成对抗指令**作为 robustness evaluation 工具。

### 待解问题 / 后续追读

1. **Humanoid 物理保真度**：SMPL-X 是简化模型，没有衣物动力学、面部表情等。
2. **VLA 上的应用少**：Habitat 3.0 主要 RL/navigation 实证，VLA training 实证缺失。
3. **数据集偏 navigation**：HSSD-200 适合 navigation，但 manipulation 任务库远小于 RoboCasa。

### 在演化线上的位置

> **Habitat 1.0 (ECCV 2019, navigation) → Habitat 2.0 (NeurIPS 2021, +interaction) → Habitat 3.0 (ICLR 2024, +humans) → 未来 Habitat 4.0?**

Habitat 系列代表 **「navigation + 多智能体仿真」**的 SOTA，但与 manipulation 主流（Isaac Lab / RoboCasa / Genesis）分支。**适合 navigation 鲁棒性研究，不适合 manipulation 鲁棒性研究**。

### 设计风险 / 复现挑战（46 GB 显存约束下的可行性）

| 用途 | 单卡可行性 | 资源 | 时间评估 |
|---|---|---|---|
| 安装 Habitat 3.0 | ✅ | 12 GB GPU | 1 天 |
| 跑示例 social rearrangement | ✅ | 24 GB GPU | 1 周 |
| HSSD-200 数据集下载 | ✅ | 50 GB disk | 1 天 |
| 训练 navigation policy | ✅ | 40 GB GPU | 1-2 周 |
| HITL VR 实验 | ⚠️ 需 VR 设备 | + Quest 2/3 | 数周 |
| **本科生路径** | ✅ 但小众 | 主要适合 navigation 方向 | 风险高 |

**关键判断**：Habitat 3.0 是**优秀的工具，但与 VLA manipulation 鲁棒性主流偏离**。**仅当导师研究方向涉及 navigation / social robotics / multi-agent 时才推荐**——manipulation 方向的本科生应优先选 RoboCasa / Genesis。

%% end my-thoughts %%

## 🔗 关联笔记

- **B4 同分类**：[[2024_RoboCasa_Nasiriany]]（manipulation 对照）, [[2024_Genesis]], [[2024_Holodeck]]
- **演化前后**：Habitat 1.0 → 2.0 → 3.0
- **方向差异**：[[2024_RoboCasa_Nasiriany]] = manipulation；[[2023_Habitat3_Puig]] = navigation + 人机交互
- **鲁棒性整合**：[[VLA-Risk_ICLR2026]]（启发 social robustness 新维度）
- **VLA 主流相关性**：偏低（VLA 主流是 manipulation）

## 📌 Action Items

- [ ] 跟踪 Habitat 3.0 + VLA 的整合工作（如有）
- [ ] 借鉴 HITL 思想：设计「真人对抗指令收集」工具
- [ ] 仅在导师方向涉及人机交互时深入 Habitat 3.0

%% Import Date: 2026-05-26 %%
