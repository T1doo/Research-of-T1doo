---
title: "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations"
authors: "Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox"
year: "2023"
journal: "CoRL 2023"
doi: "10.48550/arXiv.2310.17596"
arxiv: "2310.17596"
venue: "CoRL 2023"
citekey: "mandlekarMimicGenDataGeneration2023"
itemType: "conference paper"
status: "已精读"
tier: "⭐⭐⭐ 必读 · SE(3) 数据合成基石"
tags: [literature, T1D, 主线B, B4_合成数据, SE3变换, demo增广, MimicGen, NVIDIA, CoRL2023]
---

# MimicGen — SE(3) 变换驱动的 demo 自动生成

> [!info] 元信息
> - **作者**：Ajay Mandlekar (NVIDIA), Soroush Nasiriany (UT Austin), 等 8 人
> - **机构**：NVIDIA + UT Austin (Yuke Zhu 组) + UW (Dieter Fox 组)
> - **日期**：2023-10-26 (arXiv) → CoRL 2023
> - **arXiv**：[2310.17596](https://arxiv.org/abs/2310.17596)
> - **项目**：mimicgen.github.io
> - **GitHub**：开源（NVIDIA Lab）
> - **核心定位**：**最广泛使用的机器人 demo 自动增广工具**——从 10 个 human demo 生成 200+ synthetic demo

## 📄 Abstract

机器人 imitation learning 受限于人工演示数据成本。MimicGen 提出**SE(3) 变换驱动的 demo 自动生成 pipeline**：(1) 把 human demo 分解为**子段**（每段对应一个 subtask）；(2) 在新 scene（不同物体位置）下，对每个子段做**SE(3) 变换**（物体姿态变换 → 末端轨迹变换）；(3) 重新执行变换后的轨迹，**只保留成功 trajectory**。论文证明：(1) **10 个 human demo → 1000+ synthetic demo**，成功率保持 95%+；(2) 在 18 任务上 BC training 从 10 demo 提升到 1000 demo 后，平均 SR 从 27% → 65%；(3) **跨场景泛化**：在 sim demo 上训出来的策略 zero-shot 真机 SR 提升 30+ pp。**MimicGen 已成 RoboCasa / DexMimicGen 等 12 篇 B4 论文的标准 backbone**。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「SE(3) 变换 + segment replay」是数据合成的极简哲学**——MimicGen 没有用 LLM、没有用生成模型，纯靠**几何变换 + 重播**就把 demo 数量 ×100。这是工程上的**奥卡姆剃刀**：最简单的方法做最有效的增广。这与生成式合成数据（RoboGen/Gen2Sim）形成鲜明对比——**经典几何 vs 现代生成**两条路线。
2. **「Subtask 分解」是 SE(3) 变换的关键前提**：直接对整个 demo 做 SE(3) 变换会失败（因物体不在同一位置）。MimicGen 把 demo 分为 "approach → grasp → transport → place" 等 subtask，**每段独立变换**——这模拟了人类完成新场景任务时的思维方式。
3. **「成功 trajectory filter」让低质量自动忽略**：MimicGen 直接执行变换后轨迹，**失败的就扔掉**——这是简单但有效的质量控制。代价是合成率（生成 100 个保留 50 个），但保证质量。

### 方法论（核心算法）

#### Source Demo Segmentation
- Human demo 标注（自动 + 人工辅助）：物体接触事件作为分段点
- 例：pick-place demo = [approach_obj, grasp_obj, transport, approach_target, place, release]
- 每段一个 reference SE(3) pose

#### SE(3) Transform Application
```
For new scene s_new with object pose T_obj_new:
    For each segment i in source demo:
        T_relative = T_seg_i × T_obj_source^{-1}  (相对物体的轨迹)
        T_new_seg_i = T_relative × T_obj_new  (变换到新物体姿态)
    Concatenate all segments → new trajectory
```

#### Trajectory Filter
- 物理可行性 check（无碰撞、可达性）
- 执行检查（在 sim 跑一遍，成功标签 + reward）
- 成功率：~50-70%（取决于任务复杂度）

#### Scaling Strategy
- Per source demo → 100-200 synthetic demos
- 10 source × 100 synthetic = 1000 demos
- 总训练数据 ×100

### 关键实验数据

#### 18 任务平均 BC training SR
| Training data | Avg SR |
|---|---|
| 10 source demo only | 27% |
| 10 source + 100 synth (per source) | 51% |
| 10 source + 200 synth (per source) | **65%** |
| 10 source + 500 synth (per source) | 64% (diminishing) |

#### 真机 zero-shot transfer
| 训练数据 | 真机 SR (avg over 5 tasks) |
|---|---|
| Sim 10 demo only | 22% |
| Sim 1000 synthetic (MimicGen) | 54% |
| **Sim 1000 synthetic + Real 10 calibration** | **77%** |

#### Success rate of MimicGen generation
| 任务类型 | Generation SR |
|---|---|
| 简单 pick-place | 72% |
| 中等（stacking, drawer） | 58% |
| 复杂（cable, articulated） | 31% |

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **本科生 46 GB 可行性**：✅ **必学必用的工具**
   - 安装：基于 RoboSuite，1 天
   - 运行：单卡 24 GB（不需 GPU 训练，主要是 sim execution）
   - 生成 1000 demo：~1 天 CPU + GPU
   - **本科生第 1 周就该掌握**
2. **与 [[2024_RoboCasa_Nasiriany]] 是父子关系**：RoboCasa 直接用 MimicGen 做 demo 生成。学 MimicGen 后即可解锁 RoboCasa。
3. **与 [[2026_RobustVLA_Guo]] 联合策略**：
   - MimicGen 提供**场景几何多样性**（物体位置变化）
   - RobustVLA 提供**视觉扰动鲁棒性**（图像层）
   - **联合 pipeline**：MimicGen 增广 demo → RobustVLA worst-case δ 增广 → 双重鲁棒训练
4. **与 [[VLA-Risk_ICLR2026]] 关联**：MimicGen 数据全是 "clean instruction + clean visual" 配对，**缺乏对抗鲁棒性**。**研究空白**：「Adversarial MimicGen」——在 SE(3) 变换基础上加入指令/视觉扰动。
5. **演化指引**：MimicGen 是**双臂、灵巧手版的基石**——DexMimicGen 是直接扩展。

### 待解问题 / 后续追读

1. **复杂任务生成率仅 31%**：cable manipulation 等 contact-rich 任务 SE(3) 变换难成功。需要更智能的变换（**学习型变换**？）。
2. **没考虑视觉/语义增广**：MimicGen 只变几何，不变视觉。可结合 [[2024_GreenAug]] / [[2025_RoboEngine]] 做联合增广。
3. **指令-动作对齐 missing**：MimicGen 假设指令固定，没考虑指令变化。**这是与 VLA-Risk 衔接的关键缺口**。

### 在演化线上的位置

> **DAgger (2011) → demo augmentation (2018+) → MimicGen (2023.10) → DexMimicGen (2024.10) → DYNAMIMICGEN (2025.11)**

MimicGen 是 **「demo 自动增广」路线的奠基石**，几乎所有 2024+ 的合成数据 pipeline（RoboCasa / DexMimicGen / DYNAMIMICGEN）都基于 MimicGen 思想扩展：
- **DexMimicGen**：双臂灵巧手
- **DYNAMIMICGEN**：dynamic environments
- **Iterative MimicGen**：迭代生成（GenSim2-style）

### 设计风险 / 复现挑战（46 GB 显存约束下的可行性）

| 模块 | 单卡可行性 | 资源 | 时间评估 |
|---|---|---|---|
| 安装 MimicGen | ✅ | CPU + 12 GB GPU | 1 天 |
| 跑示例 demo 生成 | ✅ | 24 GB GPU | 1 天 |
| 1000 synthetic demo 生成 | ✅ | 30 GB GPU | 1-2 天 |
| BC-Transformer training | ✅ | 40 GB GPU | 1 周 |
| SmolVLA fine-tune on MimicGen data | ✅ | 40 GB GPU | 2 周 |
| **本科生路径**：1 个月掌握 MimicGen + RoboCasa | ✅ **必经之路** | A6000 | 4 周 |

**关键判断**：MimicGen 是 **B4 方向的 "Hello World"**——任何想做合成数据的本科生**第 1 个月**都该跑通 MimicGen，再展开后续选择（RoboCasa / DexMimicGen / 自研扩展）。

%% end my-thoughts %%

## 🔗 关联笔记

- **B4 同分类**：[[2024_RoboCasa_Nasiriany]]（直接基于）, [[2024_DexMimicGen_Jiang]], [[2023_RoboGen_Wang]]
- **演化前后**：→ [[2024_DexMimicGen_Jiang]] → DYNAMIMICGEN (2025) → Iterative Compositional Gen (2025)
- **互补路线**：[[2023_RoboGen_Wang]]（LLM 驱动生成）
- **鲁棒性整合**：[[2026_RobustVLA_Guo]], [[VLA-Risk_ICLR2026]]
- **视觉增广配套**：[[2024_GreenAug]], [[2025_RoboEngine]], [[2026_RoboVIP]]

## 📌 Action Items

- [ ] **P0**：跑通 MimicGen 官方示例（1 周）
- [ ] **P1**：在 LIBERO 任务上用 MimicGen 生成 1000+ demo（2 周）
- [ ] **P2**：设计「MimicGen + GreenAug 联合增广」实验（3 周）
- [ ] **P3**：评估 MimicGen-generated data 对 RobustVLA 训练的增益

%% Import Date: 2026-05-26 %%
