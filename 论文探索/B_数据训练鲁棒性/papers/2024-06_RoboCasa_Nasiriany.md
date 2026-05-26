---
title: "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots"
authors: "Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, et al. (UT Austin + NVIDIA)"
year: "2024"
journal: "RSS 2024"
doi: "10.48550/arXiv.2406.02523"
arxiv: "2406.02523"
venue: "RSS 2024"
citekey: "nasirianyRoboCasaLargeScaleSimulation2024"
itemType: "conference paper"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 厨房场景大规模仿真"
tags: [literature, T1D, 主线B, B4_合成数据, 厨房场景, 大规模仿真, RoboCasa, RSS2024]
---

# RoboCasa — 厨房场景大规模仿真

> [!info] 元信息
> - **作者**：Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, et al. (UT Austin Robin Lab + NVIDIA)
> - **机构**：UT Austin + NVIDIA + (Yuke Zhu 组)
> - **日期**：2024-06-04 (arXiv) → RSS 2024
> - **arXiv**：[2406.02523](https://arxiv.org/abs/2406.02523)
> - **项目主页**：robocasa.ai
> - **GitHub**：开源
> - **核心定位**：**专门为厨房场景**优化的大规模仿真平台 + 自动 demo 生成

## 📄 Abstract

通用 VLA 训练需要大规模多样化的真实场景仿真。RoboCasa 提出**厨房特化的大规模仿真平台**：(1) **120+ 厨房场景** + **2500+ 物体资产**（导自 Objaverse），覆盖橱柜/冰箱/烤箱/餐具等；(2) **10 类技能 × 65 原子任务**——pick/place/open/close/pour/wipe 等；(3) **MimicGen-driven demo 生成**——单个人工 demo → 100+ 自动生成 demo（用 MimicGen pipeline）；(4) **VLA 训练验证**——在 RoboCasa demo 上预训练 + 真机迁移。论文展示 100K+ 仿真 demo + 100 真机 demo 联合训练 BC policy，**真机泛化显著优于纯真机数据**。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「垂直特化场景」是 VLA 落地的现实路径**：通用 simulator（Genesis/Holodeck）追求场景多样性，但实际机器人应用往往垂直领域（**厨房机器人、餐厅机器人**）。RoboCasa 选择**深耕厨房**——65 原子任务覆盖了真实厨房的 80% 操作。这是商业落地的明智路线，对学术研究也是**有边界、可衡量**的目标。
2. **「Sim 100K demo + Real 100 demo」的混合策略最优**：纯 sim 训出来 sim2real gap 大；纯 real 数据量不足。RoboCasa 实验证明**1000:1 sim:real 混合**效果最佳——真机数据是「校准」，仿真数据是「容量」。这是 sim-to-real 的工程黄金比例。
3. **MimicGen 集成是关键工程胜利**：RoboCasa 不自己造数据生成轮子，直接复用 [[2024_MimicGen_Mandlekar]]：1 个 human demo → 200+ 自动 demo（SE(3) 变换）。这给本科生重要启示：**MimicGen 是数据生成的"瑞士军刀"，必须掌握**。

### 方法论（核心架构）

#### Kitchen Scene Generation
- **资产库**：2500+ 厨房物体（Objaverse 筛选）
- **场景模板**：120 个厨房布局（手工 + LLM 辅助）
- **程序化变化**：每个模板 ×10 视觉变体（光照/纹理/视角）
- 总场景数：**1200+**

#### Skill Taxonomy（10 类 × 65 原子任务）
```
1. Pick-and-Place (15 tasks): 各类物体 pick/place
2. Pour (8 tasks): 不同容器间倒液体/颗粒
3. Cabinet/Door (10 tasks): 开关橱柜
4. Drawer (6 tasks): 开关抽屉
5. Cook (6 tasks): 用炉灶/烤箱
6. Microwave (4 tasks): 微波炉操作
7. Sink (8 tasks): 水龙头/洗菜
8. Wipe (4 tasks): 清洁桌面
9. Stack (2 tasks): 堆叠餐具
10. Other (2 tasks): 复合任务
```

#### Demo Generation Pipeline
- **Source demo**：每任务 1-5 个 human teleoperation demo
- **MimicGen 增广**：每个 source → 100-200 synthetic demo
- **总规模**：10K+ × 100 = **1M+ synthetic demos**
- 过滤：物理可行性 + 成功标签

#### VLA Training Protocol
- Backbone：BC-Transformer / Diffusion Policy
- 训练数据：RoboCasa sim + real RoboMimic data
- Sim:Real ratio：10:1, 100:1, 1000:1（experimental）

### 关键实验数据

#### 单任务 vs 多任务训练
| 训练方式 | Sim SR | Real SR |
|---|---|---|
| 单任务 sim only | 78% | 31% |
| 65 任务 sim only | 71% | 42% |
| 65 任务 sim + 100 real | 73% | **64%** |
| **65 任务 sim + 1K real** | **75%** | **74%** |

#### Sim:Real ratio 消融
| Ratio | Real SR |
|---|---|
| 0:1 (real only, 100 demo) | 38% |
| 10:1 | 51% |
| 100:1 | 62% |
| **1000:1** | **74%** |
| 10000:1 | 71% (diminishing) |

#### 真机泛化（新厨房环境）
- 见过的厨房：74% SR
- 新厨房：61% SR (drop 13 pp, 较好的泛化)
- 严格 OOD（不同房型）：38% SR

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **本科生 46 GB 可行性**：✅ **极强推荐**
   - RoboCasa 开源 + 文档完善
   - 安装：基于 RoboSuite，1 天搞定
   - 数据生成：单卡 30 GB，每天 10K demo
   - VLA training：单卡 40 GB（SmolVLA/FocusVLA 规模）
   - **本科生 6-8 周可完成「RoboCasa + RobustVLA」实验**
2. **与 [[2024_MimicGen_Mandlekar]] 强耦合**：必须先掌握 MimicGen 才能用 RoboCasa demo 生成。**学习路径**：MimicGen → RoboCasa → RoboCasa + RobustVLA。
3. **与 [[2026_RobustVLA_Guo]] 直接结合**：
   - RoboCasa 提供 **「场景 × 任务」多样性**
   - RobustVLA 提供 **「图像扰动」鲁棒性**
   - **联合训练**：在 RoboCasa 数据 + worst-case δ 增广上训 VLA → 应该既泛化又鲁棒
4. **与 [[VLA-Risk_ICLR2026]] 关联**：RoboCasa 提供测试场景，但**没做对抗扰动评测**——这是研究空白。**可设计**：在 RoboCasa 场景上跑 VLA-Risk 风格扰动，扩展 benchmark。
5. **「Sim 100K + Real 100 = 黄金比例」给我的策略指引**：本科生项目应**优先扩大 sim 数据**（边际成本低），**少量真机数据做校准**——不要试图全靠真机。

### 待解问题 / 后续追读

1. **65 任务的 task diversity 不足**：仅厨房，没有客厅/工业场景。需要多个垂直 RoboCasa 拼起来。
2. **MimicGen 的 SE(3) 变换的局限**：复杂多步任务可能合成失败率高（追读 DexMimicGen）。
3. **VLA 在 OOD 厨房性能下降 36 pp**：仍需更强 generalization 机制（FocusVLA 风格 attention？）。

### 在演化线上的位置

> **RoboMimic (2021) → MimicGen (2023.10) → RoboCasa (2024.06) → DexMimicGen (2024.10) → 未来 RoboCasa-2.0（多场景）**

RoboCasa 是 **「MimicGen 在垂直场景上的大规模应用」**，证明了 MimicGen + 场景特化 pipeline 的可扩展性。

### 设计风险 / 复现挑战（46 GB 显存约束下的可行性）

| 模块 | 单卡可行性 | 资源 | 时间评估 |
|---|---|---|---|
| 安装 RoboCasa | ✅ | 12 GB GPU | 1 天 |
| 跑通 65 原子任务示例 | ✅ | 24 GB GPU | 1 周 |
| MimicGen 自动 demo 生成 | ✅ | 30 GB GPU | 10K/day |
| BC-Transformer 训练 | ✅ | 40 GB GPU | 1-2 周 |
| SmolVLA/FocusVLA fine-tune | ✅ | 40 GB GPU | 2-4 周 |
| **本科生完整路径** | ✅ **强烈推荐** | A6000 + LLM API | 6-8 周 |

**关键判断**：RoboCasa 是**12 篇 B4 论文中本科生最高 ROI 的选择之一**：
- 完整闭环（场景 + 任务 + 数据 + VLA）
- 开源 + 文档好
- 单卡可行
- 直接对接 VLA 训练

%% end my-thoughts %%

## 🔗 关联笔记

- **B4 同分类**：[[2023_RoboGen_Wang]], [[2023_Gen2Sim_Katara]], [[2024_Genesis]], [[2024_Holodeck]], [[2024_MimicGen_Mandlekar]]
- **强耦合**：[[2024_MimicGen_Mandlekar]]（数据生成内核）
- **演化前后**：← [[2024_MimicGen_Mandlekar]] → [[2024_DexMimicGen_Jiang]]
- **鲁棒性整合**：[[2026_RobustVLA_Guo]], [[VLA-Risk_ICLR2026]]
- **场景特化对照**：[[2023_Habitat3]]（人机交互特化）

## 📌 Action Items

- [ ] **优先级 P0**：安装 RoboCasa + 复现 65 任务子集（3 周）
- [ ] **P1**：用 RoboCasa demo + SmolVLA 训练 baseline policy
- [ ] **P2**：在 RoboCasa 数据上跑 RobustVLA worst-case δ
- [ ] **P3**：在 RoboCasa 场景上做 VLA-Risk 风格对抗扰动评测

%% Import Date: 2026-05-26 %%
