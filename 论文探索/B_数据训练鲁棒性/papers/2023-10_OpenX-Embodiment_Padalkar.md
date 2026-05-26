---
title: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
authors: "Open X-Embodiment Collaboration (Padalkar, Pertsch, Brohan, et al.)"
year: "2023"
journal: "ICRA 2024 / arXiv"
doi: "10.48550/arXiv.2310.08864"
arxiv: "2310.08864"
venue: "ICRA 2024 Best Paper"
citekey: "padalkarOpenXEmbodiment2024"
itemType: "conference paper"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B7 cross-embodiment 起点"
tags: [literature, T1D, 主线B, B7, cross-embodiment, OXE, RT-X, 数据集, 跨机体]
---

# Open X-Embodiment / RT-X — 精读笔记（B7 起点）

> [!info] 元信息
> - **作者**：Google DeepMind 联合 22 机构、34 实验室
> - **arXiv**：[2310.08864](https://arxiv.org/abs/2310.08864)
> - **数据集**：https://robotics-transformer-x.github.io/
> - **B7 价值定位**：**cross-embodiment 训练的开山数据集**；催生了后续所有跨机体 VLA（OpenVLA、CrossFormer、HPT、X-VLA）

## 📄 Abstract

机器人学习长期受困于"每个 embodiment 一套数据、一套模型"的孤岛。本文提出 **Open X-Embodiment (OXE)** 数据集：
- **22 个机构** 联合贡献
- **34 个数据集** 整合
- **1M+ 轨迹**（精筛后约 970k）
- **22 种 embodiment**（Franka, xArm, Sawyer, WidowX, Google Robot, ALOHA …）
- 跨 **527 个 skill**，**160266 个 task**

模型：基于 RT-1 / RT-2 训出 **RT-X**，**第一次证明 cross-embodiment 训练可以正迁移**：在每个机体上比单机体训练好 1.5-3×。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（B7 时代的奠基性洞察）

1. **「跨机体训练带来正迁移」是反直觉的实验性发现**：
   - 历史认知：不同机体 action space 不同（7DoF vs 14DoF vs gripper-only），混训会互相干扰
   - OXE 实证：**统一 action 编码为相对 end-effector pose + gripper width** 后，多机体数据正迁移
   - RT-X 在每个机体上 vs 单机体训练：性能提升 1.5-3×（绝对成功率 +15~30pp）
   - **这一发现催生了整个 B7 子领域**

2. **统一 action 表征是 cross-embodiment 的关键桥梁**：
   - 不归一化：直接拼接 → 形态干扰、不可学习
   - 关节空间归一化：仍依赖 DoF 数
   - **EE-relative pose（论文方案）**：所有机体都可投影到 6DoF EE pose + gripper width = 7D 统一空间
   - 缺点：丢失关节空间的精细控制（如冗余自由度规划）
   - 后续 [[2024_CrossFormer]] 用更激进的 tokenization 解决

3. **数据规模的边际效益曲线**：
   - 1k demos → 60% success
   - 10k → 75%
   - 100k → 85%
   - 970k → 89%
   - **边际收益递减明显**：从 100k 到 970k 只增 4pp
   - 暗示：单纯堆数据不够，**模型架构 + 数据质量比规模更关键**（这是 OpenVLA / π0 后续工作的导火索）

### 方法论

#### 数据统一化（关键技术）

```
所有 dataset 投影到统一 schema:
  observation: {RGB, depth?, state}
  action: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper_open]  # 7D
  language: text instruction
  metadata: embodiment_id, dataset_source
```

#### RT-X 模型架构
- RT-1-X：22M 参数 EfficientNet + Transformer encoder
- RT-2-X：55B 参数 vision-language 模型 + action head
- 训练：multi-dataset 加权采样，按 dataset size 平方根加权

#### 统一 evaluation 协议
- 跨 22 embodiment 的 13 个真机评估场景
- 引入 **co-training factor**：单机体 fine-tune vs 多机体 + fine-tune

### 实验关键数据

#### RT-X 跨机体迁移
| Robot | RT-1 (single) | RT-1-X (cross-embod) | Δ |
|---|---|---|---|
| Google Robot | 56% | 67% | +11 |
| Stretch | 27% | 55% | +28 |
| ALOHA (bimanual) | - | 70% | new |
| xArm | 33% | 60% | +27 |
| Franka | 65% | 78% | +13 |

#### 数据 scaling law（log scale）
- success ∝ log(n_demos) 在 n < 100k
- saturated 在 n > 500k

### B7 时间线（OXE 是起点）

```
2023-10: Open X-Embodiment / RT-X (本笔记) ── 数据集 + 第一个跨机体模型
2024-06: OpenVLA (2406.09246)                ── 7B 开源 VLA，970k OXE
2024-08: CrossFormer (2408.11812)             ── 900k × 30 embod，更激进 tokenization
2024-09: HPT (2409.20537)                     ── 异构 tokenizer + 共享 trunk
2024-10: π0 (2410.24164)                      ── Physical Intelligence，flow matching action head
2025-10: X-VLA (2510.10274)                   ── 软提示 transformer，进一步通用化
2026-02+: Hierarchical Equivariant Policy 等   ── 加入等变结构、形态嵌入
```

### 与曦源的关联（极重要）

1. **OXE 是 SmolVLA / OpenVLA 训练数据的源头**：
   - SmolVLA pretrain 用 LeRobot 数据 + OXE 子集
   - 我们做 fine-tune 时是否要混入 OXE？这是个开放问题
   
2. **cross-embodiment 训练能否提升单机体鲁棒性**（开放问题）：
   - 假设：多机体数据让模型学到更"形态无关"的特征 → 对单机体的物理扰动也鲁棒
   - 反假设：多机体数据引入噪声 → 单机体性能下降
   - **这是一个有研究价值的实证问题**——我们可以做对比实验：
     - 配置 A：SmolVLA 只在单 Franka 数据上 fine-tune
     - 配置 B：SmolVLA 在 Franka + xArm 数据上 fine-tune
     - 测：单 Franka 上的鲁棒性（VLA-Risk / LIBERO-Plus）
     - 看 B 是否优于 A
   
3. **与 RobustVLA 协同的潜力**：
   - cross-embod 训练 = **数据维度的多样化**
   - RobustVLA worst-case = **扰动维度的多样化**
   - 两者正交，可组合
   - **可能产生：多机体 + worst-case 训练 = 极强鲁棒 VLA**（这是个有原创性的思路）

4. **数据混合权重问题**：
   - 22 dataset 怎么加权？OXE 用 sqrt(size) 加权
   - 可用 [[2024_ReMix_DRO]] 的 DRO 方法替换 → **OXE + DRO 数据混合**

### 待解问题

1. **action 统一化的损失**：EE-pose 丢了关节信息；某些任务（avoid singularity, redundancy resolution）需要 joint-space 控制
2. **embodiment-specific 调优**：单机体真机部署时，是否需要 embodiment-specific fine-tune？OXE 只做 zero-shot 评估
3. **数据质量 vs 规模**：1M demos 但质量参差，是否应该筛选？

### 复现挑战

- OXE 数据集 ~10TB，下载 + 处理 + 存储成本极高
- 训练 RT-2-X 需 256+ TPU，非学术团队不可行
- **替代路径**：用 OpenVLA / CrossFormer 的预训练 checkpoint，跳过 from-scratch 训练

%% end my-thoughts %%

## 🔗 关联笔记

- **B7 时间线**：[[2024_OpenVLA]], [[2024_CrossFormer]], [[2024_HPT]], [[2024_pi0]], [[2025_X-VLA]]
- **B7 形态嵌入**：[[2026_EmbedMorphology_Transformer]], [[2025_HierarchicalEquivariant]]
- **VLA 训练数据**：[[2024_OpenVLA]] 用 OXE 子集
- **数据混合方法论**：[[2024_ReMix_DRO]]
- **鲁棒性协同**：[[2026_RobustVLA_Guo]]（多机体 + worst-case 双重鲁棒？）

## 📌 Action Items

- [ ] 下载 OXE 子集（先 LIBERO 兼容的 5 个 dataset）
- [ ] **关键实验**：SmolVLA 单机体 vs 多机体 fine-tune，对比单机体鲁棒性
- [ ] 写一段「cross-embodiment as implicit data augmentation」纳入开题报告
- [ ] 对比 OXE 加权策略：sqrt vs uniform vs DRO

%% Import Date: 2026-05-26 %%
