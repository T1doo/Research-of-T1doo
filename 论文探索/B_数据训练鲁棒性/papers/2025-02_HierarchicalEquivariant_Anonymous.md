---
title: "Hierarchical Equivariant Policy for Robust Robotic Manipulation"
authors: "Anonymous (under review)"
year: "2025"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2502.05728"
arxiv: "2502.05728"
venue: "arXiv 2025-02"
citekey: "anonymousHierarchicalEquivariant2025"
itemType: "preprint"
status: "精读完成"
tier: "⭐⭐ B7 等变 hierarchical"
tags: [literature, T1D, 主线B, B7, hierarchical, equivariant, SE3, manipulation]
---

# Hierarchical Equivariant Policy — 精读笔记

> [!info] 元信息
> - **arXiv**：[2502.05728](https://arxiv.org/abs/2502.05728)
> - **B7 价值定位**：把**SE(3) 等变性**与**分层策略**结合，对几何变换天然鲁棒；与跨机体研究的几何不变性相关

## 📄 Abstract

机器人操作策略对物体的几何变换（旋转、平移、镜像）应该等变——但标准 transformer 不天然等变。本文提出 **Hierarchical Equivariant Policy (HEP)**：
- **高层**：SE(3)-equivariant transformer 推理任务级 plan
- **低层**：equivariant action head 输出 motor command
- 对旋转/平移天然 invariant 或 equivariant
- 实验：相比非等变 baseline，在物体姿态扰动下成功率提升 25-40%；且**少 demo 即可学**（数据效率 3-5×）

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「等变性」是隐式数据增强**：
   - 模型对 SE(3) 变换天然等变 → 不需要数据增强教它"物体旋转后还是同样物体"
   - 节省数据 + 模型容量
   - **天然提供旋转/平移鲁棒性**

2. **分层 + 等变的组合**：
   - 高层 plan（"先去 picking pose，再 grasp"）应该 SE(3) invariant（任务语义不依赖坐标系）
   - 低层 action（具体关节命令）应该 SE(3) equivariant（坐标系变化时 action 相应变化）
   - 这种分层等变设计**理论优雅、实践有效**

3. **对 cross-embodiment 的隐含意义**：
   - 不同 embodiment 的 base frame 不同 → SE(3) 等变模型可直接处理坐标系差异
   - 与 EmbedMorph 的 morphology 嵌入互补
   - **几何不变性 → 隐式跨机体鲁棒**

### 方法论

#### 等变性正式定义
对 SE(3) 群作用 $g \in SE(3)$，policy $\pi$ 满足：
$$
\pi(g \cdot s) = g \cdot \pi(s)  \quad \text{(equivariant)}
$$
或
$$
f(g \cdot s) = f(s)  \quad \text{(invariant, for high-level features)}
$$

#### 架构
```
Input: Point cloud + state + language
  ↓
SE(3)-Equivariant Encoder (EqGNN / EquiTransformer)
  ↓
High-level: Invariant features → Task plan token
Low-level: Equivariant features → Motor command
```

实现细节：用 SE(3)-Transformers (Fuchs 2020) 或 PaiNN-style 等变 GNN。

### 实验

#### Object pose perturbation robustness
| Method | Original Pose | ±30° Rotated | ±10cm Translated |
|---|---|---|---|
| Diffusion Policy | 85% | 50% | 60% |
| OpenVLA | 78% | 55% | 65% |
| **HEP** | **88%** | **80%** | **82%** |

#### Sample efficiency
| # Demos | Diffusion | OpenVLA | HEP |
|---|---|---|---|
| 10 | 20% | 25% | 60% |
| 100 | 65% | 75% | 88% |

数据效率 3-5×。

### B7 视角下的定位

```
B7 主流方法（数据驱动）：
  HPT / X-VLA / EmbedMorph ── 通过形态/数据多样性学跨机体

B7 几何流派（结构归纳偏置）：
  HEP (本文)               ── 通过等变性天然处理几何变换
```

两个流派可组合：等变 backbone + soft prompt embodiment。

### 与曦源的关联

1. **直接关联：等变性 = 几何扰动鲁棒**：
   - 曦源关心视觉/物理扰动鲁棒
   - 物体姿态偏移是常见 failure mode
   - HEP 的等变设计天然鲁棒
   - **可借鉴**：在 SmolVLA 中加入 SE(3)-equivariant 视觉编码器（如 EqGNN）

2. **与 RobustVLA 的对比/协同**：
   - RobustVLA：通过对抗训练学鲁棒
   - HEP：通过架构归纳偏置学鲁棒
   - **本质不同**：一个是数据驱动，一个是结构驱动
   - **协同可能**：等变 backbone + worst-case fine-tune = 双重鲁棒
   - 这是一个有 thesis 价值的方向

3. **对 cross-embodiment 的潜在贡献**：
   - 不同 embodiment 的工作空间几何不同
   - 等变模型对坐标系变换鲁棒
   - **可成为 cross-embodiment 鲁棒训练的新组件**

4. **工程实施成本高**：
   - SE(3)-Transformer 实现复杂，需 GPU 优化的等变 conv
   - 与 SmolVLA / OpenVLA pipeline 集成成本高
   - **不建议作为短期路线**

### 待解问题

1. **等变性 vs 表达力的权衡**：严格等变限制模型容量，可能损失非几何相关的能力
2. **部分等变**：是否可以"软等变"（approx equivariance）？
3. **VLA 中的等变**：当前等变工作多基于 point cloud，VLA 用 RGB，如何把等变思想迁移到 image

### 复现挑战

- 需 e3nn 或 escnn 等等变深度学习库
- 数据需 point cloud 或 RGBD 输入
- 等变 layer 的 GPU 优化不如标准 transformer

%% end my-thoughts %%

## 🔗 关联笔记

- **B7 时间线**：[[2023-10_OpenX-Embodiment_Padalkar]], [[2024-09_HPT_Wang]], [[2025-10_X-VLA_Anonymous]], [[2026-03_EmbedMorphology-Transformer_Anonymous]]
- **等变理论前驱**：SE(3)-Transformer (Fuchs 2020), PaiNN (Schütt 2021)
- **鲁棒性协同**：[[2026_RobustVLA_Guo]] (data-driven robust)
- **VLA 主线**：[[2025_SmolVLA]], [[2024-06_OpenVLA_Kim]]

## 📌 Action Items

- [ ] 阅读 SE(3)-Transformer 原文（理解等变 attention）
- [ ] 评估：在 SmolVLA 中加 SE(3)-equivariant 视觉编码器的工程成本
- [ ] 思考：「等变 backbone + RobustVLA worst-case」组合的研究价值
- [ ] 实验：HEP vs OpenVLA 在 LIBERO-Plus 几何扰动 benchmark 上对比

%% Import Date: 2026-05-26 %%
