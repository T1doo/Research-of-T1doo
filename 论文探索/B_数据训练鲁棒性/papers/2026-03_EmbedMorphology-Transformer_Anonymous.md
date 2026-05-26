---
title: "Embedding Morphology into Transformers for Cross-Embodiment Generalization"
authors: "Anonymous (under review)"
year: "2026"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2603.00182"
arxiv: "2603.00182"
venue: "arXiv 2026-03"
citekey: "anonymousEmbedMorphology2026"
itemType: "preprint"
status: "精读完成"
tier: "⭐⭐ B7 形态嵌入路线"
tags: [literature, T1D, 主线B, B7, 形态嵌入, morphology, transformer, kinematic_graph]
---

# Embed Morphology into Transformers — 形态嵌入精读笔记

> [!info] 元信息
> - **arXiv**：[2603.00182](https://arxiv.org/abs/2603.00182)
> - **B7 价值定位**：把**机器人形态（kinematic graph / URDF）显式编码**到 transformer，让 policy 自动适应任意机体几何

## 📄 Abstract

cross-embodiment VLA 的隐藏假设：模型从大量数据中**隐式**学到形态信息。本文提出**显式**方案：
1. 把每个机体的 URDF 解析为 **kinematic graph**（关节 + 链接 + 自由度）
2. 用 **graph transformer** 编码 graph → morphology embedding
3. 将 embedding 拼接到 VLA 输入
4. 训练时见过的形态可正向迁移，**zero-shot 到未见过的形态**（如 8DoF arm vs 7DoF arm）

效果：在 6 个未见 embodiment 上 zero-shot SR 平均 45%（vs OpenVLA 15%），大幅领先。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「显式 morphology 编码」是 cross-embodiment 的归约简化**：
   - HPT / X-VLA 让模型「隐式」学每个 embodiment 的特性
   - 本文「显式」给模型一个 URDF graph 描述
   - 类比：HPT/X-VLA 是「黑盒记每个机体的样子」，本文是「给模型一张机体说明书」
   - **更可解释、更可扩展**

2. **kinematic graph 作为通用机体语言**：
   - 任何机器人都可用 URDF（链接 + 关节 + 几何）描述
   - graph transformer 编码后得到 morphology embedding（256-512 dim）
   - **这与 X-VLA 的 soft prompt 异曲同工**，但 X-VLA prompt 是 learned-from-scratch，本文是 from-graph-structure
   - **更好的归纳偏置**

3. **「形态等变」的隐含思想**：
   - 机器人执行任务时，形态决定了动作可行域
   - 显式 graph encoder 让 policy 自动适应形态变化
   - 与 [[2025_HierarchicalEquivariant]] 的等变思想呼应（不同实现路径）

### 方法论

#### 架构
```
URDF (per embod) → parse → Kinematic Graph G = (V, E)
                                ↓
                         Graph Transformer (GAT-like)
                                ↓
                         Morphology Embedding m (256 dim)
                                ↓
Image / Lang / State → Tokens → [m; tokens] → VLA Transformer
                                ↓
                         Action (universal output)
```

#### Graph encoding
- Node features：link 几何（质量、惯性、长度）+ joint type（revolute / prismatic / fixed）
- Edge：connectivity + transform
- GAT 多层聚合，最后 pooling 得 morphology embedding

#### 训练
- 见过 20 种 morphology
- Loss：BC + 可选的 morphology contrastive loss（相似 morphology 应有相似 embedding）

### 实验

#### Zero-shot 到新 morphology
| Embodiment Type | OpenVLA | HPT | **EmbedMorph** |
|---|---|---|---|
| 7DoF → 7DoF new | 30% | 60% | **75%** |
| 7DoF → 6DoF | 10% | 25% | **50%** |
| 7DoF → 8DoF redundant | 5% | 30% | **65%** |
| Single arm → bimanual | 0% | 20% | **40%** |

### B7 时间线中的位置

```
2024-09: HPT          ── per-embod tokenizer (隐式)
2025-10: X-VLA        ── soft prompt (隐式)
2026-03: EmbedMorph   ── kinematic graph (显式)  ← 本文
```

本文是「显式形态化」流派的代表。与 HPT / X-VLA 路线对比：
| 维度 | HPT/X-VLA | EmbedMorph |
|---|---|---|
| 形态编码 | 隐式 learned | 显式 from URDF |
| 新机体迁移 | 需 100-1000 demo | 可 zero-shot |
| 可解释性 | 黑盒 | 可解释 (graph) |
| 工程复杂度 | 中 | 高 (需 URDF) |

### 与曦源的关联

1. **对鲁棒性的隐性贡献**：
   - 显式形态化 → 让模型"知道自己的物理形态" → 对形态扰动鲁棒
   - 想象：把单 Franka 的 URDF 加入噪声（关节长度 ±5%），看模型是否仍可执行
   - **可做实验**：morphology robustness benchmark（新方向）

2. **LeRobot 多机体支持的工程优势**：
   - LeRobot 上的 Koch / SO-100 / Franka 都有 URDF
   - 用 EmbedMorph 可一次训练，跨所有 LeRobot 机体部署
   - **省去每机体单独 fine-tune 的麻烦**

3. **与 X-VLA / HPT 的可组合性**：
   - 不互斥：X-VLA soft prompt 可由 EmbedMorph graph embedding 初始化
   - 这是个**新颖的组合方案**

4. **对曦源的选择**：
   - **当前不建议作为主路线**：URDF 解析 + graph transformer 增加工程复杂度
   - **可作为长期 future work**：跨机体研究阶段考虑

### 待解问题

1. **URDF 标准化**：不同仿真器/真机的 URDF 格式不一致，预处理复杂
2. **未来研究方向**：morphology embedding 是否能扩展到 soft robot / 变形机器人
3. **graph encoder 选择**：GAT vs Graph Transformer vs SE(3)-Transformer 的对比

### 复现挑战

- 需 PyG (PyTorch Geometric) 处理 graph
- URDF parser：可用 urdfpy / urdf_parser_py
- 训练数据中需 URDF 与 demo 对应关系（不是所有 OXE 子集都有）

%% end my-thoughts %%

## 🔗 关联笔记

- **B7 时间线**：[[2023-10_OpenX-Embodiment_Padalkar]], [[2024-08_CrossFormer_Doshi]], [[2024-09_HPT_Wang]], [[2025-10_X-VLA_Anonymous]]
- **B7 等变路线**：[[2025_HierarchicalEquivariant]]
- **形态化思想前驱**：MetaMorph (Gupta et al. 2022), Graph-Based Locomotion (Wang et al. 2018)
- **可组合方案**：[[2025-10_X-VLA_Anonymous]] + EmbedMorph

## 📌 Action Items

- [ ] 阅读 MetaMorph (Gupta 2022) 前驱
- [ ] 评估：URDF + graph transformer 集成到 SmolVLA 的工程成本
- [ ] 思考：morphology robustness benchmark 是否值得作为单独研究方向
- [ ] 等待代码发布（双盲）

%% Import Date: 2026-05-26 %%
