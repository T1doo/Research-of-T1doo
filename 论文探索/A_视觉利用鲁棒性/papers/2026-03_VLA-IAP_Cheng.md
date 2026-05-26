---
title: "VLA-IAP: Training-Free Visual Token Pruning via Interaction Alignment for Vision-Language-Action Models"
authors: "Jintao Cheng, Haozhe Wang, Weibin Li, Gang Wang, Yipu Zhang, Xiaoyu Tang, Jin Wu, Xieyuanli Chen, Yunhui Liu, Wei Zhang"
year: "2026"
venue: "arXiv 2603"
arxiv: "2603.22991"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, token-pruning, training-free]
---

# VLA-IAP

> [!info] 元信息
> - **作者**：Jintao Cheng, Haozhe Wang, Weibin Li, Gang Wang, Yipu Zhang, Xiaoyu Tang, Jin Wu, Xieyuanli Chen, Yunhui Liu, Wei Zhang
> - **机构**：多家高校联合（含华南师大、港中大等）
> - **arXiv**：[2603.22991](https://arxiv.org/abs/2603.22991)

## 📄 TL;DR

VLA-IAP 是一个 **training-free** 的视觉 token 剪枝方法，主打"interaction alignment"。作者指出现有剪枝方法（依赖语义显著性或简单时序线索）忽略了机器人任务中**连续物理交互**这一核心约束。提出两大机制：(1) **Geometric prior**——保留结构锚点（如 gripper-object 接触区域），避免误剪视觉稀疏但几何关键的区域；(2) **Dynamic scheduling**——基于语义-运动对齐度动态调整剪枝强度，前期保守、后期激进。LIBERO 上 97.8% 成功率 + 1.25× 加速，最高 1.54× 加速且性能可比。

## 🧠 我的思考

### 核心观点

1. **"几何先验"是 VLA pruning 的盲点**：传统语义剪枝会丢失「与任务结构相关但语义不显著」的区域（比如桌面边缘的标记、机械臂关节投影）。这些区域在 VLM 注意力图上是低权重的，但对精细操作至关重要。VLA-IAP 用几何 prior 显式保护。
2. **Dynamic scheduling 体现了"任务阶段感知"**：动作早期需要全局规划信息（多保留），动作末期 gripper 接近目标时聚焦局部（可激进剪）。这是 VLA 推理时序的真实结构，被 VLA-IAP 显式建模。
3. **Training-free 是关键卖点**：与 FocusVLA、VLA-Pruner 都需 hook 内部 attention 不同，VLA-IAP 可以即插即用部署到任何现有 VLA 上，零训练成本。

### 方法论关键

- **Geometric prior 提取**：可能基于深度图 / RGB-D 几何特征 / SAM 或 SuperPoint 类关键点检测器，识别物理交互锚点对应的 patch。
- **Semantic-motion alignment**：定义某种度量 $\phi(t)$ 评估当前帧语义特征与动作目标的对齐度，alignment 高时剪枝可以激进。
- **保留集合**：$\mathcal{S}_t = \mathcal{S}^{geom} \cup \text{TopK}(\phi(t) \cdot s^{sem})$。

具体几何 prior 的实现细节需读正文 method 部分确认。

### 关键实验数据

| Setting | LIBERO Success | Speedup |
|---|---|---|
| Full baseline | ~97-98% | 1× |
| **VLA-IAP (default)** | **97.8%** | **1.25×** |
| **VLA-IAP (aggressive)** | comparable | **1.54×** |

跨多个架构（OpenVLA、VLA-Adapter 等）、三个仿真环境、真机平台均验证。

### 与曦源（FocusVLA 主线）的关联

VLA-IAP 与 FocusVLA 形成 **training-free vs train-time** 的互补：
- FocusVLA：训练时改架构 → 学到 task-aware attention。
- VLA-IAP：推理时剪 token → 利用几何 prior 加速。

**模块复用**：VLA-IAP 的 **geometric prior** 是 FocusVLA 没有的维度——FocusVLA 的 Focus Attention 完全靠 learned attention 选 patch，对几何稀疏区域（薄边缘、低纹理表面）可能不敏感。可以把 geometric prior 作为 FocusVLA Focus Attention 的额外 bias，提升对接触敏感任务（peg-in-hole、cable routing）的鲁棒性。

**Dynamic scheduling 启发**：FocusVLA 的 top-K 是固定 K=256，VLA-IAP 显示 K 应该随任务阶段变化。可以扩展 FocusVLA 让 K 自适应。

**研究空白**：VLA-IAP 的 geometric prior 在视觉退化场景（深度噪声、光照异常）下还稳定吗？如果几何 prior 提取本身受扰，整体性能可能崩塌——这是 robustness 实验的好切入点。

### 待解问题

- Geometric prior 具体怎么算？深度图 + 边缘检测还是 learned saliency？
- 与 [[2025-11_VLA-Pruner_Liu]] 在同 benchmark 下的横向对比
- Training-free 方法的天花板在哪里？vs train-time methods

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2025-11_VLA-Pruner_Liu]]
- [[2025-11_Compressor-VLA_Gao]]
- [[2026-03_TPRL_Cao]]
