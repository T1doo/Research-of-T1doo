---
title: "VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference"
authors: "Ziyan Liu, Yeqiu Chen, Hongyi Cai, Tao Lin, Shuo Yang, Zheng Liu, Bo Zhao"
year: "2025"
venue: "arXiv 2511"
arxiv: "2511.16449"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, token-pruning]
---

# VLA-Pruner

> [!info] 元信息
> - **作者**：Ziyan Liu, Yeqiu Chen, Hongyi Cai, Tao Lin, Shuo Yang, Zheng Liu, Bo Zhao
> - **机构**：未在 abstract 中明示
> - **arXiv**：[2511.16449](https://arxiv.org/abs/2511.16449)
> - **代码**：未确认

## 📄 TL;DR

VLA 推理慢的根因之一是视觉 token 数量过多。VLA-Pruner 指出现有 VLM-style 剪枝只看「语义显著性」，忽略 VLA 的 dual-system 本质：语义理解与动作执行对视觉信息的依赖**不同**。论文提出 dual-level 重要度准则——semantic level 用 vision-language prefill attention 找语义对齐 token，action level 用 temporal smoothing 估计 action decode attention，两者协同保留对「看懂」和「做动」都关键的紧凑 token 集合。在多个 VLA 架构与多类机器人任务上 SOTA。

## 🧠 我的思考

### 核心观点

1. **Dual-objective 是 VLA token pruning 的范式分水岭**：VLM-style 剪枝（FastV、SparseVLM 等）只优化"语义重要度"，但 action decode 在不同时间步对同一帧的关注点会迁移（如先看物体、再看 gripper 位置）。仅靠 prefill attention 无法捕捉这种 action-induced 的 token 重要性变化。
2. **Temporal smoothing 是关键工程化点**：单步 action attention 噪声大，跨时间步平滑后得到更稳健的 token 重要度，避免"这一步看到、下一步剪掉"导致的轨迹抖动。
3. **架构无关性**：作者声称 VLA-Pruner 在多个 VLA 架构上都 SOTA，说明 dual-level 准则不依赖特定 backbone，可作为通用 inference-time 加速组件。

### 方法论关键

VLA-Pruner 的核心是两阶段重要度评分：
- **Semantic score** $s^{sem}_i = \text{Attn}_{prefill}(\text{lang}, v_i)$ —— 沿用 FastV 的 prefill attention 思路，识别与指令对齐的 token。
- **Action score** $s^{act}_i = \text{EMA}_t(\text{Attn}_{decode}(a_t, v_i))$ —— 用指数移动平均跨时间步聚合 action query 对视觉 token 的注意力。
- **融合**：$s_i = \alpha \cdot s^{sem}_i + (1-\alpha) \cdot s^{act}_i$，取 top-K 保留。

关键在 action attention 的提取——需要 hook decoder 内部的 cross-attention，工程上需深入 attention 模块。

### 关键实验数据

| Method | LIBERO Avg | Speedup |
|---|---|---|
| Full tokens baseline | ~SOTA | 1× |
| FastV (VLM-style) | 略降 | ~1.3× |
| **VLA-Pruner** | **SOTA on multiple VLAs** | 未在 abstract 给具体数 |

具体数字需读正文，但作者强调"prevents critical action-execution information from being discarded"是与 baseline 主要区别。

### 与曦源（FocusVLA 主线）的关联

VLA-Pruner 与 FocusVLA 的 Focus Attention 在思路上**高度相关但定位不同**：
- FocusVLA 在 action head 内部做 patch-level top-K，是**训练时**架构改造，与 attention gate 一起端到端训练。
- VLA-Pruner 是**inference-time** training-free 加速，无需重训。
- 两者可叠加：FocusVLA 训练得到 task-relevant attention 分布，VLA-Pruner 在部署时再剪一层，理论上能进一步压 token 数。

**复用价值**：dual-level 评分中的 "action attention via temporal EMA" 是个独立模块，可移植到 FocusVLA 的 Focus Attention 里作为更平滑的 top-K 选择信号——目前 FocusVLA 的 top-K 是逐步独立计算，可能存在时序抖动。

**研究空白**：VLA-Pruner 没在 robustness benchmark（如 LIBERO-Plus、VLA-Risk）评测过——视觉 token 剪枝在视觉扰动下会不会反而放大噪声？这是个值得做的实证课题。

### 待解问题

- VLA-Pruner 的 action attention 是从哪一层 decoder 提取？跨层平均还是单层？
- 与 [[2026-03_VLA-IAP_Cheng]] 的 geometric prior pruning 对比 LIBERO 上谁优？
- 在视觉 OOD 场景（背景变化、光照扰动）下，dual-level 剪枝是否仍鲁棒？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2026-03_VLA-IAP_Cheng]]
- [[2025-11_Compressor-VLA_Gao]]
- [[2025-12_LUVC_Zheng]]
