---
title: "Towards Lossless Ultimate Vision Token Compression for VLMs"
authors: "Dehua Zheng, Mouxiao Huang, Borui Jiang, Hailin Hu, Xinghao Chen"
year: "2025"
venue: "arXiv 2512"
arxiv: "2512.09010"
status: "已精读"
tier: "⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, token-pruning, VLM, lossless]
---

# LUVC

> [!info] 元信息
> - **作者**：Dehua Zheng, Mouxiao Huang, Borui Jiang, Hailin Hu, Xinghao Chen
> - **arXiv**：[2512.09010](https://arxiv.org/abs/2512.09010)（2025-12-09）

## 📄 TL;DR

LUVC（Lossless Ultimate Vision Token Compression）是面向 **VLM 通用场景**而非纯 VLA 的 token 压缩框架。诊断已有压缩方法的两大病灶：**position bias** 与 **class imbalance**。提出两个解药：(1) **Progressive Token Merging**——在 visual encoder 内做迭代 merge，空间正交 schema；(2) **Spectral Pruning**——在 LLM 内插 attention/similarity-free 的低通滤波器，渐进剪除冗余视觉 token，与 FlashAttention 兼容。整体训练-free，**LLM 端 2× 加速、精度几乎无损**。

## 🧠 我的思考

### 核心观点

1. **"Position bias" 是被忽视的失败模式**：很多基于 attention 的剪枝方法天然偏好图像中间或起始 token（因为 transformer 位置编码导致 attention 分布不均），结果剪掉的常是边缘但语义重要的 token。LUVC 用 spectral filter（attention-free）绕过这个问题。
2. **Encoder + LLM 联合压缩才能"ultimate"**：现有方法多集中在 LLM 端剪 token，但 visual encoder 输出本身已经 N=576+ patches，encoder 端不压缩等于浪费机会。Progressive merging 在 encoder 内做空间迭代合并，把 encoder 输出维度也降下来。
3. **Spectral pruning 与 FlashAttention 兼容**：这是工程上极重要的——传统基于 attention score 的剪枝需要拿到完整 attention matrix，与 FlashAttention 的 fused kernel 冲突。Spectral filter 是 attention-free，可以直接套在 FlashAttention 之后。

### 方法论关键

- **Progressive Token Merging（spatial-orthogonal）**：迭代地按行/列交替方向合并空间相邻 token，类似 ToMe 但有空间结构约束，避免破坏空间局部性。
- **Spectral Pruning（low-pass filter）**：把视觉 token 序列看作信号，用 DCT/FFT-like 变换分解到频域，低频对应主要语义、高频对应噪声/冗余，逐层只保留低频成分。
- **Layer-wise 渐进剪枝**：每层 LLM 都剪一点，浅层保留多、深层剪激进，符合"deep layers more abstract"的层次假设。

### 关键实验数据

| Configuration | Accuracy Δ | LLM Speedup |
|---|---|---|
| Baseline VLM | 0 | 1× |
| **LUVC** | **≈0 (lossless)** | **2×** |

跨多个 VLM 部署即用，无需 fine-tuning。具体 VLA 任务上的验证未在 abstract 给出。

### 与曦源（FocusVLA 主线）的关联

LUVC 是 **VLM 端通用方案**，不是专为 VLA 设计。但对 FocusVLA 主线有几个启示：
1. **Spectral pruning 思路可借鉴**：FocusVLA 的 patch-level top-K 是 hard selection，会损失被剪 token 的信息。如果改成 spectral low-pass，能"软压缩"，可能在视觉退化场景更鲁棒（噪声多在高频，低通自然抑制）。
2. **Progressive Encoder Merging 可前置到 FocusVLA**：FocusVLA 现在直接用 DINOv2/SigLIP/VGGT 多 encoder 输出，token 数已经爆炸。可以先用 LUVC 的 progressive merging 在 encoder 端预压缩，再进 Focus Attention。
3. **Position bias 是 FocusVLA 的潜在 failure mode**：FocusVLA 的 top-K 用的是 action query × key attention，同样有 position bias 风险。值得做一个 ablation：把任务关键物体放在画面边缘，看 FocusVLA 是否仍能选中。

**研究空白**：LUVC 在 LLM 通用任务上验证 lossless，但 VLA 任务（精细操作、连续 control）对 token 准确性要求远高于 VQA/captioning。Spectral pruning 在 VLA 上是否真的 lossless？这是个值得做的实证课题。

### 待解问题

- LUVC 是否在任何 VLA 任务上做过验证？
- Spectral filter 的具体频率截断点如何选？是固定 hyperparam 还是 learned？
- 与 [[2025-11_VLA-Pruner_Liu]] 比较：spectral vs attention-based 剪枝在 VLA 上谁优？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2025-11_VLA-Pruner_Liu]]
- [[2026-03_VLA-IAP_Cheng]]
- [[2025-11_Compressor-VLA_Gao]]
