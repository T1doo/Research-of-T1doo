---
title: "Compressor-VLA: Instruction-Guided Visual Token Compression for Efficient Robotic Manipulation"
authors: "Juntao Gao, Feiyang Ye, Jing Zhang, Wenjing Qian"
year: "2025"
venue: "arXiv 2511"
arxiv: "2511.18950"
status: "已精读"
tier: "⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, token-pruning, instruction-conditioned]
---

# Compressor-VLA

> [!info] 元信息
> - **作者**：Juntao Gao, Feiyang Ye, Jing Zhang, Wenjing Qian
> - **arXiv**：[2511.18950](https://arxiv.org/abs/2511.18950)

## 📄 TL;DR

Compressor-VLA 提出指令条件化的混合视觉 token 压缩框架，专门解决 VLA 部署的算力瓶颈。架构上 dual-module：**Semantic Task Compressor** 蒸馏与任务相关的整体语境，**Spatial Refinement Compressor** 保留细粒度空间细节。压缩过程根据自然语言指令动态调整，选择性保留任务关键信息。LIBERO 上性能可比 baseline，但 **FLOPs 降低 59%、视觉 token 数减少超 3×**，且在真实双臂机器人上验证。可视化显示模型注意力会根据指令"steer"到不同物体上。

## 🧠 我的思考

### 核心观点

1. **指令是 VLA 视觉压缩的"自然 router"**：同一帧图像在不同指令下应该激活不同的视觉 token 子集（"拿杯子" vs "按按钮"），但现有方法大多用单一压缩策略。Compressor-VLA 显式把指令嵌入到压缩决策中，让 token 选择 condition on language。
2. **Dual-module 解耦语义与空间**：语义任务压缩器输出全局上下文（少量 high-level token），空间细化压缩器保留 fine-grained patch（用于精确定位）。这种"语义 + 空间"双通道是与 VLA-Pruner、VLA-IAP 单一压缩准则的差异。
3. **3× token 缩减 + 59% FLOPs 节省**：这个压缩比已经接近 token efficiency 的极限，意味着原始 VLA 视觉 token 中至少 2/3 是冗余的。

### 方法论关键

- **Semantic Task Compressor**：可能用 cross-attention 让 instruction embedding query 视觉特征，输出 N_sem 个全局 token（N_sem 远小于原始 patch 数）。
- **Spatial Refinement Compressor**：保留与语义 token 空间对齐的 fine-grained patch，可能用 deformable attention 或 top-K 选择。
- **Final visual representation**：$V_{out} = [V_{sem}, V_{spatial}]$ 拼接后送 action decoder。
- **训练目标**：BC loss 同步训练压缩器与 policy，可能加 distillation loss 让压缩后输出逼近 full token 输出。

### 关键实验数据

| Method | LIBERO Avg | FLOPs | Token Count |
|---|---|---|---|
| Baseline (full tokens) | ~SOTA | 100% | 1× |
| **Compressor-VLA** | **comparable** | **41% (-59%)** | **<33% (>3× reduction)** |

真机双臂任务额外验证可部署性。

### 与曦源（FocusVLA 主线）的关联

Compressor-VLA 与 FocusVLA 在「dual-module 设计」上**思想撞车**但实现路径不同：
- FocusVLA：patch-level top-K（空间）+ channel-level gate（特征通道）
- Compressor-VLA：semantic compressor（全局）+ spatial refinement（细粒度）

两者都在做"双层 attention"，但维度不同：FocusVLA 在 patch & channel 两个 axis，Compressor-VLA 在 semantic abstraction level。可以考虑融合——在 FocusVLA Focus Attention 之前先用 Compressor-VLA 的 semantic compressor 做一次 instruction-conditioned 粗筛，再用 Focus Attention 做精细 top-K。

**潜在弱点**：指令条件化压缩在"指令模糊 / 多步指令"下可能失效——如果 instruction 不够明确（"清理桌面"），semantic compressor 无法精确路由。这是个潜在 failure mode，值得做 ablation。

**研究空白**：Compressor-VLA 没在多扰动 OOD 设定下评测。指令是固定的，但视觉端有扰动时，semantic compressor 的语义匹配是否仍准？——这是个鲁棒性研究切入点。

### 待解问题

- Semantic Task Compressor 是用 Q-Former 风格还是 cross-attention pooling？
- 3× token 缩减后，复杂长序列任务（LIBERO-Long）性能是否仍稳定？
- 与 FocusVLA 直接对比（同 backbone、同 benchmark）会如何？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2025-11_VLA-Pruner_Liu]]
- [[2026-03_VLA-IAP_Cheng]]
- [[2025-12_LUVC_Zheng]]
