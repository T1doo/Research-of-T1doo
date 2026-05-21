---
title: "Adaptive Action Chunking at Inference-time for Vision-Language-Action Models"
authors: "Yuanchang Liang, Xiaobo Wang, Kai Wang, Shuo Wang, Xiaojiang Peng, Haoyu Chen, David Kim Huat Chua, Prahlad Vadakkepat"
year: "2026"
journal: ""
doi: "10.48550/arXiv.2604.04161"
citekey: "liangAdaptiveActionChunking2026"
zotero-link: "zotero://select/library/items/S348VPEN"
itemType: "preprint"
tags: [literature, T1D]
---

# Adaptive Action Chunking at Inference-time for Vision-Language-Action Models

> [!info] 元信息
> - **作者**：Yuanchang Liang, Xiaobo Wang, Kai Wang, Shuo Wang, Xiaojiang Peng, Haoyu Chen, David Kim Huat Chua, Prahlad Vadakkepat
> - **日期**：2026-04-10
> - **来源**：
> - **DOI**：[10.48550/arXiv.2604.04161](https://doi.org/10.48550/arXiv.2604.04161)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/S348VPEN)
> - **Citekey**：`@liangAdaptiveActionChunking2026`

## 📄 Abstract

In Vision-Language-Action (VLA) models, action chunking (i.e., executing a sequence of actions without intermediate replanning) is a key technique to improve robotic manipulation abilities. However, a large chunk size reduces the model's responsiveness to new information, while a small one increases the likelihood of mode-jumping, jerky behavior resulting from discontinuities between chunks. Therefore, selecting the optimal chunk size is an urgent demand to balance the model's reactivity and consistency. Unfortunately, a dominant trend in current VLA models is an empirical fixed chunk length at inference-time, hindering their superiority and scalability across diverse manipulation tasks. To address this issue, we propose a novel Adaptive Action Chunking (AAC) strategy, which exploits action entropy as the cue to adaptively determine the chunk size based on current predictions. Extensive experiments on a wide range of simulated and real-world robotic manipulation tasks have demonstrated that our approach substantially improves performance over the state-of-the-art alternatives. The videos and source code are publicly available at https://lance-lot.github.io/adaptive-chunking.github.io/.

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点
- 「固定 chunk size」是当前 VLA 一个被低估的设计偏置：大 chunk 牺牲响应性、小 chunk 引发 mode-jumping，AAC 是首次把 chunk 长度作为推理时可优化变量的工作。
- 用 **action entropy** 衡量预测稳定性是一个极轻量的工程信号：离散动作用 Shannon 熵、连续动作用高斯微分熵 $E_t=\frac{1}{2}\log[(2\pi e)^d\det \Sigma_t]$，N=20 个采样即可估计协方差。
- 触发规则 $h^* = \max(\arg\max_h(\bar E_{h+1}-\bar E_h), \xi)$ 的核心是「下一步熵相对当前显著上升时立刻 replan」，$\xi$ 是动作幅度门槛，防止微动抖动触发。
- 在 GR00T 基座上，LIBERO 平均 94.1%→95.0%、RoboCasa 59.7%→62.0%、**真实世界 67%→82%**，真实场景增益远大于仿真，说明 entropy 在分布外更有用。
- 边际成本仅 +23 ms（N=20 时 106 ms vs baseline 83 ms），是「白送」的推理增强。

### 方法论
- 关键公式：动作分布需可估计；离散动作直接用预测概率，连续动作需要多次采样（默认 N=20）得到协方差矩阵 $\Sigma_t$。
- 决策算法：对当前 chunk 内每个潜在长度 h 计算平均熵 $\bar E_h$，找熵增量最大的 h；若所有 h 的动作幅度都 < $\xi$，强制取 $\xi$ 对应的下界。
- 训练：**AAC 不需要重训** — 它只改推理流程，可挂在任何输出动作分布的 VLA 之上（GR00T、π0、SmolVLA 都适用）。
- 实验细节：N=20 是性能/延迟的甜点，N=5 已能拿大部分收益；Long-horizon 任务（LIBERO-Long 88.8→92.8）和 Rotation 任务（57.6→61.4）增益最大。
- 推理时间：106 ms（N=20）vs 83 ms（N=1），25% 额外开销换 0.9～15 pp 成功率提升。

### 与我研究的关联（T1D）
- **方向一核心论文**：曦源项目「动态与自适应推理」方向的标杆工作，直接定义了 chunk-level 自适应。
- **硬件契合度极佳**：AAC 只改推理流程不改训练，46 GB 显存约束下完全无影响；N=20 采样可并行 batch 化，单卡 A100 充分够用。
- **可复用模块**：
    - entropy 计算函数（10 行代码可移植到 SmolVLA action expert 输出端）
    - chunk 截断决策器（独立模块，可作为 LeRobot policy wrapper）
    - 协方差估计（高斯微分熵公式直接套用）
- **天然扩展点**：AAC + SmolVLA async inference 的组合（AAC 决定何时停 chunk，async 决定何时刷新观测），是本立项最直接的「切入点 A」。

### 待解问题 / 后续追读
- 论文未在 **LIBERO-Plus** 上做鲁棒性评测，ξ 阈值在视觉扰动下的稳定性未知（曦源项目的核心研究空白）。
- N 个采样的并行实现细节论文一笔带过，需追代码看是否做了 KV cache 复用 / batch parallelism。
- 双臂场景（RoboTwin）下左右臂的 entropy 是否应独立计算未涉及，是天然扩展点。
- 追读：GR00T (NVIDIA) 原文以理解 baseline 实现；项目页 https://lance-lot.github.io/adaptive-chunking.github.io/ 的源码组织结构。
- 待对比：AAC 与 SmolVLA async inference 的组合实测；与 A2A 1-step 推理结合时 N=20 采样如何摊销。

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%


### 📍 p.1 (Yellow)


![[References/images/liangAdaptiveActionChunking2026-undefined-x65-y393.png]]



---


%% end annotations %%


%% Import Date: 2026-05-21T17:07:44.824+08:00 %%
