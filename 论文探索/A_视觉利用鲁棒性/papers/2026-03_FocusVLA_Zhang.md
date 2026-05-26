---
title: "FocusVLA: Focused Visual Utilization for Vision-Language-Action Models"
authors: "Yichi Zhang, Weihao Yuan, Yizhuo Zhang, Xidong Zhang, Jia Wan"
year: "2026"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2603.28740"
arxiv: "2603.28740"
venue: "arXiv 2026-03-30 (under review)"
citekey: "zhangFocusVLAFocusedVisual2026"
itemType: "preprint"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 导航星"
tags: [literature, T1D, 主线A, 视觉利用, 注意力机制, FocusVLA, 导航星]
---

# FocusVLA — 导航星精读笔记

> [!info] 元信息
> - **作者**：Yichi Zhang, Weihao Yuan*, Yizhuo Zhang, Xidong Zhang, Jia Wan
> - **机构**：哈工大深圳 / DaiMon Robotics / 南京大学 / 中国人民大学
> - **日期**：2026-03-30 (arXiv)
> - **arXiv**：[2603.28740](https://arxiv.org/abs/2603.28740)
> - **本地 PDF**：`论文探索/_pdf_cache/FocusVLA_2603.28740.pdf`
> - **导师定位**：曦源项目的导航星——指向「视觉利用率 → 隐性鲁棒性」这条工程可行、一年期可产出的路线

## 📄 Abstract

VLA 模型在精细操作上表现欠佳，根本原因不是视觉编码器质量，而是 **视觉信息利用机制**。论文识别三大瓶颈：(1) 架构偏差让模型绕开视觉细节；(2) 过多视觉 token 让注意力难以聚焦；(3) 任务无关视觉信息引入噪声。FocusVLA 提出两个机制对症下药：**Modality Cascaded Attention** 消除 shortcut 路径，**Focus Attention** 动态选 task-relevant patch 同时抑制 channel noise。LIBERO 多权重 98.7%、单权重 97.0% （0.5B 参数），打败 7B 的 OpenVLA-OFT / Spatial Forcing；RoboTwin 双臂任务 SOTA；真机背景/空间/目标变化均胜出 baseline；收敛速度 1.5× 提升（LIBERO-Spatial 上 5×）。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「不是编码器不行，是用法不对」**——这是本文最有破坏性的论断。前驱 OpenVLA-OFT 直接 *在 action 生成阶段省略视觉特征* 仍能跑通 LIBERO；VLA-Adapter 的 mixed attention 让 action query 抢走全部注意力、视觉细节被绕过。这暗示当前 VLA 的视觉鲁棒性问题**根源在架构 shortcut**，不在数据 / encoder / 训练扰动。FocusVLA 通过架构改造让模型"被迫看图"。

2. **Token 数量与 attention 质量是 trade-off**：512 默认 token → 256 最优、128 不够。这反过来说明 VLA 的视觉信号其实有大量冗余、但盲目剪枝又会丢失关键细节。Patch-level top-K + channel-level element-wise gate 的组合点正是这个平衡的工程化。

3. **顺序 cascaded attention 优于 mixed attention**：自注意力机制中并行混合不同模态会形成"信息走捷径"——action query 会偏好同质来源（action embedding）。论文用串行的 `H_A → H_AQ → H_V` 切断了这条快路。这对所有 transformer-based VLA 都有借鉴价值（包括 RobustVLA、π0、OpenVLA）。

### 方法论（要重现的关键技术细节）

#### Modality Cascaded Attention
原 VLA-Adapter（公式 1-5）将 `[C^V_t, C^AQ_t, A_t]` 拼接进单次 attention，导致 action 注意力分布偏向 query embedding。FocusVLA 替换为顺序独立 attention：

$$
\begin{aligned}
H_A &= \text{Attn}(A_t, A_t) \\
H_{AQ} &= \text{Attn}(A_t, C^{AQ}_t) \\
H_V &= \text{Attn}(A_t, C^V_t) \\
\hat{A}_t &= \sigma_{fusion}([H_A, H_{AQ}, H_V]) \\
A_{t+1} &= \text{FFN}(\hat{A}_t) + A_t
\end{aligned}
$$

**机制层面看为什么 work**：每个 modality 都对 action latent 做独立 cross-attention，融合在最后；意味着模型不能"偷懒"用一种模态替代另一种，必须分别提取信息。

#### Focus Attention（patch + channel 双层）

**Patch-level top-K**：
$$
W_V = \text{Softmax}\big(\text{TopK}\big((\sigma_q(A_t)) (\sigma_k(C^V_t))^T\big)\big)
$$

- query 来自 action latent，key 来自 **深层** VLM 视觉特征（语义对齐）
- value 来自 **浅层** 视觉 backbone 输出（保留 spatial detail）
- K=256（从 512）是 LIBERO-Long 上消融最优点

**Channel-level element-wise gate**：
$$
H'_V = H_V \odot \sigma_g(A_t)
$$

- σ_g 是 MLP，输出 element-wise 门控向量
- 击败 single-parameter gate（97.3% vs 98.2%）和 head-wise gate（97.1% vs 98.2%）

### 实验关键数据（对标 baseline 的硬数字）

#### LIBERO 多权重（每任务一个 policy）
| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Spatial Forcing (7B) | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| VLA-Adapter-Pro (0.5B) | 99.6 | 99.6 | 98.2 | 96.4 | 98.5 |
| **FocusVLA (0.5B)** | **99.6** | **100.0** | **98.8** | **96.2** | **98.7** |

#### LIBERO 单权重（一个 shared policy）
| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| VLA-Adapter-Pro (0.5B) | 98.8 | 96.2 | 95.6 | 91.6 | 95.6 |
| EVO-1 (0.5B) | 92.7 | 97.7 | 96.3 | 92.3 | 94.8 |
| **FocusVLA (0.5B)** | **99.2** | **98.4** | **97.0** | **92.4** | **97.0** |

#### 消融（LIBERO-Long）
| 组件 | 效果 |
|---|---|
| Mixed → Cascaded Attention | 93.6 → 97.0 (+3.4 pp) |
| Patch-level 256 tokens（vs 128 / 512） | 256 最优 |
| Channel-level element-wise gate | 98.2% vs 1-param 97.3% / head-wise 97.1% |
| VLM + DINOv2 + SigLIP | 98.7% （最佳 encoder 组合） |

#### 训练效率
- LIBERO 平均 1.5× 加速达到 baseline 最优
- LIBERO-Spatial 上 5× 加速

### 与我研究（曦源 EfVLA-鲁棒性）的关联

1. **完美契合"轻量 backbone × 工程可行"约束**：0.5B 参数 + LoRA + 标准 VLM backbone（Qwen2.5-0.5B），与 SmolVLA 同量级，46 GB 显存内完全 batch=4-8 微调可行。
2. **隐性鲁棒性的工程化路径**：FocusVLA 没显式做对抗扰动训练，但通过架构改造**间接**让模型对视觉扰动鲁棒——这正是导师推这条路线的核心思路。验证假设：「**让 attention 真正看任务相关区域 → 自然抑制背景扰动 / 视觉噪声 → 鲁棒性提升**」。
3. **可立刻搬到 RobustVLA / π0 上做组合**：Modality Cascaded Attention 是架构层改造，可以接在任何 transformer-based VLA 上；patch-level top-K 是推理时的视觉计算节省。
4. **正向预兆**：FocusVLA 把 1.5× 收敛加速 + 0.5B 参数打败 7B baseline，证明"小聪明 + 好架构 > 大力出奇迹"。这对我们的 BC/CMR 申请书极有说服力。

### 待解问题 / 后续追读

#### FocusVLA 自己留下的空白（可成为你的研究空间）

1. **「鲁棒性对背景 + 纹理同时变化仍有限」**（作者自述局限）— **这就是你的研究空白！** 你可以做 FocusVLA + 对抗训练 / domain randomization 联合，专门解决多扰动叠加。
2. **「对初始状态敏感」**（作者自述局限）— 机器人臂不在标准初始位置就失败。这是 VLA 鲁棒性的一个具体 failure mode，可做实证研究。
3. **「VLM 内部的视觉利用没涉及」**（作者自述局限）— FocusVLA 只改了 action head 那一层 attention；VLM 内部更深的视觉 token 利用还是黑盒。可探索 LoRA-tuning VLM 内部 attention 是否进一步提升鲁棒性。
4. **没有显式跑 robustness benchmark（VLA-Risk / LIBERO-Plus）**— FocusVLA 的鲁棒性都是"side effect"，没在标准 robustness benchmark 上量化。**这是你立项的关键定位点**：用 VLA-Risk + LIBERO-Plus 系统评测 FocusVLA 的鲁棒增益，与 RobustVLA 等显式鲁棒方法对比。

#### 必须追读的衍生 / 相关工作

1. **VLA-Adapter 原论文**（FocusVLA 的 baseline）— 理解 mixed attention 设计动机
2. **OpenVLA-OFT**（参考 baseline，7B vs 0.5B）— 理解"省略视觉特征"的设计选择
3. **Spatial Forcing / EVO-1**（强 baseline）— 主流 VLA 当前 ceiling
4. **VLA-Pruner / VLA-IAP / SemanticVLA**（同代 token pruning 工作）— 与 FocusVLA 的方法学差异
5. **Gaze-Regularized VLA**（A2）— 另一种 attention guidance 路径，是 FocusVLA 的 alternative
6. **InSpire / CAST / CF-VLA**（A6 spurious correlation）— FocusVLA 隐性解决的问题的显式诊断工具

### 设计风险 / 复现挑战

- **Modality Cascaded Attention 的实现复杂度**：需要重写 transformer attention block，不是 plug-and-play。需要 PyTorch attention 模块的深度修改。
- **Top-K patch selection 的可微化**：TopK 是不可微操作，论文用 softmax + topK，需要确认梯度回传。
- **VGGT 额外特征**：论文用了 DINOv2 + SigLIP + VGGT，VGGT 是新工作（3D vision foundation model），多个特征流的融合训练资源开销大。
- **代码未释**：abstract 说"will be made public"，目前 GitHub 不可知。我们要么等代码、要么自己 from-scratch 复现 attention 模块。

%% end my-thoughts %%

## 🔗 关联笔记

- **导师推荐路径**：[[2026_RobustVLA_Guo]] 是另一条主线 B 的锚点（数据训练）；FocusVLA 是主线 A（架构 / 视觉利用）
- **同代竞争**：[[2025_VLA-Pruner]], [[2026_VLA-IAP]], [[2025_SemanticVLA]], [[2024_EF-VLA]]
- **同源 attention 思想**：[[2026_Gaze-Regularized_VLA]], [[2024_AutoFocus-IL]]
- **基础架构**：[[2024_OpenVLA]], [[2025_SmolVLA]]（同 0.5B 量级 backbone）
- **FocusVLA 留的空白**：[[2026_LIBERO-Plus]], [[2026_VLA-Risk]] benchmark 评测
- **自我反思的子方向**：[[A6_shortcut_learning_总结]]，[[C-ii_OOD_UQ_VLA]]

## 📌 Action Items

- [ ] 复现：让团队成员用 VLA-Adapter codebase 改 Cascaded Attention（2 周）
- [ ] 评测：跑 LIBERO-Plus + VLA-Risk 看 FocusVLA 鲁棒性增益（1 周）
- [ ] 对比：FocusVLA vs RobustVLA 在同一扰动集上的对比（4 周）
- [ ] 联合：FocusVLA + RobustVLA worst-case δ 训练（6 周）

%% Import Date: 2026-05-26 %%
