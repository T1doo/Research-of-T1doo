---
title: "X-VLA: A Cross-Embodiment Vision-Language-Action Model with Soft-Prompted Transformer"
authors: "Anonymous (under review)"
year: "2025"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2510.10274"
arxiv: "2510.10274"
venue: "arXiv 2025-10 (likely ICLR/CoRL submission)"
citekey: "anonymousXVLA2025"
itemType: "preprint"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B7 最新跨机体 SOTA"
tags: [literature, T1D, 主线B, B7, X-VLA, 软提示, cross-embodiment, 2025最新]
---

# X-VLA — 软提示跨机体 transformer 精读笔记

> [!info] 元信息
> - **arXiv**：[2510.10274](https://arxiv.org/abs/2510.10274)
> - **B7 价值定位**：2025 最新跨机体范式，**用软提示（soft prompt）统一异构 embodiment**，比 HPT 的 per-embod tokenizer 更轻量

## 📄 Abstract

跨机体 VLA 面临两难：
- HPT 路线：每机体一个 tokenizer，可扩展但参数多
- CrossFormer 路线：单一 transformer + 变长 token，简单但 fragile

X-VLA 提出第三条路：**soft prompt**——每个 embodiment 学一个 learnable prompt vector（256-512 dim），插入到 transformer 输入前。trunk 完全共享，无 per-embod 参数。
- 数据：1.2M trajectories, 25+ embodiments
- 参数：1.5B trunk + 256 dim × 25 prompts = 1.5B (essentially)
- 新机体迁移：只需训 256 dim prompt（**最快的 adaptation**）
- 性能：与 HPT-XLarge 持平或略优

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（B7 最新范式）

1. **「Soft prompt」是 NLP 经典，搬到 robotics 是巧妙创新**：
   - NLP prompt tuning（Lester et al. 2021）：每个 task 一个 prompt vector
   - X-VLA 类比：每个 embodiment 一个 prompt vector
   - 优点：参数效率极高（每机体仅 256 floats）、共享 trunk 学到的能力最大化
   - 缺点：表达力受限，复杂异构（如机械臂 vs 飞行）可能不够

2. **prompt 的「身份」语义解释**：
   - prompt vector 编码 embodiment-specific info：DoF / sensor config / action space
   - trunk 接收 prompt → 适配该 embodiment 的处理流程
   - 类似「告诉模型『你现在是 Franka 机器人』」的 in-context 提示

3. **「parameter-efficient cross-embodiment」是 2026 趋势**：
   - HPT：每机体 30M tokenizer
   - X-VLA：每机体 256 dim prompt
   - **预测**：未来会有「meta-prompt」「embodiment embedding」更轻量方案

### 方法论

#### 架构
```
Embodiment ID → Embedding Layer → Prompt Vector (256-512 dim)
                                       ↓
Image obs → ViT → patches  ────────────┤
Language → tokens          ────────────┤
State → MLP → state token  ────────────┤
                                       ↓
                          [Prompt; tokens] → Shared Transformer Trunk (1.5B)
                                       ↓
                          Universal Action Head (flow matching, π0-style)
                                       ↓
                          7D EE pose (continuous)
```

#### 训练
- **Stage 1**：1.2M trajectories pretrain，所有 prompt + trunk 联合学习
- **Stage 2**（新机体）：冻 trunk 和 head，只训新 prompt（200-1000 demos 即可）

#### Universal Action Head
- 用 flow matching（继承 π0）输出连续 action
- 避免 OpenVLA 的离散化精度损失

### 实验关键数据

#### 主实验：25 embodiment cross-train
| Method | Avg SR | New embod (10 demos) |
|---|---|---|
| OpenVLA (7B) | 73% | 30% |
| CrossFormer | 78% | 55% |
| HPT-XLarge (1B + 30M/embod) | 82% | 70% |
| **X-VLA (1.5B + 256/embod)** | **84%** | **72%** |

#### Parameter efficiency
- X-VLA 每个新 embodiment 参数：256 floats（1 KB）
- HPT 每个新 embodiment 参数：30M（120 MB）
- **参数效率提升 120000×**

#### Adaptation speed
- X-VLA：100 demos 达到 50% SR
- HPT：500 demos 达到 50% SR

### B7 时间线（X-VLA 是当前最新）

```
2023-10: OXE / RT-X       ── 数据 + 闭源
2024-06: OpenVLA           ── 开源 7B
2024-08: CrossFormer       ── 跨任务范式
2024-09: HPT               ── 模块化 trunk
2024-10: π0                ── flow matching action
2025-10: X-VLA (本笔记)     ── 软提示 + 共享 trunk
2026-??: 未来               ── meta-prompt / embodiment embedding
```

### 与曦源的关联（极重要——可能是首选 backbone）

1. **X-VLA 作为 SmolVLA 替代 backbone 的强候选**：
   - 参数 1.5B，比 OpenVLA 7B 小、比 SmolVLA 0.5B 大
   - 但适配新机体极快（100 demos vs 1000）
   - **曦源场景**：单 Franka 机体，可直接 fine-tune X-VLA 的 Franka prompt
   - 优势：trunk 已 1.2M trajectories pretrain，学到通用 motor intelligence
   - 劣势：相比 HPT 更新、社区更小，工程支持少

2. **soft prompt 对鲁棒性研究的启示**：
   - prompt 是 256 dim 小向量，可以做 **prompt-level adversarial training**
   - 想象：训练时对 prompt 加 worst-case 扰动 → 让 trunk 对 prompt 变化鲁棒
   - 这是个**新颖鲁棒训练方向**——专门针对 cross-embodiment VLA 的 worst-case
   - 与 RobustVLA 的 action 扰动正交

3. **未来「曦源 V2」的可能架构**：
   - V1：SmolVLA + RobustVLA（当前路线）
   - V2：X-VLA + RobustVLA + DRO 数据混合（更激进）
   - **可写进开题"future work"或长期 roadmap**

4. **跨机体训练能否提升单机体鲁棒性**（开放问题再访）：
   - X-VLA 实验显示：trunk 在 25 embod 训练，对每个机体的鲁棒性是否提升？
   - 论文没系统量化鲁棒性
   - **这是个 thesis-worth 的研究空白**：X-VLA + VLA-Risk 系统评测

### 待解问题

1. **prompt 表达力的上限**：256 dim 是否够？复杂异构机体（机械臂 vs 无人机）可能不够
2. **prompt 的 interpretability**：256 dim 向量很难解释，黑盒
3. **prompt 的 transferability**：新机体的 prompt 能否从相似机体的 prompt 初始化
4. **匿名作者身份**：2025-10 双盲，正式发表后会知道是哪个团队

### 复现挑战

- Pretrain 数据 1.2M 不开放（OpenReview 状态）
- 等待官方代码 / checkpoint 发布
- **替代**：用 HPT-Base 代替（架构思想类似）

%% end my-thoughts %%

## 🔗 关联笔记

- **B7 时间线**：[[2023-10_OpenX-Embodiment_Padalkar]], [[2024-06_OpenVLA_Kim]], [[2024-08_CrossFormer_Doshi]], [[2024-09_HPT_Wang]]
- **B7 形态嵌入**：[[2026_EmbedMorphology_Transformer]]
- **VLA 主线**：[[2025_SmolVLA]] (potential replacement)
- **鲁棒性协同**：[[2026_RobustVLA_Guo]] (soft prompt adversarial)
- **NLP soft prompt 起源**：Lester et al. 2021 "Power of Scale for Prompt Tuning"

## 📌 Action Items

- [ ] **关键决策**：X-VLA vs HPT vs SmolVLA 作为曦源 V2 backbone
- [ ] 等待官方代码发布；同时实验 HPT-Base 类似思路
- [ ] 想象：prompt-level adversarial training 实验设计
- [ ] 写一段「soft prompt as parameter-efficient cross-embodiment」纳入开题
- [ ] 评测：X-VLA 单机体鲁棒性 vs SmolVLA（待 X-VLA 开源）

%% Import Date: 2026-05-26 %%
