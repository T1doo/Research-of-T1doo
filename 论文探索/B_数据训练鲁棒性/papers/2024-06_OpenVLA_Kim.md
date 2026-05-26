---
title: "OpenVLA: An Open-Source Vision-Language-Action Model"
authors: "Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn"
year: "2024"
journal: "CoRL 2024 / arXiv"
doi: "10.48550/arXiv.2406.09246"
arxiv: "2406.09246"
venue: "CoRL 2024"
citekey: "kimOpenVLA2024"
itemType: "conference paper"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B7 开源 VLA 基石（与 A7 共享）"
tags: [literature, T1D, 主线B, B7, 主线A, A7, OpenVLA, 开源, VLA, 7B]
---

# OpenVLA — 开源 VLA 基石（B7 视角）

> [!info] 元信息
> - **作者**：Stanford / Berkeley / Google / Toyota Research / Physical Intelligence 联合（18 位作者）
> - **arXiv**：[2406.09246](https://arxiv.org/abs/2406.09246)
> - **代码 + checkpoint**：https://github.com/openvla/openvla
> - **B7 视角定位**：第一个**完全开源、SOTA、跨机体**的 7B VLA；OXE 数据 + Llama-2 7B + DINOv2 + SigLIP

## 📄 Abstract

VLA 时代亟需开源基础模型。OpenVLA：
- **7B 参数**（Llama-2 7B 基础）
- **训练数据**：OXE 970k demos 子集（22 datasets, 22 embodiments）
- **视觉**：DINOv2 + SigLIP 双 encoder
- **action 编码**：离散化为 256 bin × 7D = 1792 tokens
- 跨 29 个真机任务，平均 success rate 79.5%
- 比 RT-2-X (55B closed) 高 16.5pp，参数 8× 小
- **fine-tune-ready**：LoRA fine-tune 7-15 GB 显存即可

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（B7 视角下的 4 大贡献）

1. **「7B 开源 SOTA」改变了 VLA 研究生态**：
   - RT-2 / Octo 等闭源 / 不易用
   - OpenVLA 是第一个**学术团队都能下载即用 + fine-tune** 的 7B VLA
   - 直接催生 SmolVLA（0.5B 复刻版）、LeRobot 生态、RobustVLA 等下游研究
   - **没有 OpenVLA 就没有曦源现在的研究空间**

2. **「跨机体 + 跨任务」证明 transformer scaling 在 robotics 也成立**：
   - 970k demos 横跨 22 embodiment
   - 7B 参数把所有 task 压缩进一个模型
   - 跨机体迁移：在新机体上 fine-tune 几百 demo 就能 80%+
   - **这是 RT-X 路线的开源延续**

3. **DINOv2 + SigLIP 双 encoder 的视觉设计**：
   - DINOv2：自监督，强空间结构
   - SigLIP：language-aligned，强语义
   - 双流融合 > 单 encoder
   - **后续 FocusVLA 进一步利用了这个双 encoder 设计**

4. **离散 action token 化的工程权衡**：
   - 256 bin × 7D 连续动作 → 离散 token
   - 优点：与 LLM 训练 pipeline 无缝集成
   - 缺点：精度损失（特别是 fine manipulation）
   - **这是 π0 / X-VLA 用 flow matching 改进的动机**

### 方法论

#### 架构
```
Vision: DINOv2 + SigLIP → patch tokens
Language: SentencePiece tokenizer → text tokens
Action: 7D continuous → 256 bin × 7D → 1792 vocab id
↓
Llama-2 7B Transformer (autoregressive)
↓
Output: discrete action tokens → de-tokenize to continuous
```

#### 训练
- **Pretrain**：970k OXE demos，1 epoch，2k A100 GPU hours
- **Fine-tune**：LoRA rank 16，可在 1 A100 (40GB) 上跑

#### Action discretization
- 每维 action 离散到 256 bin
- 用 uniform binning over data range
- 推理：用 token logits 取 argmax → 中心值

### 实验关键数据

#### 真机 29 task 跨 7 embodiment
| Model | Avg Success |
|---|---|
| RT-1 (35M) | 28% |
| RT-2-X (55B) | 63% |
| Octo (93M) | 55% |
| **OpenVLA (7B)** | **79.5%** |

#### Fine-tune 样本效率（new task, BridgeData V2）
| # Demos | Scratch | LoRA OpenVLA |
|---|---|---|
| 10 | 5% | 35% |
| 100 | 20% | 75% |
| 1000 | 60% | 90% |

### B7 时间线中的位置

```
2023-10: OXE / RT-X       ── 数据 + 闭源模型
2024-06: OpenVLA (本笔记)   ── 开源 7B + OXE 子集
2024-08: CrossFormer       ── 跨任务范式
2024-09: HPT               ── 模块化 cross-embod
2024-10: π0                ── flow matching action
2025-10: X-VLA             ── 软提示
```

**OpenVLA 的角色**：从"概念验证（RT-X 闭源）"到"社区可复现"的关键节点。

### 与曦源的关联（核心）

1. **直接关联：曦源使用 SmolVLA = OpenVLA 0.5B 衍生**：
   - SmolVLA 是 LeRobot 团队的 OpenVLA 轻量版
   - 同样架构、同样训练 pipeline、同样 fine-tune 范式
   - **理解 OpenVLA 就是理解 SmolVLA**

2. **OXE pretrain 对单机体 fine-tune 的影响**：
   - OpenVLA 在 970k OXE 上预训
   - 然后在单机体 fine-tune
   - 单机体性能远超 from-scratch
   - **暗示 cross-embodiment pretrain 提供了"通用 motor prior"**
   - 我们应该利用这点：fine-tune 时保留 OXE pretrain 知识，加 LoRA 而不是 full fine-tune

3. **离散 action 对鲁棒性的影响**（未被充分研究）：
   - 256 bin 离散化引入了 quantization error
   - 这种误差 + 视觉扰动 + 语言扰动会叠加吗？
   - **可成为实证研究点**：OpenVLA 离散 vs π0 连续在鲁棒性上的差异

4. **作为 RobustVLA 的 base model**：
   - RobustVLA 在 OpenVLA 上做 worst-case fine-tune
   - 我们做 [[2026_RobustVLA_Guo]] 复现时直接用 OpenVLA checkpoint

5. **fine-tune 的工程便利性**：
   - LoRA 7-15GB 显存 → 我们的 RTX 4090 24GB 单卡可跑
   - 提供了清晰的 fine-tune 脚本和 docker
   - **这是曦源能落地的根本**

### 待解问题

1. **离散 action 的精度天花板**：高精度任务（cap removal, peg-in-hole）OpenVLA 表现差
2. **数据混合权重**：OXE 22 datasets 用 sqrt-size 加权，是否最优？(可用 ReMix DRO 测试)
3. **scaling beyond 7B**：13B / 70B 会继续涨吗？还是 saturate

### 复现挑战

- Pretrain 从头：2000+ A100 hours，学术团队不可行
- **替代**：直接下载 OpenVLA checkpoint + LoRA fine-tune
- LIBERO 任务：作者发布了 LIBERO-specific fine-tuned checkpoint，可直接用

%% end my-thoughts %%

## 🔗 关联笔记

- **B7 时间线**：[[2023-10_OpenX-Embodiment_Padalkar]], [[2024-08_CrossFormer_Doshi]], [[2024-09_HPT_Wang]], [[2025-10_X-VLA_Anonymous]]
- **VLA 主线**：[[2025_SmolVLA]] (0.5B 复刻), [[2024_pi0]] (continuous action 改进)
- **鲁棒性**：[[2026_RobustVLA_Guo]] (base model), [[2026_FocusVLA_Zhang]] (利用 DINOv2+SigLIP)
- **A7 共享**：（同一论文，A7 视角侧重视觉编码器）

## 📌 Action Items

- [ ] 跑通 OpenVLA LoRA fine-tune 在 LIBERO（已有官方 checkpoint）
- [ ] 对比：OpenVLA 7B vs SmolVLA 0.5B 在 LIBERO-Plus 鲁棒性
- [ ] 实验：离散 vs 连续 action 在 RobustVLA fine-tune 下的鲁棒性
- [ ] 写一段「OpenVLA 是曦源 base model 选择基础」纳入开题

%% Import Date: 2026-05-26 %%
