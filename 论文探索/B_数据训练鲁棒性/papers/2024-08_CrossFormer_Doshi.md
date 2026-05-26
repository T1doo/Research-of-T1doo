---
title: "CrossFormer: Scaling Cross-Embodied Learning for Manipulation, Navigation, Locomotion, and Aviation"
authors: "Ria Doshi, Homer Walke, Oier Mees, Sudeep Dasari, Sergey Levine (UC Berkeley / CMU)"
year: "2024"
journal: "CoRL 2024 / arXiv"
doi: "10.48550/arXiv.2408.11812"
arxiv: "2408.11812"
venue: "CoRL 2024"
citekey: "doshiCrossFormer2024"
itemType: "conference paper"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B7 跨机体 tokenization"
tags: [literature, T1D, 主线B, B7, cross-embodiment, CrossFormer, tokenization, 多模态]
---

# CrossFormer — 跨机体 tokenization 精读笔记

> [!info] 元信息
> - **arXiv**：[2408.11812](https://arxiv.org/abs/2408.11812)
> - **代码**：https://github.com/rail-berkeley/crossformer
> - **B7 价值定位**：OXE 之后的下一代，把 cross-embodiment 推到**900k 轨迹 × 20+ embodiment × 4 大类任务（manipulation / navigation / locomotion / aviation）**

## 📄 Abstract

OXE 证明跨机体训练可行，但仅限于 manipulation。CrossFormer 进一步把范围扩展到 **4 大类机器人任务**：
- **Manipulation**（机械臂）
- **Navigation**（移动机器人）
- **Locomotion**（四足 / 双足）
- **Aviation**（无人机）

用单一 transformer + **变长 action token** 统一处理。900k+ 轨迹训练，20+ embodiment。结果：在每个 embodiment 上不劣于专门训的 baseline，且**zero-shot 迁移到新机体**性能优于从头训练。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「跨任务类别 + 跨机体」双重扩展是关键创新**：
   - OXE 只跨 manipulation 内的 embodiment（22 个机械臂）
   - CrossFormer 跨**任务范式**：manipulation / navigation / locomotion / aviation
   - 这是个更激进的假设：不同任务范式的 motor primitive 可共享 representation
   - 实证：work，但需要更灵活的 tokenization

2. **变长 action tokenization 是技术关键**：
   - OXE / OpenVLA 用固定 7D action token
   - CrossFormer 用**变长 token**：4-leg robot 12 关节 → 12 tokens；机械臂 7DoF → 7 tokens；无人机 4 rotor → 4 tokens
   - 每个 token 是一个 motor 的归一化命令
   - 优点：自然处理 DoF 异构
   - 缺点：动作 sequence 变长，推理速度依 token 数变化

3. **「unified transformer = unified intelligence」的工程证据**：
   - 单模型处理 4 种 robot type，性能不劣于专家模型
   - 暗示：transformer 已能学到**任务无关**的 motor primitive 表征
   - 这与 [[2024_HPT]] 的"shared trunk + specific tokenizer"思想呼应

### 方法论

#### 架构
```
Input:
  - Image obs: ViT patch encoder → image tokens
  - Proprioception: MLP → state tokens
  - Language: T5 encoder → text tokens
  - Action history: linear → action tokens

Transformer (decoder-only):
  - Causal attention over all tokens
  - Each timestep prediction: variable-length action tokens

Output:
  - Variable length, depends on embodiment
  - Each token = one motor command
```

#### 训练
- 900k+ trajectories, 20+ embodiments
- 标准 BC loss
- 多任务混合权重：按 dataset size sqrt 加权（继承 OXE）
- 16k context length（处理长 horizon）

### 实验

#### 跨 4 类任务（每类 vs 专家 baseline）
| Task Category | Specialist | CrossFormer | Δ |
|---|---|---|---|
| Manipulation (Franka pick) | 78% | 81% | +3 |
| Navigation (ANYmal) | 85% | 87% | +2 |
| Locomotion (A1 walk) | 92% | 89% | -3 |
| Aviation (drone hover) | 70% | 75% | +5 |

#### Zero-shot 迁移到新机体（K-Quad，未见过）
- 从头训练：50%
- CrossFormer zero-shot：65%
- + 100 demo fine-tune：78%

### B7 时间线中的位置

```
2023-10: OXE / RT-X      ── 跨机器臂
2024-06: OpenVLA          ── 7B 开源，OXE 970k
2024-08: CrossFormer (本笔记) ── 跨任务范式，变长 token
2024-09: HPT              ── 异构 tokenizer + 共享 trunk
2025-10: X-VLA            ── 软提示统一
```

### 与曦源的关联

1. **「变长 action token」对 LeRobot 多机体支持的启示**：
   - LeRobot 上有 Franka / Koch / Moss / SO-100 等多种机体
   - 当前每种机体用 separate policy
   - 可借鉴 CrossFormer 的变长 token 设计 → 一个 SmolVLA 处理所有 LeRobot 机体
   - **可能成为曦源的工程贡献点**

2. **跨任务范式对鲁棒性的意义**（推测）：
   - CrossFormer 训练时见过 manipulation + navigation + locomotion
   - 模型学到的"motor primitive"可能更通用、对扰动更鲁棒
   - **假设**：跨任务范式训练是隐式的 worst-case curriculum
   - 可与 RobustVLA 形成对照实验

3. **与 HPT / X-VLA 的对比表（待填）**：
   | 维度 | CrossFormer | HPT | X-VLA |
   |---|---|---|---|
   | 处理异构 | 变长 token | per-embod tokenizer | 软提示 |
   | 共享层 | 全部 transformer | 中间 trunk | 全部 |
   | 任务跨度 | 4 大类 | manipulation only | manipulation + nav |
   | 数据规模 | 900k | 50 dataset | 1.2M |

4. **不适合作为 SmolVLA backbone**：
   - CrossFormer 用大模型 + 大数据 + 大 context
   - 不像 SmolVLA / OpenVLA 是开源 fine-tuning-ready 的小模型
   - 但**其 tokenization 思想可移植**

### 待解问题

1. **变长 token 的推理稳定性**：DoF 不同导致 sequence 长度不同，beam search / sampling 难以处理
2. **跨任务范式的负迁移**：不同 task category 是否存在干扰？论文显示 locomotion -3pp 暗示有
3. **safety in flight / locomotion**：跨任务包含安全敏感的飞行任务，鲁棒性不达标会摔机

### 复现挑战

- 900k 数据 + 大 transformer：需要 64+ GPU
- 多 embodiment 仿真器集成：MuJoCo + IsaacGym + 飞行 sim
- **替代路径**：使用作者发布的 CrossFormer-base checkpoint

%% end my-thoughts %%

## 🔗 关联笔记

- **B7 时间线**：[[2023-10_OpenX-Embodiment_Padalkar]], [[2024-06_OpenVLA_Kim]], [[2024-09_HPT_Wang]], [[2025-10_X-VLA_Anonymous]]
- **B7 形态**：[[2026_EmbedMorphology_Transformer]], [[2025_HierarchicalEquivariant]]
- **架构对比**：[[2024-09_HPT_Wang]] (per-embod tokenizer)
- **VLA 主线**：[[2024-06_OpenVLA_Kim]], [[2025_SmolVLA]]

## 📌 Action Items

- [ ] 跑通 CrossFormer 官方 zero-shot inference demo
- [ ] 制作 CrossFormer vs HPT vs X-VLA 三方对比表
- [ ] 思考：变长 action token 能否移植到 SmolVLA
- [ ] 思考：4 类任务跨度训练对鲁棒性的影响

%% Import Date: 2026-05-26 %%
