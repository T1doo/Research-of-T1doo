---
title: "HPT: Heterogeneous Pre-Trained Transformers for Scalable Cross-Embodiment Learning"
authors: "Lirui Wang, Xinlei Chen, Jialiang Zhao, Kaiming He (MIT / Meta FAIR)"
year: "2024"
journal: "NeurIPS 2024 / arXiv"
doi: "10.48550/arXiv.2409.20537"
arxiv: "2409.20537"
venue: "NeurIPS 2024"
citekey: "wangHPT2024"
itemType: "conference paper"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B7 异构 tokenizer + 共享 trunk"
tags: [literature, T1D, 主线B, B7, cross-embodiment, HPT, 异构, tokenizer, 共享trunk]
---

# HPT — 异构 tokenizer + 共享 trunk 精读笔记

> [!info] 元信息
> - **作者**：Lirui Wang, Xinlei Chen, Jialiang Zhao, Kaiming He（MIT + Meta FAIR）
> - **arXiv**：[2409.20537](https://arxiv.org/abs/2409.20537)
> - **项目页**：https://liruiw.github.io/hpt/
> - **B7 价值**：明确解耦「embodiment-specific tokenizer」与「shared transformer trunk」，提供 scalable cross-embodiment 的清晰范式

## 📄 Abstract

跨机体训练的核心矛盾：每个 embodiment 的传感/运动空间不同 vs 想用一个模型。HPT 提出**模块化范式**：
- **Per-embodiment tokenizer**（小，特异）：把每种机体的 obs → 统一 token 空间
- **Shared transformer trunk**（大，通用）：跨所有 embodiment 共享
- **Per-task action head**（小，特异）：trunk → action

预训练数据：**50+ datasets, 1M+ trajectories, 30+ embodiments**。
模型规模：HPT-XLarge 1B 参数。
新机体迁移：只需训 tokenizer + head（trunk 冻结），样本效率提升 10-100×。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（B7 最 elegant 的范式）

1. **「模块化解耦」是 HPT 的核心 insight**：
   - 不要试图让一个 transformer 端到端处理所有异构（CrossFormer 做法）
   - 把特异性放在**输入/输出 adapter**，通用性放在**中间 trunk**
   - 类似 NLP 里的 multilingual BERT：不同语言用不同 tokenizer，但 transformer 共享
   - **简洁、可扩展、可维护**——增加新 embodiment 只需训 tokenizer + head

2. **trunk 的规模 vs tokenizer 的轻量**：
   - HPT-XLarge：trunk 1B 参数，每个 tokenizer 10-50M
   - 总参数 ≈ 1B + 30 × 30M = 1.9B
   - 实际部署：用某个 embodiment 时，只需 tokenizer + trunk + head ≈ 1.05B
   - **比 CrossFormer 单一 transformer 部署更灵活**

3. **「shared trunk 学到什么」的实验性回答**：
   - 论文做 probing：trunk 中层 representation 编码「场景对象」「motor primitive」等通用概念
   - embodiment-specific 信息主要在 tokenizer 的最后一层
   - **暗示**：trunk 是「通用动作智能」，tokenizer 是「身体适配器」

### 方法论

#### 架构
```
Embodiment A (Franka):
  RGB → ViT → tokens_A
  State → MLP_A → state_tokens_A
  Action history → linear_A → action_tokens_A
  ↓
  Concat → Tokenizer_A → unified tokens
  ↓
  ┌─────────────────────────────────┐
  │   Shared Transformer Trunk      │  (frozen during fine-tune)
  │   1B parameters, 32 layers      │
  └─────────────────────────────────┘
  ↓
  Action Head_A → 7D EE action

Embodiment B (xArm), C (UR5), …:
  独立 tokenizer + head，共用 trunk
```

#### 训练阶段
1. **Pre-training**：50 datasets 联合训练，trunk + 所有 tokenizers 联合更新
2. **Adaptation**：新 embodiment 来了，**只训 tokenizer + head**，trunk 冻结

#### Per-embodiment tokenizer 设计
- **Image tokenizer**：ViT (small) + cross-attention
- **State tokenizer**：MLP with embodiment-specific input dim
- **Action tokenizer**：linear projection to fixed-dim token

### 实验关键数据

#### Cross-embodiment Pre-training Scale
| Model | Trunk Size | # Embodiments | # Trajectories |
|---|---|---|---|
| HPT-Small | 50M | 30 | 1M |
| HPT-Base | 200M | 30 | 1M |
| HPT-Large | 500M | 30 | 1M |
| HPT-XLarge | 1B | 30 | 1M |

**Scaling law**：模型大小翻倍 → 真机 success rate 提升 5-10pp（log scale）。

#### Few-shot adaptation 到新机体
| # Demos for new embod | From scratch | HPT-frozen-trunk |
|---|---|---|
| 10 | 15% | 55% |
| 100 | 40% | 75% |
| 1000 | 70% | 85% |

**样本效率 10-100×**——这是 HPT 最有说服力的卖点。

### B7 时间线中的位置

```
2023-10: OXE                      ── 数据
2024-06: OpenVLA                  ── 单一大模型
2024-08: CrossFormer              ── 单一 transformer + 变长 token
2024-09: HPT (本笔记)              ── 模块化：tokenizer + 共享 trunk
2025-10: X-VLA                    ── 软提示 transformer（进一步抽象）
```

### 与曦源的关联（极重要——可能影响 backbone 选择）

1. **HPT vs SmolVLA 作为 backbone 的对比**：
   - **SmolVLA**：0.5B 单一模型，针对单 embodiment fine-tune
   - **HPT**：模块化，可拓展多 embodiment
   - **问题**：曦源是否考虑切换到 HPT-base 作为 backbone？
   - **优势**：
     - HPT 1B 比 SmolVLA 0.5B 略大但仍可控
     - 模块化支持未来多 LeRobot 机体扩展
     - shared trunk 已 pretrain on 1M trajectories
   - **劣势**：
     - 中文社区不熟，文档不如 SmolVLA 完整
     - LeRobot 集成需自己做
   - **建议**：作为 backup backbone，等 SmolVLA 主线遇瓶颈再切

2. **HPT 思想对鲁棒性研究的启示**：
   - shared trunk 训练在 30 embodiment 数据上 → 学到「形态无关」的通用 motor primitive
   - **假设**：这种 trunk 对**单 embodiment 的物理扰动**更鲁棒（因为它见过多种 dynamics）
   - **实验**：HPT-frozen-trunk + small adapter vs SmolVLA 在 VLA-Risk 上的鲁棒性对比

3. **可借鉴的工程模式**：
   - 即使不换 backbone，可以借鉴「tokenizer 解耦」思想
   - 把 SmolVLA 的视觉/状态编码器抽象成可替换模块
   - **未来扩展**：曦源做完单 Franka，扩展到 Koch / SO-100 时，模块化设计大幅省力

4. **与 RobustVLA 的协同**：
   - HPT shared trunk = 隐式数据多样化
   - RobustVLA worst-case = 显式扰动多样化
   - **组合**：HPT pretrain + RobustVLA fine-tune = 双重鲁棒

### 待解问题

1. **trunk 冻结的极限**：30 embodiment 后能继续加吗？trunk 容量饱和点在哪
2. **catastrophic forgetting**：fine-tune tokenizer 时，新 embodiment 学好后旧的会忘吗
3. **tokenizer 设计的最佳实践**：每种 embodiment 用相同 tokenizer 架构？还是按 DoF / 传感配置定制

### 复现挑战

- 1M trajectories 的预训练成本不菲（256+ GPU days）
- 但 HPT 开放预训练 checkpoint，可直接做 adaptation 实验
- 关键复现资源：HPT-Base checkpoint + tokenizer recipe

%% end my-thoughts %%

## 🔗 关联笔记

- **B7 时间线**：[[2023-10_OpenX-Embodiment_Padalkar]], [[2024-06_OpenVLA_Kim]], [[2024-08_CrossFormer_Doshi]], [[2025-10_X-VLA_Anonymous]]
- **B7 形态**：[[2026_EmbedMorphology_Transformer]], [[2025_HierarchicalEquivariant]]
- **架构对比**：[[2024-08_CrossFormer_Doshi]] (vs 端到端)
- **VLA 主线**：[[2025_SmolVLA]] (potential backbone switch?)
- **鲁棒性协同**：[[2026_RobustVLA_Guo]]

## 📌 Action Items

- [ ] **关键决策**：HPT vs SmolVLA 作为曦源 backbone（与导师讨论）
- [ ] 下载 HPT-Base checkpoint，跑 adaptation tutorial
- [ ] 实验：HPT-frozen-trunk vs SmolVLA 在 LIBERO-Plus / VLA-Risk 鲁棒性对比
- [ ] 制作 4 大 cross-embodiment 方法（OpenVLA / CrossFormer / HPT / X-VLA）对比表

%% Import Date: 2026-05-26 %%
