---
title: "A Generalist Agent"
authors: "Scott Reed, Konrad Żołna, Emilio Parisotto, et al. (DeepMind)"
year: "2022"
venue: "TMLR 2022"
arxiv: "2205.06175"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, 多模态, generalist]
---

# Gato

> [!info] 元信息
> DeepMind 2022 年发布的「通才智能体」，1.2B 参数 Transformer，在 604 个任务（Atari、图像描述、机械臂控制、对话）上联合训练。它是第一次明确把「机器人控制 + 视觉 + 语言」全部 tokenize 成同一个序列建模问题的尝试，是 VLA 范式的精神祖先。

## 📄 TL;DR（100-150 字）

Gato 提出用单个 Transformer 统一处理所有模态——把图像、文本、离散动作、连续动作全部 tokenize 成同一个序列，在 604 个不同任务上 next-token prediction 联合训练。模型 1.2B 参数，能玩 Atari、描述图像、操控 Sawyer 机械臂堆积木。性能远不到 SOTA 专家水平，但首次证明了「一个模型 + 一套接口 + 多模态序列建模」的范式可行，为 RT-1、RT-2、OpenVLA 等 VLA 工作奠定了方法论基础。

## 🧠 我的思考

### 在 VLA 谱系中的位置

Gato 是 VLA 谱系的**精神起源**，但严格说不算 VLA——它没有把「自然语言指令」作为一等公民的条件输入（语言只是任务描述里的一段 token），而真正的 VLA 必须以语言指令作为驱动。但它确立了三个后续 VLA 都遵守的原则：

1. **统一的 token 接口**：所有模态（图像 patch、文本 BPE、动作离散化）拼成一个序列；
2. **单一架构**：一个 decoder-only Transformer 处理所有任务；
3. **多任务联合训练**：用一个模型权重应付所有任务，靠 prompt 切换。

时间线上：Gato（2022.05）→ RT-1（2022.12）把「语言指令 + 真机数据 + 模仿学习」从 Gato 的多任务大杂烩里精炼出来，专注机器人；→ PaLM-E（2023.03）把 LLM 主体引入；→ RT-2（2307）正式确立 VLA-as-language 范式。Gato 被 RT-1、RT-2、OpenVLA 全部引用作为「unified sequence modeling」的先驱。它的局限——没有专门的语言指令通道、动作离散化粒度粗、机器人任务只占很小比例——恰好成了后续工作的改进方向。

### 核心方法

**架构**：1.2B 参数的 decoder-only Transformer（24 层、d=2048、16 头）。

**Tokenization** 是核心创新：
- **文本**：SentencePiece BPE（32k vocab）；
- **图像**：ViT 风格 16×16 patch + linear projection；
- **离散动作**（如 Atari、对话）：直接映射到 vocabulary；
- **连续动作 / 状态**（如机械臂关节）：μ-law 压缩后均匀离散化到 1024 个 bin，再嵌入到 vocabulary。

**训练**：自回归 next-token prediction，loss 只在 action 和 text token 上算（mask 掉观测）。604 个任务的数据按经验混合权重采样。

**部署**：给一个 prompt（任务描述 + demo），模型自回归生成 action token。

### 鲁棒性视角下的解读

Gato 的设计选择对鲁棒性有几个重要含义：

**天生脆弱的部分**：
1. **离散化误差**：1024 个 bin 的动作离散化对精细操作（如抓取小物体）误差很大，这在 RT-1 沿用后被广泛诟病，直到 Diffusion Policy / π0 用 flow matching 才解决；
2. **没有视觉鲁棒性设计**：完全依赖大规模数据隐式学习视觉不变性，对训练分布外的光照、背景变化几乎没有抗性；
3. **小模型容量分散**：1.2B 参数要兼顾 604 个任务，每个任务实际容量很小，机器人部分性能远不如专家。

**天生鲁棒的部分**：
1. **token 接口的可扩展性**：新任务、新机体、新模态都可以加 token 接进来，这个设计后来被 Open X-Embodiment 大规模利用；
2. **多任务 co-training**：跨任务知识共享对零样本泛化是潜在 buff，PaLM-E 和 RT-2 都验证了这一点。

后续鲁棒性研究的改进：RT-1 把范围缩小到机器人 + 真机数据规模化；OpenVLA 引入 LLM 预训练知识增强语义泛化；FAST tokenizer 用 DCT+BPE 解决离散化粒度问题。

### 与曦源（鲁棒性主线）的关联

Gato **不适合作为曦源的 baseline**。理由：
1. **没有开源权重**（DeepMind 一贯做派），无法复现；
2. **机器人部分用的是 Sawyer 模拟器 + 自家数据**，不是社区通用基准；
3. **模型已经过时**——任何 2024+ 的 VLA（OpenVLA、π0、Octo）都在机器人任务上完胜它。

但要在综述里作为「VLA 范式的起点」引用一次，说明「统一 token 序列建模」的思想源头。46GB 显存即使有权重也跑不起来 1.2B 的多任务联合推理（KV cache + 多模态 prompt 很容易爆显存）。

### 待解问题

1. Gato 多任务联合训练的「负迁移」问题被后续工作怎么验证 / 缓解？（影响曦源能否用 multi-task 提升鲁棒性的判断）
2. 1024-bin 离散化在精细 manipulation 上的实际误差量级是多少？（决定 baseline 选型时是否必须避开纯离散动作模型）

## 🔗 关联笔记

- **同代/先驱**：Decision Transformer（21.06）——单任务序列建模的先驱，Gato 把它推广到多任务。
- **直接后继**：RT-1（22.12）——把 Gato 的多任务范式收敛到真机 manipulation，是更工程化的 Gato。
- **范式继承**：RT-2（23.07）、OpenVLA（24.06）——「token 化所有模态 + 单 Transformer」的直接继承者。
- **鲁棒性改进**：FAST tokenizer（25.01）——解决了 Gato/RT-1 沿用至今的 naive 离散化问题。
