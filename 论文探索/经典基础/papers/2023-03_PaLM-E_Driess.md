---
title: "PaLM-E: An Embodied Multimodal Language Model"
authors: "Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, et al. (Google + TU Berlin)"
year: "2023"
venue: "ICML 2023"
arxiv: "2303.03378"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, LLM, embodied]
---

# PaLM-E

> [!info] 元信息
> Google 2023.03 发布。把 562B 参数的 PaLM 大模型作为机器人的「大脑」，让多模态观测（图像、状态、3D 点云）直接以 embedding 形式塞进 LLM 的 token 序列。是第一次明确证明「LLM 预训练知识能 transfer 到具身任务」，并直接催生了 RT-2 的 VLA-as-language 范式。

## 📄 TL;DR（100-150 字）

PaLM-E 的核心思想：把任意模态（图像、机器人状态、神经表征）的连续 embedding 当成 LLM 词表外的「虚拟 token」，与文本 token 拼成统一序列输入 PaLM。最大版本 PaLM-E-562B = PaLM-540B + ViT-22B。在三大任务上证明价值：（1）高层任务规划（拆解「把蓝色方块给我」为子任务序列）；（2）视觉问答；（3）low-level 控制（Language Table、TAMP）。关键发现是 LLM 越大、机器人任务上的 catastrophic forgetting 越少，且单一模型能 transfer 跨任务知识。

## 🧠 我的思考

### 在 VLA 谱系中的位置

PaLM-E 是 VLA 谱系里的**「LLM 拐点」**——它正式把大语言模型推上具身智能的主舞台，宣告「机器人控制可以借用 LLM 预训练知识」。

**时间线上的关键节点**：
- SayCan（22.04）：用 LLM 做高层规划，但与低层 policy 解耦（LLM 不感知图像）；
- Code as Policies（22.09）：LLM 输出 Python 代码控制机器人；
- **PaLM-E（23.03）：第一次让 LLM 直接感知多模态观测并输出（高层 + 低层）动作**；
- RT-2（23.07）：同一团队，把 PaLM-E 的思路具体化成「VLA-as-language」——动作变成文本 token；
- OpenVLA（24.06）：开源版 RT-2，证明 7B Llama-2 足够；
- ECoT（24.07）：在 LLM-VLA 上加 CoT 推理，直接受 PaLM-E 启发。

引用关系：PaLM-E 引用 Flamingo、Gato、SayCan、ViT、PaLM；被 RT-2、Octo、OpenVLA、Voxposer、RoboFlamingo、ECoT 等所有 LLM-based VLA 引用。PaLM-E 没有开源，但它的「embedding-as-token」思想成了所有后续 VLM-based 机器人模型的标准做法。

### 核心方法

**核心架构**：标准 decoder-only LLM（PaLM-8B / 62B / 540B），关键改造在输入端：
1. **文本**：标准 token embedding；
2. **图像**：ViT 编码后通过 affine projection 映射到 LLM embedding 空间，每张图变成 N 个虚拟 token 插入序列；
3. **机器人状态**：MLP encode 到 embedding 空间；
4. **3D 表征**：实验了 ViT-22B、OSRT（Object Scene Representation Transformer）等。

**训练**：在多个数据集上 mix（Language Table、TAMP、SayCan、WebLI VQA、纯文本），所有任务统一为 text-in/text-out 格式。输出可以是高层规划（自然语言子任务）或低层动作（文本编码的 EE pose）。

**关键发现**：
1. **scale matters**：PaLM-E-562B 在保持机器人能力的同时几乎没有 catastrophic forgetting 通用 NLP/VQA 能力；小模型版本明显遗忘；
2. **正迁移**：跨任务联合训练（机器人 + VQA + 纯文本）对每个任务都有 buff，特别是 low-shot；
3. **新涌现能力**：multimodal CoT、视觉条件下的零样本规划。

### 鲁棒性视角下的解读

PaLM-E 是 VLA 鲁棒性讨论的**关键节点**，它确立了一个新的鲁棒性来源：

**天生鲁棒的部分**：
1. **LLM 预训练知识**：对语言指令的同义改写、新词汇、复杂从句天然鲁棒——这是 RT-1 时代 USE 词向量做不到的。后续 RT-2、OpenVLA 都证实了这一点；
2. **多模态联合训练的正迁移**：互联网图文数据带来的视觉概念泛化能力（认识新物体、理解空间关系）远超纯机器人数据训练的模型；
3. **CoT-style 中间推理**：对长指令、需要常识的任务鲁棒。ECoT 直接发扬了这一点。

**天生脆弱的部分**：
1. **巨大模型不可实时部署**：562B 的推理延迟对 closed-loop 控制不可接受，所以 PaLM-E 的低层控制实验都用小版本（12B）；
2. **embedding alignment 脆弱**：ViT 输出和 PaLM embedding 空间靠 affine projection 对齐，分布外图像可能产生「幻觉 embedding」误导 LLM；
3. **action representation 弱**：直接用文本编码 EE pose 精度差（被 RT-2 沿用、被 FAST 改进）；
4. **没有显式时序建模**：单帧输入为主，对动态任务鲁棒性差。

后续鲁棒性改进：RT-2 把 action 也变成 token 解决了表征统一性；OpenVLA 缩到 7B 解决了部署问题；FAST tokenizer 解决了 action 离散化；π0 用 diffusion 解决了连续控制平滑性。

### 与曦源（鲁棒性主线）的关联

PaLM-E **不能作为曦源的 baseline**：
1. **权重未开源**（Google 一贯做派）；
2. **562B 在 46GB 显存上完全不可能**（需要多卡 H100 集群）；
3. **早期工作已被 OpenVLA / RT-2 全面取代**。

但在综述里**必须引用**——它是「LLM 在具身智能里的价值」的奠基论文，讨论「鲁棒性来自 LLM 预训练知识」时是必引文献。可以把 PaLM-E 和 RT-2 一起当作 VLA-as-LLM 范式的「奠基双子」。

### 待解问题

1. PaLM-E 报告的「大模型缓解 catastrophic forgetting」对曦源是否有启示？（如果 7B Llama-2 base 的 OpenVLA 容量不够，是否要考虑 13B+ 模型？）
2. PaLM-E 的「3D 表征实验」（OSRT）成功了吗？这条路径后续被谁继承？（影响是否要在曦源里考虑 3D 输入鲁棒性）

## 🔗 关联笔记

- **直接前身**：Flamingo（22.04）——多模态 LLM 模板；SayCan（22.04）——LLM 规划机器人。
- **直接后继**：RT-2（23.07）——同团队的具体化产品。
- **范式继承**：OpenVLA（24.06）——开源版的 PaLM-E + RT-2 综合。
- **CoT 延伸**：ECoT（24.07）——把 PaLM-E 启发的 multimodal reasoning 显式化。
- **3D 分支**：3D Diffuser Actor、RVT——继承 PaLM-E 3D 输入实验思路。
