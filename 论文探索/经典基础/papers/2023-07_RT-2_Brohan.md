---
title: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
authors: "Anthony Brohan, Noah Brown, Justice Carbajal, et al. (Google DeepMind)"
year: "2023"
venue: "CoRL 2023"
arxiv: "2307.15818"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, VLM, 范式奠基]
---

# RT-2

> [!info] 元信息
> Google DeepMind 2023.07。基于 PaLI-X (55B) / PaLM-E (12B) 的 VLM 微调出的 VLA 模型。**正式确立「VLA-as-language」范式**——把机器人动作编码成文本 token，让 VLM 像生成文本一样生成动作。是 VLA 领域影响力最大的论文之一，被几乎所有后续 VLA 工作引用为范式奠基。

## 📄 TL;DR（100-150 字）

RT-2 的核心创新：把 RT-1 的 7+ 维连续动作每个维度离散化到 256 bin，每个 bin 映射成 PaLI-X 词表里最不常用的一个文本 token，然后整个动作变成 8 个文本 token 的字符串。这样动作生成和语言生成完全统一，可以**联合训练机器人数据 + 互联网 VQA 数据**。结果是 VLM 的语义知识、CoT 推理能力直接迁移到机器人——能识别训练集没出现过的物体、能理解抽象指令（"把咖啡放在不能吃的东西上"）、能简单 CoT 规划。性能比 RT-1 提升 2-3 倍。

## 🧠 我的思考

### 在 VLA 谱系中的位置

RT-2 是 VLA 谱系里**最重要的「范式定型」论文**。它把 PaLM-E 的「LLM 控制机器人」从「LLM 输出文本子任务」推进到「LLM 直接输出动作 token」，从此 VLA 有了清晰、可复现的定义：

> **VLA = VLM with action tokens in vocabulary**

**时间线上的关键节点**：
- RT-1（22.12）：真机大规模 VLA 范式起点；
- PaLM-E（23.03）：LLM 嵌入观测做高层规划；
- **RT-2（23.07）：VLA-as-language 范式正式确立**；
- Open X-Embodiment / RT-X（23.10）：RT-1/RT-2 跨 22 个机体验证；
- Octo（24.05）：开源 Transformer baseline（不是 VLM-based）；
- OpenVLA（24.06）：**RT-2 的开源版**，社区标杆；
- ECoT（24.07）：在 RT-2/OpenVLA 范式上加显式 CoT；
- FAST（25.01）：解决 RT-2 沿用的 naive 离散化问题；
- π0（24.10）、π0.5（25.04）：flow matching 替代离散动作，但仍是 VLA-as-language 的变体。

引用关系：RT-2 是 2023 后被引最多的 VLA 论文之一，OpenVLA、Octo、ECoT、FAST、NORA、Gemini Robotics 全部把 RT-2 列为范式奠基。可以说 2024 年之后的开源 VLA 几乎都是「试图开源化、轻量化、改进 RT-2」。

### 核心方法

**Backbone**：两个版本——
- **RT-2-PaLI-X**：基于 PaLI-X-55B（5B vision + 32B/55B language），最强版本；
- **RT-2-PaLM-E**：基于 PaLM-E-12B，更小但仍闭源。

**动作 tokenization**（核心创新）：
- 11 维动作（7 arm + 3 base + 1 mode），每维离散化到 256 bin；
- 把每个 bin 编号映射到 PaLI-X 词表里**最不常用的 256 个 token**（保证不破坏语言能力）；
- 一个动作 = 8 个文本 token 的字符串，例如 `"1 128 91 241 5 101 127"`。

**训练**：co-fine-tune on（1）机器人 episodes（RT-1 dataset 13万条 + Language Table）；（2）互联网 VQA/VLP 数据。损失统一是 next-token CE。

**推理**：标准 LLM 自回归生成 8 个 action token，反离散化得到动作，5Hz 闭环。

**关键实验结论**：
1. 比 RT-1 提升 2-3×（特别是未见物体、抽象语义任务）；
2. 涌现能力：CoT 提示能让 RT-2 简单推理（"我需要先把杯子拿起来才能擦"）；
3. co-training 关键：纯机器人微调会损失语言能力，必须 co-train；
4. scale matters：55B > 12B > 5B。

### 鲁棒性视角下的解读

RT-2 是 VLA 鲁棒性研究的**核心 baseline**，它的鲁棒性 profile 设定了 2024+ 所有研究的对照系：

**天生鲁棒的部分**（成为综述里「语义鲁棒性」的标杆）：
1. **语义泛化爆表**：能识别训练集没出现的物体（"草莓"、"红牛罐"），理解抽象语义（"健康食物"、"不能吃的东西"），这是 RT-1 完全做不到的，纯靠 VLM 预训练；
2. **指令多样性鲁棒**：同义改写、长指令、CoT 提示都能 handle；
3. **co-training 防止灾难遗忘**：保留了 VLM 的视觉常识。

**天生脆弱的部分**（成为后续鲁棒性研究的攻击点）：
1. **动作 token 离散化**：256 bin 对精细操作粒度不足，FAST tokenizer 直接针对这点改进；
2. **单帧 / 短历史输入**：长时序任务鲁棒性差；
3. **闭源 + 巨大模型**：55B / 12B 不可复现、不可改造，社区只能等开源版（OpenVLA）；
4. **5Hz 闭环延迟**：对动态扰动响应慢；
5. **视觉 OOD**：虽然 VLM 预训练强，但机器人摄像头视角和互联网图差异大，对极端光照、相机抖动仍脆弱。

后续鲁棒性研究路线全部围绕 RT-2 展开：
- **开源化** → OpenVLA、Octo；
- **轻量化** → NORA（3B）、TinyVLA；
- **CoT 推理增强** → ECoT；
- **动作表征改进** → FAST、π0 (diffusion)；
- **多机体泛化** → Open X-Embodiment、RT-X；
- **真机鲁棒性测试** → 大量后续 paper 在 OpenVLA 上做扰动测试。

### 与曦源（鲁棒性主线）的关联

RT-2 **不能作为曦源直接 baseline**（闭源、巨大），**但必须作为综述的范式锚点详细引用**。曦源的实际 baseline 应该用 **OpenVLA（RT-2 的开源版）**，所有「批评 RT-2 鲁棒性」的论点都可以平移到 OpenVLA。

46GB 显存对 RT-2 完全不够（55B 模型推理至少需要 4×80G H100），但 OpenVLA-7B 在 46GB 显存上可推理（bf16 推理约 14GB，留余量做 fine-tune 可能需要 LoRA + gradient checkpointing）。

讨论曦源研究背景时，RT-2 是必引「VLA 范式起点」+ 「为什么需要鲁棒性研究」的双重论据：因为 RT-2 已经定义了 VLA 是什么，所以鲁棒性研究就是在这个范式内做改进。

### 待解问题

1. RT-2 的 co-training 配方（机器人 : VQA 数据比例、loss 加权）是否被 OpenVLA 完整复现？（影响曦源在做 robust fine-tune 时是否要保留 VQA co-training）
2. RT-2 的「emergent CoT」是真涌现还是 prompting trick？ECoT 论文如何回答了这个问题？

## 🔗 关联笔记

- **直接前身**：RT-1（22.12）、PaLM-E（23.03）——同一团队的两条线汇合。
- **跨机体扩展**：Open X-Embodiment（23.10）、RT-2-X——把 RT-2 重训于 22 机体数据。
- **开源对应**：OpenVLA（24.06）——曦源主 baseline 候选 #1。
- **推理增强**：ECoT（24.07）——RT-2 范式 + 显式 CoT。
- **动作改进**：FAST tokenizer（25.01）、π0（24.10）——解决 RT-2 离散化问题。
- **轻量化**：NORA（25.04）——3B 开源 VLA，RT-2 范式的边缘部署版本。
