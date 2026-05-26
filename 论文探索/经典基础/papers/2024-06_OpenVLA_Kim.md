---
title: "OpenVLA: An Open-Source Vision-Language-Action Model"
authors: "Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al. (Stanford + UC Berkeley + Toyota Research)"
year: "2024"
venue: "CoRL 2024"
arxiv: "2406.09246"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, open-source, Llama-2, baseline]
---

# OpenVLA

> [!info] 元信息
> Stanford + UC Berkeley + Toyota Research 2024.06 发布。**7B 开源 VLA 标杆模型**，Llama-2-7B + DINOv2 + SigLIP，在 970k Open X-Embodiment episodes 上微调。性能在 BridgeData V2 上超过 RT-2-X（55B 闭源），成为社区事实上的 VLA baseline。**曦源主 baseline 首选**。

## 📄 TL;DR（100-150 字）

OpenVLA 是开源版的 RT-2。架构：Prismatic VLM（Llama-2-7B + DINOv2-L + SigLIP-SO400M 双视觉编码器） + RT-2 风格的动作 tokenization（256 bin / 维，映射到 Llama 词表最不常用的 256 token）。在 OXE 970k episodes 上 27 epochs 微调，64 A100 训了 14 天。性能：BridgeData V2 上 16.5% 绝对提升超过 RT-2-X-55B，跨 29 任务平均成功率最高。同时提供高效 LoRA fine-tune（24GB 显存）、量化（4bit 推理）、完整训练代码——成为 2024 年下半年所有 VLA 论文的事实 baseline。

## 🧠 我的思考

### 在 VLA 谱系中的位置

OpenVLA 在 VLA 谱系里是**「2024 年开源标杆 + 社区共识 baseline」**。可以毫不夸张地说，2024 年下半年开始的几乎所有 VLA 论文（ECoT、FAST、NORA、Robo-MUTUAL、各种 robustness 论文）都把 OpenVLA 作为对照基准。

**时间线上的关键节点**：
- RT-2（23.07）：闭源 VLA-as-language 范式；
- Open X-Embodiment（23.10）：970k episodes 数据；
- Octo（24.05）：第一个开源大规模 generalist，但走 diffusion 路线；
- **OpenVLA（24.06）：第一个开源 VLM-based VLA，超过 RT-2-X**；
- ECoT（24.07）：在 OpenVLA 上加 CoT 推理；
- π0（24.10）：商业版，flow matching + PaliGemma；
- RDT-1B（24.10）：1.2B diffusion VLA；
- FAST tokenizer（25.01）：解决 OpenVLA 离散化问题；
- NORA（25.04）：3B 轻量版；
- Gemini Robotics（25.03）：Google 商业 closed-source。

引用关系：OpenVLA 在 2024.06 之后被引用爆炸——成了 RT-2 不开源后的「社区指定 baseline」。所有讨论 VLA 鲁棒性、scaling、fine-tune efficiency、动作表征的论文都用 OpenVLA 做对比。这种「事实标准」地位是论文综述时必须重点叙述的。

### 核心方法

**Backbone：Prismatic VLM**
1. **视觉编码器**：双视觉融合——
   - **DINOv2-L**（自监督，强空间结构）；
   - **SigLIP-SO400M**（语义对齐强）；
   - 两个 patch token 在 channel 维 concat，过 MLP projector 到 Llama embedding 空间；
2. **语言主干**：Llama-2-7B；
3. **VLM 预训练**：先在 LLaVA-1.5 风格的多模态指令数据上预训练（重要！这是 OpenVLA 强于「直接 fine-tune Llama-2」的关键）。

**动作 tokenization**（继承 RT-2）：
- 7-DoF 动作（位置 xyz + 旋转 RPY + gripper）；
- 每维独立离散化到 256 bin（用训练数据的 1-99 percentile 做归一化）；
- 每个 bin 编号映射到 Llama-2 词表最不常用的 256 token；
- 一个动作 = 7 个文本 token，自回归生成。

**训练**：
- 数据：OXE 970k episodes（精选 25 个子集 + 重采样）；
- 64 A100 训 14 天，27 epochs；
- Loss：仅 action token 的 next-token CE。

**部署友好性**（最大贡献之一）：
- 完整 HuggingFace 集成，3 行代码 inference；
- LoRA fine-tune：24GB 显存即可在 RTX 4090 上 fine-tune；
- 4-bit 量化推理：<8GB 显存；
- 完整训练代码、数据预处理脚本、评测脚本全开源。

**关键实验**：
- BridgeData V2：超 RT-2-X-55B 16.5% 绝对值；
- 29 任务平均成功率最高；
- LoRA fine-tune 性能接近 full fine-tune。

### 鲁棒性视角下的解读

OpenVLA 是 VLA 鲁棒性研究的**绝对核心 baseline**——所有讨论必须围绕它展开。

**天生鲁棒的部分**：
1. **DINOv2 + SigLIP 双视觉编码器**：DINOv2 提供强空间结构鲁棒性（对纹理 / 光照变化），SigLIP 提供强语义对齐——这是相对 RT-1 / Octo 最大的视觉鲁棒性升级；
2. **Llama-2 预训练知识**：对自然语言指令、同义改写、长指令、CoT 提示鲁棒；
3. **OXE 大规模多机体数据**：970k episodes 跨多个机体提供视角 / 物体 / 场景多样性；
4. **VLM 预训练 + 机器人微调的两阶段**：保留 VLM 通用能力同时获得 manipulation 技能；
5. **完整开源**：可以被独立审计、改进、对抗测试——这是研究鲁棒性的物质基础。

**天生脆弱的部分**（曦源应该重点研究的目标）：
1. **256-bin 离散化**：精细操作粒度不足，FAST tokenizer 已经证明这是性能瓶颈；
2. **单帧输入 + 短上下文**：长时序任务、需要状态记忆的任务脆弱；
3. **Argmax 动作生成**：纯 deterministic，多模态动作分布无法表达；
4. **闭环 5-10Hz 推理**：相对 ACT / Diffusion Policy 慢，动态扰动响应不足；
5. **OXE 标签噪声 + 摄像头不一致**：跨子集的视角 / 校准差异可能引入隐藏脆弱性；
6. **没有 recovery / online adaptation 机制**：纯模仿学习，OOD 即失败；
7. **对相机抖动 / 视角偏移敏感**：训练数据视角分布外即性能大降——这是大量 2024-2025 鲁棒性论文的实验切入点。

后续鲁棒性改进路线全部针对 OpenVLA：
- **CoT 增强** → ECoT（24.07）：直接在 OpenVLA 上加 CoT 推理；
- **动作表征改进** → FAST（25.01）：DCT+BPE 替换 256-bin 离散化；
- **轻量化** → NORA（25.04）：3B 替代 7B；
- **多模态扩展** → 各种加 force sensor、tactile、point cloud 的 fine-tune 工作；
- **测试时鲁棒性** → 大量论文报告 OpenVLA 在视觉扰动 / 语言改写下的失败模式。

### 与曦源（鲁棒性主线）的关联

OpenVLA 是 **曦源主 baseline 首选，毫无悬念**：

**优势**：
1. **社区共识 baseline**：用 OpenVLA 做实验，结果可以直接和 2024+ 几乎所有 VLA 论文对比；
2. **46GB 显存完全够用**：
   - 推理：bf16 约 14GB，4-bit 量化 <8GB；
   - LoRA fine-tune：约 24GB（rank 32，batch 16），有充足余量；
   - 全参数 fine-tune：需要 gradient checkpointing + Zero-2，可能勉强；
3. **完整工具链**：HuggingFace 模型卡、训练代码、LoRA 配方、量化、评测脚本一应俱全；
4. **鲁棒性研究素材丰富**：2024-2025 已有大量论文报告 OpenVLA 失败模式，可以对照设计实验；
5. **跨机体 fine-tune 成熟**：OXE 内任意机体子集可以快速实验。

**部署细节建议**：
- 优先用 LoRA fine-tune（rank 32-64），全参数 fine-tune 显存压力大；
- 鲁棒性测试用 BridgeData V2 / LIBERO benchmark，社区有标准结果可对比；
- 视觉鲁棒性测试可以在推理时注入扰动（高斯噪声、颜色抖动、视角偏移）；
- 语言鲁棒性测试可以用 GPT-4 改写指令。

### 待解问题

1. OpenVLA 在 LIBERO 等小规模 benchmark 上的「真实鲁棒性」是否被高估？（许多论文报告 LIBERO 高分但真机部署脆弱）
2. LoRA fine-tune 后的 OpenVLA 是否会损失原 OXE 知识？rank 多大才能平衡 specialization 和 generalization？（这是曦源做 robust fine-tune 必须回答的问题）
3. OpenVLA 的 256-bin 离散化误差在曦源关注的任务（精细操作？）上贡献多少失败？（如果是主因，应该优先看 FAST）

## 🔗 关联笔记

- **直接前身**：RT-2（23.07）——闭源对照；PaLM-E（23.03）——LLM-VLA 范式起源。
- **VLM backbone 前身**：Prismatic VLM（24.02，Karamcheti）——OpenVLA 的多模态主干。
- **数据来源**：Open X-Embodiment（23.10）——970k episodes 训练数据。
- **同期姊妹**：Octo（24.05）——diffusion 路线对比。
- **CoT 改进**：ECoT（24.07）——曦源潜在研究方向参考。
- **动作改进**：FAST tokenizer（25.01）——直接改进 OpenVLA 的 256-bin 离散化。
- **轻量化**：NORA（25.04）——3B 版本，曦源备选 baseline。
- **下一代**：π0（24.10）、π0.5（25.04）——flow matching + PaliGemma 的进化方向。
