---
title: "Octo: An Open-Source Generalist Robot Policy"
authors: "Octo Model Team: Dibya Ghosh, Homer Walke, Karl Pertsch, et al. (UC Berkeley + Stanford + CMU)"
year: "2024"
venue: "RSS 2024"
arxiv: "2405.12213"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, diffusion, open-source, transformer]
---

# Octo

> [!info] 元信息
> UC Berkeley + Stanford + CMU 学术联合体 2024.05 发布的**开源通才机器人策略**。27M / 93M 参数 Transformer + diffusion action head，在 80 万条 Open X-Embodiment episodes 上预训练。是第一个真正开源（模型 + 数据 + 训练代码 + fine-tune 流程）的大规模 VLA。和 OpenVLA 是同期姊妹工作，代表了**非 LLM-based 的通用 baseline**。

## 📄 TL;DR（100-150 字）

Octo 是 Open X-Embodiment 数据集发布后第一个充分利用它的开源 generalist policy。架构是 small/base 两版（27M / 93M），用 Transformer encode 多模态观测（图像 + 语言 + 目标图像），输出层用 **diffusion head 生成动作 chunk**。关键设计是支持**多种观测和动作空间的灵活切换**（不同机体、不同摄像头数、不同 DoF），通过 readout token + 模态特定的输入 head 实现。下游 fine-tune 5 分钟即可适配新机体，是社区首个 "OpenAI Whisper for robotics" 式的开源 baseline。

## 🧠 我的思考

### 在 VLA 谱系中的位置

Octo 在 VLA 谱系里是**「开源大规模 Transformer baseline」的旗手**，与 OpenVLA 同期但走不同路线：

| 维度 | Octo | OpenVLA |
|---|---|---|
| Backbone | 从头训 Transformer | Llama-2 + ViT |
| 规模 | 27M / 93M | 7B |
| 动作头 | Diffusion | Discrete token |
| 路线 | 专用 IL + scaling | VLM 微调 |

**时间线上的关键节点**：
- RT-1（22.12）、RT-2（23.07）：闭源大模型；
- Open X-Embodiment（23.10）：数据集发布，社区开始拥有大规模数据；
- **Octo（24.05）：第一个充分利用 OXE 的开源通才模型**；
- OpenVLA（24.06）：紧随其后，走 VLM 路线，成为更主流的 baseline；
- π0（24.10）：Physical Intelligence 商业级 generalist policy，flow matching；
- RDT-1B（24.10）：1.2B diffusion VLA；
- FAST tokenizer（25.01）：解决 OpenVLA 离散化问题；
- NORA（25.04）：3B 开源轻量 VLA。

引用关系：Octo 引用 RT-1/2、Open X-Embodiment、Diffusion Policy、PaLM-E；被 OpenVLA、π0、ECoT、FAST、NORA 等几乎所有后续 VLA 引用作为「diffusion VLA 开源 baseline」对比。Octo 团队成员（Karl Pertsch, Dibya Ghosh, Homer Walke）后来很多加入 Physical Intelligence 做 π0，可以看作 π0 的方法学前身。

### 核心方法

**架构**：
1. **输入 tokenizer**：
   - **语言**：T5-base encoder → token；
   - **图像**：ResNet shallow stem 切 patch → token，按摄像头分组（primary / wrist）；
   - **目标图像**（goal-conditioned）：同上；
2. **Main Transformer**：12 层 base / 6 层 small，处理所有输入 token + 可学习的 readout token；
3. **Diffusion action head**：
   - 在 readout token 上接 conditional MLP；
   - 用 DDIM diffusion 生成 4 步 action chunk；
   - 每个动作 7-DoF（位置 + 方向 + gripper）；
4. **关键设计**：模态可变性——
   - 不同机体可能没有 wrist camera、没有 language goal 等；
   - 用「block-wise attention mask」让缺失模态不参与注意力；
   - Fine-tune 时可以加新输入 head（例如新的 force sensor），主干冻结。

**数据**：Open X-Embodiment 25 个子集，800k episodes。

**训练**：3e6 steps，masked predict next 4-step action chunk with diffusion loss。

**关键实验**：
1. 在 9 个机体 / 任务的 zero-shot 测试中接近 RT-1-X 性能；
2. 5 分钟 fine-tune 即可适配新机体；
3. Diffusion head 比 MSE / 离散动作显著好（特别在 multi-modal 任务）；
4. 模态 mask 设计让单一模型 handle 异构 IO。

### 鲁棒性视角下的解读

Octo 在鲁棒性维度上有几个独特贡献：

**天生鲁棒的部分**：
1. **Diffusion action head**：天然 handle multi-modal 动作分布（同一观测多种可行动作），比 OpenVLA 的 argmax 离散动作鲁棒，特别在 multi-stage 任务；
2. **Action chunking（4 步）**：减少 compounding error，借鉴 ACT 思想；
3. **大规模 OXE 数据**：跨 25 个机体的视觉 / 任务多样性提供良好的视觉鲁棒性基线；
4. **模态灵活性**：fine-tune 时可以加 / 减输入而不破坏主干，对真实部署友好；
5. **学术开源**：模型 / 数据 / 代码 / 训练流程全开源，可以被独立改进和审计——这是「研究鲁棒性」的前提。

**天生脆弱的部分**：
1. **无 LLM 先验**：T5 encoder 比 Llama-2 弱很多，语义泛化（认识新物体、理解抽象指令）远弱于 OpenVLA；
2. **27M / 93M 容量小**：scaling 不足，长尾任务性能差；
3. **从头训视觉**：ResNet shallow stem 缺少 DINOv2/SigLIP 级别的视觉鲁棒性，对纹理、光照 OOD 脆弱；
4. **Diffusion 推理慢**：4 步 DDIM 仍比 single-pass 慢，对实时性敏感；
5. **OXE 数据噪声**：跨 25 个数据集的 inconsistent label / 摄像头校准可能引入噪声，限制了上限。

后续鲁棒性改进：
- π0 用 flow matching（1 步即可）+ PaliGemma VLM backbone 解决了 Octo 几乎所有弱点；
- RDT-1B 把 Octo 思想推到 1.2B 规模；
- OpenVLA 走 VLM 路线证明 LLM 先验更重要。

### 与曦源（鲁棒性主线）的关联

Octo 是 **曦源 baseline 候选 #2**（仅次于 OpenVLA）：

**优势**：
1. **完全开源**：模型权重（27M / 93M）、训练代码、fine-tune 流程、数据预处理脚本全在 HuggingFace + GitHub；
2. **小模型 46GB 显存绰绰有余**：93M 推理只需 <1GB，fine-tune 整套都能放下；
3. **diffusion head 提供不同的攻击面**：和 OpenVLA 的离散动作鲁棒性研究形成对比，综述里可以两线对比；
4. **多机体 fine-tune 友好**：曦源如果有 Franka / xArm 等机体，5 分钟 fine-tune 可以快速验证；
5. **学术血统纯正**：UC Berkeley + Stanford 学术联合，论文 / 代码 / blog 都齐全。

**劣势**：
1. **性能确实弱于 OpenVLA**：作为 baseline 可能让曦源研究显得「打弱鸡」；
2. **没有 LLM 先验**：曦源如果要做「语言指令鲁棒性」研究，Octo 不是好对照；
3. **Diffusion 推理时间长**：对实时性研究不友好。

**建议**：作为曦源 baseline 阵列里的「diffusion 代表」，与 OpenVLA（discrete + LLM）形成对比。综述里要清楚说明 Octo vs OpenVLA 的方法学差异和鲁棒性 trade-off。

### 待解问题

1. Octo 的 diffusion head 在分布外推理时的「鲁棒性曲线」是什么形状？是否比 OpenVLA 的离散 argmax 更平滑？（曦源核心实验问题）
2. Octo 的 modality-flexibility 设计在 fine-tune 时加新模态（如 tactile）是否真的不损害主干？这对鲁棒性 fine-tune 友好度有多大？

## 🔗 关联笔记

- **数据来源**：Open X-Embodiment（23.10）——核心训练数据。
- **同期姊妹**：OpenVLA（24.06）——曦源主 baseline 候选 #1，对比路线。
- **方法学前身**：Diffusion Policy（23.03，Chi）——diffusion action head 的灵感。
- **直接后继**：π0（24.10，Physical Intelligence）——Octo 团队成员主导的商业级版本。
- **平行方案**：RDT-1B（24.10）——1.2B 规模的 diffusion VLA。
