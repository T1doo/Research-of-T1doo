---
title: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
authors: "Open X-Embodiment Collaboration (Google DeepMind + 21 institutions)"
year: "2023"
venue: "ICRA 2024 (Best Paper)"
arxiv: "2310.08864"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, dataset, cross-embodiment]
---

# Open X-Embodiment (OXE)

> [!info] 元信息
> 21 个机构（Google DeepMind 牵头 + Stanford、UC Berkeley、CMU、Toyota Research、MIT 等）2023.10 联合发布的**史上最大跨机体机器人数据集**。1.4M 真机 episodes，22 个机体形态，160k+ 任务。同时发布 RT-1-X / RT-2-X 模型证明跨机体迁移可行。**事实上的开源 VLA 数据基底**——Octo、OpenVLA、π0、RDT、NORA 全部基于它训练。

## 📄 TL;DR（100-150 字）

OXE 是机器人学界的「ImageNet 时刻」。21 个机构合并各自的真机数据集到统一 RLDS 格式，得到 1.4M episodes、22 种机体（从 Franka、xArm 到 Google Robot、Mobile Manipulators）。同时训练 RT-1-X 和 RT-2-X 证明：用 X-formation 数据 co-train，比单机体训练在原机体上**提升 50%**，跨机体迁移涌现明显。是 2024+ 所有开源 generalist VLA（Octo、OpenVLA、π0、RDT、NORA）的训练数据来源。**没有 OXE 就没有 2024 年开源 VLA 爆发**。

## 🧠 我的思考

### 在 VLA 谱系中的位置

OXE 在 VLA 谱系里是**「数据基础设施」级别的奠基工作**——它不是一个模型，而是支撑了整个 2024+ 开源 VLA 生态的数据底座。意义类似 NLP 的 The Pile、视觉的 LAION。

**时间线上的关键节点**：
- Gato（22.05）、RT-1（22.12）：闭源数据时代，每个团队自己采集；
- BridgeData V1/V2（22.10, 23.08）：UC Berkeley 开源中等规模数据集尝试；
- **OXE（23.10）：第一次大规模跨机构数据合并 + 统一格式**；
- Octo（24.05）：第一个充分利用 OXE 的开源模型；
- OpenVLA（24.06）：970k OXE 子集训练；
- DROID（24.03）：Stanford 后续大规模数据集（76k episodes），后并入 OXE 体系；
- π0（24.10）：Physical Intelligence 在 OXE + 自家数据上训练；
- AgiBot World（24.12）、RoboMIND（24.12）：中国团队的类似大规模数据集。

OXE 的影响力体现在：之后任何 generalist VLA 论文的「数据」部分都会提到 OXE 子集组成。RT-X 模型本身也是范式证明——「用统一架构在 22 机体上训练 + 单机体测试 = 性能反而上升」。

### 核心方法

**数据集构成**：
- **22 个机体**：包括 Franka Panda、Google Robot、xArm、Sawyer、ALOHA、PR2、Stretch、UR5、Spot 等；
- **34 个研究实验室**贡献子集；
- **1.4M episodes** 跨 311 个场景；
- **160k+ 任务**（自然语言描述）；
- 顶部贡献者：Google RT-1 (130k)、BridgeData V2 (60k)、Stanford Hydra、DLR、Jaco Play 等。

**统一格式 RLDS**（关键工程贡献）：
- 基于 TFDS 的 RLDS（Reinforcement Learning Dataset Standard）；
- 统一字段：observation/image, observation/state, action, language_instruction, reward 等；
- 不同机体的 action space、camera angles 等保留原始信息，模型层处理；
- 加载工具完善，HuggingFace、TFDS 均有 mirror。

**RT-X 模型实验**：
- RT-1-X：在 OXE 上 train 同样 RT-1 架构；
- RT-2-X：在 OXE 上 train 同样 RT-2 架构；
- 关键结论：
  1. **OXE-trained > single-robot-trained** 50% 平均；
  2. **跨机体涌现**：RT-2-X 学到的技能能 zero-shot transfer 到训练时未见的机体（同形态）；
  3. **数据 mixing weight 重要**：naive 均匀采样不如按 episode 数 sqrt 加权；
  4. **VLM 基础越强，跨机体迁移越好**：RT-2-X (55B) > RT-1-X (35M)。

### 鲁棒性视角下的解读

OXE 对 VLA 鲁棒性的影响是**根本性的**——它定义了 2024+ 开源 VLA 的「鲁棒性下限」和「鲁棒性可能性」：

**OXE 带来的鲁棒性优势**：
1. **视觉多样性**：22 机体 = 22 种摄像头视角、光照、背景，模型被迫学习视角不变性；
2. **物体多样性**：跨实验室的物体差异（不同 mug、不同 block）天然提供 OOD 训练；
3. **语言多样性**：160k+ 任务的自然语言描述（不同人写、不同语气、不同详略度）提供语言鲁棒性；
4. **机体形态多样性**：从 6-DoF 到 14-DoF，从 single-arm 到 bimanual + mobile，模型学到的策略对动作空间变化有一定鲁棒性；
5. **场景多样性**：实验室、家庭、厨房、工业场景混合，对场景 OOD 鲁棒。

**OXE 引入的鲁棒性挑战**（这些是综述里的重点）：
1. **标签噪声**：不同实验室对「task success」定义不同，自然语言描述质量参差不齐；
2. **摄像头校准不一致**：内参 / 外参不统一，跨子集的几何信息不可靠；
3. **动作空间不一致**：有的是 EE delta、有的是 joint position、有的是 absolute pose，归一化策略影响巨大；
4. **数据质量不均**：Google 的 RT-1 数据质量远高于一些小实验室子集，naive co-training 可能被低质量数据拖累；
5. **demonstration 噪声**：人类遥操作的随机性、犹豫、错误恢复都进入数据，可能让模型学到次优行为；
6. **任务长尾分布**：130k 任务里大部分是「pick X put Y」的变体，复杂长时序任务比例小。

后续鲁棒性研究在 OXE 上的应对：
- **数据 curation**：OpenVLA 精选 25 个子集 + 重新归一化；
- **动作空间统一**：Octo 用 7-DoF EE delta 作为统一动作空间；
- **质量过滤**：π0 用质量评分过滤；
- **采样策略**：sqrt 加权 + 任务多样性平衡。

### 与曦源（鲁棒性主线）的关联

OXE 对曦源的关联是**间接但根本的**——它不是 baseline 模型，而是**baseline 模型的数据来源**和**鲁棒性 fine-tune 的数据池**：

**曦源用法**：
1. **理解 OpenVLA / Octo 的训练数据组成**：曦源在 fine-tune 时必须知道哪些子集是「in-distribution」、哪些是 OOD；
2. **选择 fine-tune 子集**：根据曦源研究的鲁棒性维度选 OXE 子集，例如要测「视角鲁棒性」就选多视角 mix 的子集（Berkeley、Stanford Hydra）；
3. **设计 OOD 测试**：从 OXE 选「训练时未用」的子集作为 OOD 测试，符合社区惯例；
4. **数据预处理参考**：OXE 的 RLDS 格式 + 归一化策略是事实标准，曦源自己采集数据也应该遵循。

46GB 显存对 OXE 全集训练完全不够（需要数十张 A100 跑数周），但对 OXE 子集 fine-tune（如 BridgeData V2 + RT-1 子集）是充足的。

**综述里的引用方式**：必须在「数据」章节单独叙述 OXE 的奠基作用，论证「2024+ 开源 VLA 的鲁棒性研究都建立在 OXE 之上」。这是曦源研究背景的关键支撑。

### 待解问题

1. OXE 数据的「质量分层」是否被系统研究？哪些子集对 OpenVLA 的最终鲁棒性贡献最大？（影响曦源选择 fine-tune 子集）
2. OXE 之后的新数据集（DROID、AgiBot World、RoboMIND）相对 OXE 在鲁棒性维度上的边际贡献？是否值得纳入曦源训练？

## 🔗 关联笔记

- **直接受益模型**：RT-1-X / RT-2-X（同论文）、Octo（24.05）、OpenVLA（24.06）、π0（24.10）、RDT-1B（24.10）、NORA（25.04）。
- **数据集前身**：BridgeData V1/V2（22.10、23.08）、RT-1 dataset（22.12）——OXE 核心子集。
- **数据集后继**：DROID（24.03，Stanford）、AgiBot World（24.12，中国）、RoboMIND（24.12）——OXE 体系扩展。
- **格式标准**：RLDS（Open X-Embodiment 推动的统一格式）。
- **方法学影响**：FAST tokenizer（25.01）——在 OXE 上训练并测试新 tokenization。
