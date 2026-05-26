---
title: "RT-1: Robotics Transformer for Real-World Control at Scale"
authors: "Anthony Brohan, Noah Brown, Justice Carbajal, et al. (Google Robotics)"
year: "2022"
venue: "RSS 2023"
arxiv: "2212.06817"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, manipulation, real-world]
---

# RT-1

> [!info] 元信息
> Google Robotics 2022 年发布。35M 参数 Transformer，13 万条真机数据，700+ 任务。是「第一个真正在数千次真机评测下证明 scaling laws 对机器人有效」的工作，奠定了「大规模真机数据 + Transformer + 语言条件」的 VLA 范式。被几乎所有后续 VLA 引用作为 baseline。

## 📄 TL;DR（100-150 字）

RT-1 用 35M 参数的 EfficientNet+FiLM+Transformer 架构，在 17 个月里收集的 13 万条真机演示（700+ 任务、13 台 Everyday Robot）上做模仿学习。动作离散化为 256 bin，自回归生成。实验证明：（1）大规模真机数据下模型 scale 越大性能越好；（2）训练任务数从 200 涨到 700 时，未见任务的成功率从 24% 涨到 76%；（3）能吸收异构数据（Kuka、模拟器）而不损害性能。是 VLA 范式的第一个 real-world 验证。

## 🧠 我的思考

### 在 VLA 谱系中的位置

RT-1 在 VLA 谱系里是**真正意义上的「VLA 元年」标志**。Gato 证明了多任务序列建模可行，但机器人部分只是「玩具」；RT-1 第一次在真机上、大规模任务上、真实场景里把这套范式做扎实。

**时间线上的关键节点**：
- Gato（22.05）：多任务统一接口的概念证明；
- BC-Z（21.09）、CLIPort（21.09）：早期语言条件 manipulation；
- **RT-1（22.12）：第一个真机大规模 VLA**；
- PaLM-E（23.03）：把 LLM 接进来做规划；
- RT-2（23.07）：把 RT-1 的 backbone 替换成 PaLI-X，VLM→VLA；
- Open X-Embodiment / RT-X（23.10）：把 RT-1/RT-2 在 22 个机体上重训，证明跨机体迁移有效；
- OpenVLA（24.06）：RT-2 的开源版本，把 Llama-2 + RT-2 范式开源。

引用关系：RT-1 引用了 Gato、BC-Z、Decision Transformer；被 RT-2、Octo、OpenVLA、Open X-Embodiment、ECoT、FAST 等几乎所有后续 VLA 引用。它的数据集（RT-1 dataset）也成了 Open X-Embodiment 里最大的子集之一。

### 核心方法

**架构**：分阶段处理：
1. **图像编码**：EfficientNet-B3（预训练 ImageNet），输入 6 帧 300×300 RGB；
2. **语言条件**：USE（Universal Sentence Encoder）embed 指令，通过 **FiLM 层**调制图像特征——这是关键设计，让语言深度参与视觉特征提取；
3. **TokenLearner**：把 9×9×512 的特征图压缩成 8 个 token / 帧；
4. **Transformer**：8 层 decoder，处理 48 个 token（6 帧×8），输出 11 维动作（7-DoF arm + 3-DoF base + 1-DoF mode）；
5. **动作离散化**：每个维度独立离散化到 256 bin，自回归生成。

**数据**：13 个 Everyday Robot 17 个月，700+ 任务，130k episodes，覆盖「pick, place, open drawer, knock over」等基础操作。

**关键实验结论**：
- 模型容量 scaling 显著（35M 比小模型好）；
- 任务数 scaling 显著（700 任务的未见泛化远强于 200 任务）；
- 异构数据共训不损害性能反而轻微提升。

### 鲁棒性视角下的解读

RT-1 的设计选择对鲁棒性影响极深，奠定了后续多年的研究方向：

**天生脆弱的部分**：
1. **256-bin 离散化**：精细操作粒度不足，对小物体抓取、毫米级对齐力不从心。后续 ACT、Diffusion Policy 用连续动作 / 多模态分布解决；FAST tokenizer 用 DCT 解决；
2. **EfficientNet 从头训 / ImageNet 弱预训练**：视觉表征对训练分布外的纹理、光照变化敏感。OpenVLA 用 DINOv2+SigLIP 大幅改善；
3. **没有 history beyond 6 帧**：长时序任务（如 Mobile ALOHA 的擦桌子）几乎不可能；
4. **依赖 USE 词向量**：语义理解远弱于 LLM，对长指令、同义改写脆弱。PaLM-E / RT-2 用 LLM 替换是直接回应；
5. **纯模仿学习无 recovery**：训练分布外即崩溃。ECoT 用 CoT 推理增强、Diffusion Policy 用 multi-modal 分布在一定程度缓解。

**天生鲁棒的部分**：
1. **FiLM 语言深度调制**：比 late fusion 鲁棒得多，被 Octo 继承；
2. **TokenLearner**：减少 token 数提高样本效率，对小数据鲁棒；
3. **大规模真机数据本身**就是最强的鲁棒性来源，700 任务下的 in-domain 表现非常稳定。

后续鲁棒性改进路线：（A）换更强的视觉 backbone → OpenVLA；（B）换更强的语言模型 → RT-2；（C）换连续动作头 → π0、Octo；（D）加 CoT 推理 → ECoT；（E）加 recovery / online RL → SERL、HIL-SERL。

### 与曦源（鲁棒性主线）的关联

RT-1 **不建议作为曦源的主 baseline**，但有特殊价值：
1. **权重未开源**（Google 内部），数据集部分开源（在 Open X-Embodiment 里）；
2. **35M 参数虽然在 46GB 显存上能跑**，但社区基本不再用 RT-1 直接对比，都用 OpenVLA / Octo；
3. **架构已被 OpenVLA、Octo 完全超越**，性能差距大。

但要在综述里**重点叙述**——它是「VLA scaling laws」的起点论文，所有鲁棒性研究讨论 baseline 鲁棒性时绕不开它，特别是讨论「离散化误差」「视觉脆弱性」时它是教科书案例。

### 待解问题

1. RT-1 的 FiLM 语言调制在 OpenVLA（去掉 FiLM 改 LLM 主干）后是否被验证为「冗余」还是「损失」？（影响是否要在曦源研究里恢复某种深度语言调制）
2. RT-1 dataset 在 Open X-Embodiment 里的子集质量如何？是否适合做曦源的鲁棒性 fine-tune 测试基底？

## 🔗 关联笔记

- **直接前身**：Gato（22.05）、BC-Z（21.09）——多任务模仿学习思想来源。
- **直接后继**：RT-2（23.07）——同一团队的 backbone 升级版（EfficientNet → PaLI-X）。
- **数据扩展**：Open X-Embodiment（23.10）——RT-1 dataset 是其核心子集。
- **开源对标**：OpenVLA（24.06）——可以理解为「开源版 RT-2」，全面超越 RT-1。
- **鲁棒性改进**：ECoT（24.07）——直接在 OpenVLA 基础上加 CoT 改善 RT-1 范式的脆弱性。
