---
title: "Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation"
authors: "Zipeng Fu, Tony Z. Zhao, Chelsea Finn (Stanford)"
year: "2024"
venue: "CoRL 2024"
arxiv: "2401.02117"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, ACT, bimanual, mobile, imitation]
---

# Mobile ALOHA

> [!info] 元信息
> Stanford Chelsea Finn 组 2024.01 发布。在 ALOHA（21.04）双臂遥操作平台上加上移动底盘，全身遥操作。架构沿用 ACT（Action Chunking Transformer），关键创新在数据采集 + co-training 策略——用 50 条移动演示 + 静态 ALOHA 数据 co-train，实现了「炒虾、擦桌、推椅」等长时序双手 + 移动任务。是 ACT 谱系最广为人知的工作，演示视频火爆全网。

## 📄 TL;DR（100-150 字）

Mobile ALOHA 提出低成本（约 32k 美元）移动双臂遥操作平台，配 14-DoF 全身遥操作。模型架构沿用 ACT（CVAE + Transformer，输出动作 chunk）。核心发现：用静态 ALOHA 数据集（825 条演示，跨任务）与移动数据 co-train，可以让 50 条 / 任务的小数据集学会 7 个复杂长时序任务（炒虾、擦红酒、推椅）。证明了**模仿学习 + 高质量遥操作 + 跨任务 co-training** 在小数据规模下可以解决复杂双手任务，不需要大模型 / 大数据。

## 🧠 我的思考

### 在 VLA 谱系中的位置

Mobile ALOHA 属于 VLA 谱系里**「ACT 分支」**的代表作。VLA 不是只有 RT-2 / OpenVLA 那条 LLM 路线，还有完全平行的「专用 imitation learning policy」路线——ACT 是其旗手。

**ACT 谱系时间线**：
- ALOHA（23.04，Tony Zhao）：低成本双臂遥操作 + ACT 算法奠基；
- ACT（同 ALOHA 论文）：CVAE + Transformer + action chunking，针对高频精细操作；
- Diffusion Policy（23.03，Chi）：并行路线，用 diffusion 替代 CVAE；
- **Mobile ALOHA（24.01）：ACT 扩展到移动双臂**；
- ALOHA Unleashed（24.10，Google）：再扩到 64 任务、大规模数据；
- π0（24.10，Physical Intelligence）：把 ACT/Diffusion Policy 思想与 VLM 融合，flow matching 输出；
- π0.5（25.04）：π0 的开源 / fine-tune 友好版本。

时间线上 Mobile ALOHA 处于 ACT 范式的**「应用爆发期」**——证明小团队、有限数据、专用平台可以打出非常 impressive 的 demo。它**不引用 RT-2**作为方法学基础（路线不同），但被 RT-2 阵营在「multi-task imitation」讨论里反复引用。

### 核心方法

**硬件**：
- AgileX Tracer AGV 底盘 + 2× ViperX 6-DoF 机械臂；
- 全身遥操作：腰部带 follower 跟随，操作员推动底盘即时反馈；
- 4 个 RealSense 摄像头（前 / 左腕 / 右腕 / 后）；
- 单台机器 32k 美元。

**算法 ACT（沿用）**：
- 输入：4 摄像头图像 + 14 维 joint state；
- 编码：ResNet-18 per camera + Transformer encoder；
- CVAE 结构：训练时编码 GT action chunk 为 z，推理时 z=0；
- 输出：64 步 action chunk（chunking 是关键，避免 compounding error）；
- 推理时用 temporal ensembling 平滑。

**Co-training 关键技巧**：
- 静态 ALOHA 数据集 825 episodes（来自 23.04 论文）；
- 每个新移动任务只采 50 条；
- 训练时按 1:1 mix 静态 + 移动数据；
- 结果：50 条数据就能学会，无 co-training 完全失败。

**任务**：7 个长时序任务——炒虾煎蛋、擦红酒洒地、推椅子靠桌、给毛巾叠角、用电梯按钮、给人击掌、储物柜放物。

### 鲁棒性视角下的解读

Mobile ALOHA 在鲁棒性谱系里代表**「专用 imitation learning」的鲁棒性极限**：

**天生鲁棒的部分**：
1. **Action chunking**：输出 64 步动作 chunk 显著减少 compounding error，对短时扰动鲁棒——这是 ACT 范式相对单步预测最大的鲁棒性 buff；
2. **Temporal ensembling**：推理时对重叠 chunk 加权平均，平滑控制；
3. **高质量遥操作数据**：人类自然的双手协调、反应、调整都隐含在数据里，比脚本数据鲁棒得多；
4. **Co-training 跨任务**：减小 overfit，对相似任务有一定 transfer。

**天生脆弱的部分**：
1. **没有语言条件**：每个任务一个 head 或 task ID，对自然语言指令、零样本任务完全没有能力——这是和 VLA 路线的根本差距；
2. **没有 VLM 先验**：视觉表征是 ImageNet-pretrain ResNet-18，对未见物体、背景变化脆弱；
3. **数据规模小（50/任务）**：分布外即崩溃，且鲁棒性高度依赖采集时的环境多样性；
4. **专用硬件 lock-in**：策略和 ALOHA 平台耦合，迁移到其他机器人需要重训；
5. **CVAE z=0 推理**：discard 了多模态信息，对多解任务（如多种抓取方式）可能不如 diffusion 鲁棒。

后续鲁棒性改进路线：
- **加语言** → ALOHA Unleashed 引入语言条件；
- **加 VLM** → π0 把 ACT/Diffusion 思想与 VLM 融合；
- **加规模** → ALOHA Unleashed 64 任务、26000 episodes；
- **加 diffusion** → Diffusion Policy 已经是平行方案。

### 与曦源（鲁棒性主线）的关联

Mobile ALOHA **不建议作为曦源的 baseline**，但要在综述里讨论：
1. **硬件强绑定**：没有 Mobile ALOHA 平台跑不起来，曦源大概率没有；
2. **任务不是开放语言指令的 VLA**，与曦源研究的 VLA 鲁棒性主题对不上；
3. **代码 / 数据开源（teleop-anything、aloha repo 都活跃）**，可以学习其 ACT 实现细节。

**曦源用法**：
- 综述里讨论「imitation learning 路线」和「VLA-as-language 路线」的分野时引用；
- 讨论「action chunking 对鲁棒性的贡献」时引用——这是普适设计，可以借鉴到任何 VLA；
- 讨论「专用 vs 通用」的取舍时引用。

46GB 显存对 Mobile ALOHA 完全充足（ACT 模型小，<100M 参数）。如果曦源研究里想做「专用 IL + VLM」的对比，Mobile ALOHA 是可选的小 baseline。

### 待解问题

1. Mobile ALOHA 的 co-training 是否依赖任务相似性？跨 modality co-train（如 mix 模拟数据）效果如何？（影响曦源做 sim-to-real 鲁棒性的策略）
2. Action chunking 在 OpenVLA / π0 等 VLA 里的对应设计是什么？是不是普适的鲁棒性 trick？

## 🔗 关联笔记

- **直接前身**：ALOHA + ACT（23.04，Tony Zhao）——硬件和算法基础。
- **同代并行**：Diffusion Policy（23.03）——另一条非 VLA 的 imitation learning 主线。
- **后续扩展**：ALOHA Unleashed（24.10，Google）——规模化版本。
- **融合 VLA**：π0（24.10）——把 ACT 思想注入 VLM 路线。
- **范式对照**：RT-2（23.07）、OpenVLA（24.06）——VLA-as-language 的平行路线。
