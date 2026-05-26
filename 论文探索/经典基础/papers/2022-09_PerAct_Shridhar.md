---
title: "Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation"
authors: "Mohit Shridhar, Lucas Manuelli, Dieter Fox (UW + NVIDIA)"
year: "2022"
venue: "CoRL 2022"
arxiv: "2209.05451"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, 3D, voxel, multi-task, transformer]
---

# PerAct

> [!info] 元信息
> UW + NVIDIA Dieter Fox 组 2022.09 发布。**3D voxel-based multi-task manipulation transformer**。把场景表示为 100³ 体素，用 Perceiver IO 处理，输出 next best voxel + 旋转 + gripper 作为下一个 keyframe。是 3D representation VLA 路线的奠基论文，启发了 RVT、3D Diffuser Actor、ManiGaussian 等一系列后续工作。在 RLBench 18 任务上 multi-task 训练，10 条 demo / 任务即可学会。

## 📄 TL;DR（100-150 字）

PerAct 把 manipulation 重新框定为「**选择下一个 best voxel**」问题。输入：RGB-D 多视角融合的 100³ 体素 + 语言指令；用 Perceiver IO 编码（latent attention 大幅降低 voxel grid 计算量），输出 next voxel coordinate（离散）+ end-effector rotation + gripper open/close。这是 **keyframe-based action representation**：不预测连续 trajectory，只预测下一个关键 waypoint，由 motion planner 完成中间轨迹。在 RLBench 18 任务、每任务 100 demos 下 multi-task train，平均成功率 49.4%，远超 baseline。

## 🧠 我的思考

### 在 VLA 谱系中的位置

PerAct 在 VLA 谱系里代表**「3D + voxel + keyframe」分支**——和 RT-2 / OpenVLA 的 2D image + dense action 主流路线平行。这条路线在 2024-2025 仍然活跃，对**精细操作 + 强几何推理** 任务有独特优势。

**3D / keyframe VLA 谱系时间线**：
- C2F-ARM（21.07）：keyframe-based manipulation 雏形（Shridhar 同作者）；
- CLIPort（21.09）：2D top-down view + CLIP feature 的语言条件 pick-place（PerAct 同作者）；
- **PerAct（22.09）：3D voxel + Perceiver 的奠基**；
- RVT（23.06）：3D voxel → 多视角 2D 渲染，效率大幅提升；
- Act3D（23.06）、3D Diffuser Actor（24.02）：用 point cloud 替代 voxel；
- RVT-2（24.06）、3DDA（24.02）：进一步优化精度和速度；
- ManiGaussian（24.03）：3D Gaussian Splatting 表征；
- GraspNet + VLA 融合的各种工作（24-25）。

PerAct **不在 OpenVLA / Octo / π0 的直接前身链上**，但被它们在「3D 替代方案」处引用，也被 RVT、3DDA 等直接继承。可以理解为 VLA 大树的**「3D 分支」起点**。

### 核心方法

**输入处理**：
1. **3D 场景重建**：4 个 RGB-D 摄像头融合，得到 100×100×100 voxel grid（每个 voxel 含 RGB + occupancy）；
2. **语言**：CLIP text encoder embed；
3. **机器人状态**：proprioception → MLP embed。

**核心架构 Perceiver IO**：
- 100³ = 1M voxel，直接 self-attention 不可行（O(N²) 爆炸）；
- Perceiver IO 用 **可学习的 K=2048 latent queries** cross-attend voxel + language + state；
- Latents 之间做 self-attention（O(K²) 可控）；
- 输出 head 用 latents cross-attend 回 voxel grid，输出每个 voxel 的「next-best-voxel」logits + 6-DoF rotation + gripper。

**动作表征**（关键）：
- **离散 voxel coordinate**：next best voxel index ∈ [0, 100³]，等价于离散化的 3D 位置；
- **离散旋转**：72 bins per axis（5° 分辨率）；
- **gripper**：binary；
- **执行**：sample best voxel → BiRRT motion planner 生成 trajectory 到达。

**数据效率**：RLBench 18 任务，每任务 100 demos，multi-task train 一个模型。

### 鲁棒性视角下的解读

PerAct 的鲁棒性 profile 和 RT-2 / OpenVLA 路线截然不同：

**天生鲁棒的部分**（3D 表征带来的）：
1. **3D 几何不变性**：voxel 是 3D 世界坐标，摄像头视角变化对 voxel 表征影响小——这是 2D 路线（OpenVLA）做不到的；
2. **跨场景泛化**：场景几何结构（桌子高度、物体相对位置）比 2D 像素更稳定；
3. **Keyframe action**：只预测关键 waypoint，由 motion planner 处理中间轨迹——天然 robust to compounding error；
4. **数据效率高**：100 demos / 任务即可 multi-task 训练，相对 RT-1 (>1000 demos/task) 数据效率高一个数量级；
5. **物理 grounded**：3D 体素表征对物理碰撞、空间约束有直接表达。

**天生脆弱的部分**：
1. **依赖准确深度**：RGB-D 缺失或噪声大时整个 voxel grid 失真，比纯 2D 路线脆弱；
2. **依赖摄像头校准**：4 个 RGB-D 摄像头外参必须精确，否则 voxel 融合错位；
3. **没有 LLM 先验**：CLIP text encoder 远弱于 Llama-2，对复杂语义鲁棒性差；
4. **离散 voxel 分辨率粗**：100³ = 1cm 精度（假设 1m 场景），毫米级精细操作不行；
5. **依赖 motion planner**：planner 失败（碰撞、不可达）即任务失败，不是端到端的；
6. **小规模数据**：RLBench 主要是模拟数据，真机泛化未被充分验证；
7. **场景结构假设**：固定的 100³ voxel grid 假设场景有限大小，对开放场景不友好。

后续鲁棒性改进路线：
- **RVT**：voxel → 多视角 2D 渲染，降低对深度精度的依赖；
- **3D Diffuser Actor**：连续 action + diffusion，提高精度；
- **3D Gaussian VLA**：更精确的 3D 表征；
- **PerAct² / RVT-2**：scale 数据规模 + 真机验证。

### 与曦源（鲁棒性主线）的关联

PerAct **不建议作为曦源的主 baseline**，但有重要参考价值：

**不作为 baseline 的理由**：
1. **路线不同**：PerAct 是 3D voxel + keyframe，曦源如果研究主流 VLA-as-language 鲁棒性，PerAct 不在对照线上；
2. **依赖 RGB-D + 多摄像头标定**：曦源如果没有这些硬件条件，跑不起来；
3. **RLBench 是模拟环境**：真机验证少，鲁棒性结论可能不通用；
4. **已被 RVT、3DDA 超越**：如果要用 3D baseline，应该用 RVT 或 3DDA。

**作为综述论据的价值**：
1. 讨论「3D vs 2D 输入对鲁棒性的影响」时是关键案例；
2. 讨论「keyframe vs dense action 对 compounding error 的影响」时是奠基论文；
3. 讨论「动作离散化粒度 vs 精度」时是另一种思路（PerAct 离散化的是位置而非速度）；
4. 论证「VLA 不是只有 RT-2 一条路」的多样性背景。

46GB 显存对 PerAct 完全够用（Perceiver IO 模型不大），代码开源（peract repo 活跃）。

### 待解问题

1. PerAct 的 3D 表征鲁棒性在真机条件下（深度噪声 + 外参漂移）实际表现如何？是否真的优于 2D？
2. Keyframe action 在 dynamic 任务（需要力觉反馈、动态物体）上为什么效果差？这对曦源是否要考虑混合 keyframe + dense action？

## 🔗 关联笔记

- **同作者前身**：CLIPort（21.09）——2D 版本；C2F-ARM（21.07）——keyframe action 雏形。
- **3D 路线后继**：RVT（23.06）——多视角 2D 渲染；3D Diffuser Actor（24.02）——point cloud + diffusion。
- **范式对照**：RT-2（23.07）、OpenVLA（24.06）——2D + dense action 主流路线。
- **数据基础**：RLBench（19）——仿真 benchmark。
- **3D 进化**：ManiGaussian、SplatLNK——Gaussian Splatting + manipulation。
