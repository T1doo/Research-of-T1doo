---
title: "TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies"
authors: "Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, Jianwei Yang"
year: "2024"
venue: "arXiv 2412"
arxiv: "2412.10345"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 主线A, 视觉利用, patch-selection, visual-prompting, spatial-temporal]
---

# TraceVLA

> [!info] 元信息
> - **作者**：Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, Jianwei Yang
> - **机构**：UMD、Microsoft Research
> - **arXiv**：[2412.10345](https://arxiv.org/abs/2412.10345)

## 📄 TL;DR

TraceVLA 提出 **visual trace prompting**——把过去的 state-action 轨迹**直接画在图像上**作为 visual prompt，让 VLA 模型通过"看图"获得 spatial-temporal awareness。在 OpenVLA 基础上 fine-tune（150k manipulation trajectory 自建数据集），SimplerEnv 比 OpenVLA 提升 10%（137 配置）、WidowX 真机 **3.5×** 性能。一个 4B Phi-3-Vision 紧凑模型即可达到 7B OpenVLA 同等性能，推理速度更快。

## 🧠 我的思考

### 核心观点

1. **"画轨迹在图上"是优雅得不可思议的设计**：传统给 VLA 提供历史信息的方式是 concatenate 多帧图像（计算贵）或加 temporal embedding（间接）。直接把过去的 gripper 轨迹**渲染**到当前帧上，让 VLM 的图像理解能力直接 parse temporal 信息——完全利用了 VLM 已有的视觉理解能力。
2. **Visual prompt 是 VLA 的低成本时序工具**：不需要改架构、不需要多帧 input、不需要加 token，只在 image 上画线。是 attention guidance 的"prompt engineering"版本——把信号编码到 input image 而非 attention 监督。
3. **4B 模型 ≥ 7B baseline 是参数效率胜利**：visual trace prompting 充当 inductive bias，等效于让小模型获得更强的时空理解。这与曦源"轻量 backbone"的约束高度契合。
4. **3.5× 真机提升是非常 striking 的数字**：4 个 WidowX 任务上 3.5×（即从 ~15% 到 ~50% 这种量级），暗示真机部署中"时空理解缺失"是主要 failure mode，trace prompt 直接命中。

### 方法论关键

- **Trace rendering**：把过去 H 步的 end-effector 位置投影到当前帧 image 上，画成彩色折线（颜色编码时间），形成 trace overlay。
- **训练数据**：150k trajectory 自建数据集，每条 trajectory 渲染对应 trace prompt + 原 instruction → 形成 (image_with_trace, instruction, action) triple。
- **Fine-tune**：在 OpenVLA / Phi-3-Vision 上 fine-tune，VLM backbone 直接 process 带 trace 的 image。
- **推理**：每帧实时渲染最近 H 步 trace，feed 到 VLA。

### 关键实验数据

| Benchmark | OpenVLA (7B) | **TraceVLA (4B)** | Δ |
|---|---|---|---|
| SimplerEnv (137 cfg) | base | **+10%** | ↑ |
| WidowX real (4 tasks) | base | **3.5×** | ↑↑↑ |
| 推理速度 | 1× | faster | ↑ |

参数减半、性能优于 7B baseline。

### 与曦源（FocusVLA 主线）的关联

TraceVLA 是与 FocusVLA 几乎 **正交**的工作——FocusVLA 改 attention 架构、TraceVLA 改 input image。两者强互补：

1. **TraceVLA + FocusVLA 可叠加**：在 TraceVLA 生成的带 trace overlay 的 image 上，应用 FocusVLA 的 Focus Attention 做 patch 选择。trace 提供时空 prior，Focus Attention 在此基础上做 task-relevant patch 选择。理论上是 "1+1>2"。
2. **TraceVLA 解决 FocusVLA 留的"时序信息不足"问题**：FocusVLA 只处理单帧 image，对需要历史信息的任务（如"继续按之前方向移动"）天然弱。trace prompt 是补救手段。
3. **共享"轻量 backbone + 巧设计 > 大模型"哲学**：TraceVLA 用 4B Phi-3 + visual prompt 打败 7B OpenVLA，FocusVLA 用 0.5B + Focus Attention 打败 7B Spatial Forcing。两者都验证了曦源"小模型 + 巧架构"的研究方向。
4. **可作为 baseline 直接对比**：在 LIBERO 上用同 backbone 跑 TraceVLA 和 FocusVLA，看哪个组件对成功率贡献更大，或两者叠加是否有 superlinear 提升。

**研究空白**：TraceVLA 只画 end-effector trace，没画 object trace（被操作物体的运动）。在 multi-object 任务中，物体轨迹同样重要。这是个明显的扩展方向。

**对鲁棒性的启示**：visual trace prompt 是在 image 上"画"信号，本身可能成为新的扰动源（trace 被画错位会误导 policy）。需要研究 trace prompt 在视觉扰动 / pose estimation 噪声下的鲁棒性。

### 待解问题

- H（trace 历史长度）如何选？太短没用、太长画乱图。
- 第三人称 vs 第一人称视角下 trace 渲染的差异？
- 是否能扩展到 object trace（不只 gripper）？
- 与 [[2026-03_ST-VLA_Wu]] 的 4D mask 比，trace prompt 是更轻量的 alternative？

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]
- [[2026-05_SpatialVLA_Qu]]
- [[2026-03_ST-VLA_Wu]]
- [[2026-05_GuidedVLA_Jia]]
- [[2024_EF-VLA_Anon]]
