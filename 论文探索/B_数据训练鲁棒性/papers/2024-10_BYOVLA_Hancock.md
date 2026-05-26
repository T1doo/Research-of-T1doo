---
title: "BYOVLA: Bring Your Own VLA — Run-time Observation Interventions for Robust Generalization"
authors: "Hancock et al."
year: "2024"
venue: "arXiv 2410.01971"
arxiv: "2410.01971"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 推理时干预"
tags: [literature, T1D, 主线B, 数据增广, 推理时干预, BYOVLA, RobustVLA对照]
---

# BYOVLA — Bring Your Own VLA + Run-time Observation Interventions

> [!info] 元信息
> - **作者**：Hancock et al.
> - **arXiv**：[2410.01971](https://arxiv.org/abs/2410.01971)
> - **日期**：2024-10
> - **核心定位**：**RobustVLA 的对照 baseline**。推理时（run-time）对视觉观察做干预（背景/干扰物清理），不修改 VLA 本身
> - **本地 PDF**：未缓存

## 📄 TL;DR

BYOVLA 提出一种**模型无关、推理时介入**的 VLA 鲁棒化方案：在 VLA 看图前，用一组外部模块（VLM 描述 + SAM 分割 + LaMa inpaint）把背景换成训练分布的纯色、把任务无关物体擦掉，**让 VLA 只看到"干净"图像**。优点：不需要重训 VLA，可即插即用到 OpenVLA / RT-2 / Octo；缺点：依赖外部 LLM/VLM，**推理 50.6× 慢于 RobustVLA**（这是 RobustVLA 论文专门 callout 的对照点）。在真机评测：BYOVLA 在视觉扰动下 SR +7-22%，但在 action / instruction / environment 扰动上 0% 提升——只解决视觉模态。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **推理时干预 vs 训练时增广 vs 训练时对抗：三条 VLA 鲁棒化路径的根本分歧**：
   - 推理时干预（BYOVLA）：把分布外输入"洗"成分布内
   - 训练时增广（GreenAug/ROSIE/RoboEngine）：把分布内训练样本"扩"到分布外
   - 训练时对抗（RobustVLA）：在 inner max 找 worst-case 然后 outer min 训鲁棒
   - **三者正交**，理论上可堆叠
   
2. **「外部 LLM 依赖」是 BYOVLA 致命缺陷**：推理时调 VLM/SAM/LaMa 链路，延迟数百 ms，**边缘部署根本不可行**。RobustVLA 在论文里专门强调 50.6× 加速正是基于此对比。

3. **只解视觉扰动 = 治标不治本**：RobustVLA 的 Table 5 显示 BYOVLA 在 action / instruction / environment 上提升 **0.0%**——说明"清洗视觉"完全无助于其他模态。**这是 RobustVLA 故事的重要反衬**。

### 方法论

- **三步推理流水线**：
  1. **VLM (e.g., GPT-4V)** 描述图像 + 识别任务无关物体
  2. **SAM** 分割识别出的干扰物 + 背景
  3. **LaMa (inpainting)** 把背景换成训练分布中的纯色 / 把干扰物擦掉
- **不修改 VLA**：处理完图像直接送入 OpenVLA / RT-2 / Octo
- **任务**：真机 pick-and-place 在多种背景 / 干扰物条件下

### 实验关键数据（来源：RobustVLA Table 5 对比）

| 扰动 | OpenVLA Baseline | BYOVLA | RobustVLA |
|---|---|---|---|
| 视觉高斯噪声 | 67% | 74% (+7.3) | 78% (+11.0) |
| 视觉死像素 | 51% | 73% (+22.3) | 76% (+25.0) |
| Action 噪声 | 52% | 52% (+0.0) | 78% (+26.0) |
| Instruction 改写 | 60% | 60% (+0.0) | 82% (+22.0) |
| Environment 干扰物 | 58% | 65% (+7.0) | 84% (+26.0) |
| **推理延迟** | 基准 | **50.6×** | 1× (无额外开销) |

### 与 [[2026_RobustVLA_Guo]] 的关系（B2 类必答）

| 维度 | BYOVLA | RobustVLA |
|---|---|---|
| 路径 | 推理时干预 | 训练时对抗 |
| 修改 VLA | 否 | 是 |
| 推理延迟 | **50.6×** | 1× |
| 鲁棒范围 | 仅视觉 | 17 类多模态 |
| 关系 | **RobustVLA 论文的对照 baseline**——RobustVLA 用此对比凸显自身在精度+延迟双维度的优势 |

→ **BYOVLA 与 RobustVLA 不互斥，可堆叠**：BYOVLA 在前级洗输入，RobustVLA 训鲁棒模型；但实际工程中 BYOVLA 的延迟代价让此组合不现实。**真正可组合的是 RobustVLA + 训练时数据增广（GreenAug/ROSIE）**——它们都是 offline 操作，无推理开销。

### 与 RobustVLA input-robust 模块的结合可能性（B2 类必答）

- **可以结合**：BYOVLA 的"识别任务相关区域"思路可作为 RobustVLA input-robust 的 augmentation 策略之一——用 SAM/VLM 分割出任务区域，做语义保留的扰动（如换背景），生成 ω*(o)。
- **但有局限**：BYOVLA 的 SAM+VLM pipeline 太重，应当**离线**用它生成训练数据，而非 online run-time。
- **建议组合**：用 BYOVLA pipeline **离线生成** "干净 + 扰动" 配对，喂给 RobustVLA 的 input-robust 训练。

### 与曦源关联

1. **曦源做"对照实验"必须包含 BYOVLA**：它是 VLA 鲁棒性领域的事实 baseline。
2. **推理时干预的死路 → 训练时方法的活路**：BYOVLA 的失败论证了曦源选 RobustVLA 路线（训练时改造）的正确性。
3. **SAM + VLM 工具链可借用做数据生成**：离线生成增广数据，规避推理延迟。

### 待解问题

1. **BYOVLA + GreenAug 组合**：BYOVLA 把背景洗白 + GreenAug 把背景换花，谁更鲁棒？
2. **SAM 分割错误对 BYOVLA 的影响**：作者未充分研究 SAM 失败模式。
3. **BYOVLA 在小模型 (SmolVLA) 上的延迟**：基础 VLA 越小，外部 LLM/VLM 占比越夸张。

%% end my-thoughts %%

## 🔗 关联笔记

- **直接对照**：[[2026_RobustVLA_Guo]]（RobustVLA 的核心 baseline）
- **同类训练时增广（可组合）**：[[2024_GreenAug]]、[[2024_ROSIE]]、[[2025_RoboEngine]]
- **同路径推理时干预**：[[2025_Model-Agnostic_AdvDef]] (diffusion purification, 更轻量)
- **被改造对象**：[[2024_OpenVLA]]、[[2023_RT-2]]
- **底层工具链**：SAM (Kirillov 2023)、LaMa (Suvorov 2022)
- **评测平台**：[[2025_Eva-VLA]]、[[2026_VLA-Risk]]
