---
title: "ROSIE: Scaling Robot Learning with Semantically Imagined Experience (Diffusion Inpainting + Text-guided Augmentation)"
authors: "Yu et al."
year: "2023"
venue: "arXiv 2302.11550 / RSS 2023"
arxiv: "2302.11550"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 增广开山之作"
tags: [literature, T1D, 主线B, 数据增广, diffusion inpainting, text-guided, ROSIE, 经典]
---

# ROSIE — Diffusion Inpainting + Text-Guided 增广开山之作

> [!info] 元信息
> - **作者**：Yu et al. (Google DeepMind / Robotics at Google)
> - **arXiv**：[2302.11550](https://arxiv.org/abs/2302.11550)
> - **发表**：RSS 2023
> - **核心定位**：robotics 视觉数据增广的**开山之作**，首次用 diffusion model 做 text-guided semantic augmentation
> - **本地 PDF**：未缓存

## 📄 TL;DR

ROSIE 是 2023 年 robotics 视觉数据增广方向的奠基性工作，由 Google DeepMind Robotics 团队提出。核心方法：用 **Imagen Editor (diffusion inpainting)** + **文本描述**对演示视频做 semantic augmentation——"把这只杯子换成苹果"、"把背景换成厨房"、"加 3 个干扰物"。所有增广由自然语言指令驱动，VLM 解释指令，diffusion 模型执行。在真机 RT-1 训练上：ROSIE 增广使新背景 SR 从 47% → 70%，新物体 SR 从 32% → 56%。论文奠定了**"text-guided diffusion augmentation"** 这条技术路线，后续 RoboEngine / GreenAug / RoboAug 都是其变体或简化版。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **ROSIE 是 robotics 数据增广领域的 "GPT-3 时刻"**：用 LLM 提示生成增广 + diffusion 模型执行，把数据增广从「pixel-level（crop/flip）」推到「semantic-level（换物体/换场景）」。后续 3 年所有相关工作都在此范式内打转。

2. **「Text-guided」 = 任意可表达的增广空间**：传统增广只能"色彩抖动 / 几何变换"，ROSIE 用文本指令可任意指定增广内容（"换厨房背景"、"杯子换苹果"），增广空间无限大。**这是 robotics 数据增广的真正 scalability 突破**。

3. **「先驱者税」体现在 2023 年的工程瓶颈**：ROSIE 用 Imagen Editor (Google 内部模型)，外部不可用；diffusion inpainting 当时还很慢（每图 10s+）；这些问题被后续 ControlNet / SD-XL / SAM2 解决。**2026 年回看 ROSIE，方法论先进，工程过时**。

### 方法论

- **三阶段流水线**：
  1. **VLM 解释指令**：用户写"把杯子换成苹果"，VLM 输出 mask + 替换 prompt
  2. **Diffusion inpainting**：用 Imagen Editor 把 mask 区域 inpaint 为新物体
  3. **训练数据合成**：每条 demo → N 个增广 episode（背景/物体/干扰物组合）
- **训练**：RT-1 / 标准 BC，无修改
- **任务**：真机 Google Robot pick-and-place + 干扰物 + 新背景

### 实验关键数据

| Setting | RT-1 SR (new bg) | RT-1 SR (new object) | RT-1 SR (distractor) |
|---|---|---|---|
| Baseline | 47% | 32% | 38% |
| **ROSIE** | **70%** | **56%** | **64%** |

→ ROSIE 在三类视觉 OOD 上都有 20-25 pp 提升，奠定后续工作 baseline。

### 与 [[2024_BYOVLA]] 对照（B2 类必答）

| 维度 | ROSIE | BYOVLA |
|---|---|---|
| 时间 | 2023（先驱） | 2024 |
| 路径 | 训练时增广 | 推理时清洗 |
| 工具 | Imagen Editor + VLM | GPT-4V + SAM + LaMa |
| 工具开源度 | Imagen 不开源 → 受限 | SAM/LaMa 开源 → 受限于 GPT-4V API |
| 训练 vs 推理 | 增广在训练，推理零开销 | 干预在推理，每次 +50.6× 延迟 |

→ **ROSIE 与 BYOVLA 是同思想的两个时机**：都用 "VLM + 视觉生成模型"，ROSIE 离线增广、BYOVLA 在线清洗。**离线增广是工程更优选**——这也是为什么后续 RoboEngine / RoboAug 都走训练时路径。

### 能否与 RobustVLA 的 input-robust 模块结合（B2 类必答）

**可以**：
- ROSIE 提供 text-guided ω*(o) 扰动（语义保留的换背景/换物体）
- 这些 ω*(o) 配对喂给 RobustVLA input-robust loss
- **优势**：ROSIE 的语义增广比纯像素扰动（高斯噪声/色彩抖动）更接近真实部署分布

但 ROSIE 用的 Imagen Editor 不开源，**实际实现需替换为 ControlNet / SDXL Inpaint**。

### 与曦源关联

1. **ROSIE 是必读经典，但实现该用 RoboEngine / RoboAug**：曦源读 ROSIE 理解范式，用后续工作的开源实现。
2. **「Text-guided」是曦源讲故事的好抓手**："我们的数据增广支持自然语言控制，可任意生成新场景"——这是给评审看的 catchy 卖点。
3. **VLM 接口可与曦源 VLA 共用**：曦源若用 SmolVLM/Qwen2-VL 作 VLA 的 backbone，同一个 VLM 也可驱动数据增广。
4. **ROSIE 的 RT-1 baseline 过时**：曦源应在 OpenVLA / π0 上重做 ROSIE 实验，作为 2026 年新基线。

### 待解问题

1. **Imagen Editor 不开源的复现障碍**：后续工作（RoboEngine）用 ControlNet 替代，效果是否完全等同？
2. **VLM 指令解释的失败模式**：模糊指令导致 mask 错位，污染训练。
3. **ROSIE 在 LIBERO benchmark 上的成绩**：论文只测真机 RT-1，缺标准 benchmark 数据。

%% end my-thoughts %%

## 🔗 关联笔记

- **同思想后续**：[[2025_RoboEngine]]、[[2024_GreenAug]]、[[2026_RoboAug]]（ROSIE 的开源简化版）
- **同思想推理时**：[[2024_BYOVLA]]（ROSIE 的镜像在推理时实现）
- **可堆叠**：[[2026_RobustVLA_Guo]] input-robust 模块
- **底层工具**：Imagen Editor (Wang 2023)、Stable Diffusion Inpaint
- **被增广对象**：[[2023_RT-1]]，可推广到 [[2024_OpenVLA]]
- **评测平台**：[[2025_Eva-VLA]]
- **历史地位**：与 [[2018_DomainRandomization_Tobin]] 并列为 robotics 数据增广的两大思想源头
