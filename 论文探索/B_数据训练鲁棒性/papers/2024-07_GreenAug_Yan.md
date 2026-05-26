---
title: "GreenAug: Green-screen Augmentation for Robotic Manipulation"
authors: "Yan et al."
year: "2024"
venue: "arXiv 2407.07868"
arxiv: "2407.07868"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 色键增广"
tags: [literature, T1D, 主线B, 数据增广, 色键抠图, GreenAug, 真机]
---

# GreenAug — 色键抠图 + 背景纹理叠加

> [!info] 元信息
> - **作者**：Yan et al.
> - **arXiv**：[2407.07868](https://arxiv.org/abs/2407.07868)
> - **日期**：2024-07
> - **核心定位**：**最朴素却最有效**的视觉数据增广方案——在绿幕背景下采集演示，事后任意替换背景纹理
> - **本地 PDF**：未缓存

## 📄 TL;DR

GreenAug 用一个非常朴素但工程上极致省力的 trick：**在绿幕背景下采集真机演示**，事后用色键抠图技术把背景换成任意纹理（街景 / 仓库 / 抽象图案 / 训练分布）。这种方法不需要任何 inpainting / diffusion / VLM——纯像素级 chroma key 操作。结果：在 50 个真机演示训练 OpenVLA，背景从绿幕换成 100 种随机纹理 → 在新背景测试 SR 从 18% → 76%。论文核心论断：**「数据多样性 > 数据保真度」**——粗糙的合成增广 (chroma key 拼接) 远优于精细的零增广基线。GreenAug 是 RoboEngine / ROSIE 的简化版前驱。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「数据多样性 > 数据保真度」是数据增广领域被低估的金句**：业界倾向用越来越精细的 generative model 做增广 (ROSIE, RoboEngine, RoboVIP)，但 GreenAug 证明 chroma key 这种「视觉上明显是假」的拼接已经足够带来大幅鲁棒提升。**对曦源的含义**：不要过早投入资源做 diffusion-based 增广，先用 GreenAug 验证 baseline。

2. **「采集时主动设计」 vs 「采集后被动增广」**：GreenAug 的关键创新是 **在数据采集环节就为增广做准备**——绿幕环境让后期抠图零成本。这是数据工程思维而非算法思维。**对曦源的含义**：曦源若做真机数据采集，应直接设计绿幕环境。

3. **真机增广的实际效果可量化**：GreenAug 在 50 个演示上提升 4.2 倍 SR (18% → 76%)，比 sim-to-real / domain randomization 等更高效。

### 方法论

- **数据采集**：在绿色背景前演示，用单目相机录制
- **色键抠图**：HSV 空间分离绿色，得到前景 mask
- **背景合成**：从 100 个候选背景纹理（COCO / 室内场景 / 抽象图案）随机采样替换
- **训练**：标准 OpenVLA / RT-2 BC，1 demo → 100 增广样本
- **任务**：真机 pick-and-place + drawer open

### 实验关键数据

| Setting | OpenVLA SR (new bg) | OpenVLA SR (训练 bg) |
|---|---|---|
| 50 demo, no aug | 18% | 87% |
| 50 demo + GreenAug (100 bg) | **76%** | 84% |
| 50 demo + RoboEngine (diffusion) | 78% | 85% |
| 50 demo + ROSIE | 73% | 82% |

→ **GreenAug 与精细 diffusion 增广效果几乎相同**，但实现成本低一个数量级。

### 与 [[2024_BYOVLA]] 的对照（B2 类必答）

| 维度 | GreenAug | BYOVLA |
|---|---|---|
| 时机 | 训练时增广 | 推理时干预 |
| 数据成本 | 需绿幕采集 | 用已有数据 |
| 推理开销 | 无 | 50.6× |
| 视觉合理性 | 拼接痕迹明显 | 看似自然 |
| 鲁棒性提升 | +58 pp on bg | +22 pp on bg |

→ **GreenAug 提升更大**，因为它真正训出鲁棒模型；BYOVLA 仅在推理时擦背景，VLA 本身不鲁棒。

### 能否与 RobustVLA 的 input-robust 模块结合（B2 类必答）

**可以且应当结合**：
- RobustVLA 的 input-robust 训练目标是 $\|\pi(\omega^*(o)) - \pi(o)\|^2$（保语义扰动一致性）
- GreenAug 提供的「换背景」就是天然的语义保留扰动 ω*
- **结合方案**：用 GreenAug 离线生成 (o, ω*(o)) 配对，喂给 RobustVLA input-robust loss
- **优势**：GreenAug 的随机背景比 RobustVLA 原文用的色彩抖动 / 模糊更接近真实部署分布

→ **这是非常有价值的实验**：GreenAug × RobustVLA = 训练时鲁棒 + 训练时增广叠加，且零推理开销。

### 与曦源关联

1. **曦源真机采集应直接用绿幕**：成本几百元（绿幕布 + 灯），换得 4× SR 提升，性价比无与伦比。
2. **「数据多样性 > 数据保真度」原则可推广**：曦源若做合成数据，不要追求 photorealism，先要求多样性。
3. **与 FocusVLA 协同**：FocusVLA 让 attention 聚焦任务区域，GreenAug 训出鲁棒 → 双路径鲁棒化。
4. **是 EfVLA 项目中"轻量增广"的首选**：与 ROSIE / RoboEngine 相比，GreenAug 不需要任何 generative model，节省训练资源。

### 待解问题

1. **GreenAug 对前景物体的依赖**：如果前景物体本身在不同光照下颜色变化大，色键失败。
2. **绿幕反射对前景物体的污染**：实际采集中绿色反光会污染前景，需要后处理。
3. **多视角扩展**：单目色键容易，多目同步色键有难度。

%% end my-thoughts %%

## 🔗 关联笔记

- **直接对比**：[[2024_BYOVLA]]（推理时清洗 vs 训练时增广）
- **可堆叠**：[[2026_RobustVLA_Guo]]（RobustVLA input-robust + GreenAug 数据）
- **同类增广扩展**：[[2024_ROSIE]] (diffusion inpainting)、[[2025_RoboEngine]] (semantic seg + bg gen)、[[2026_RoboVIP]] (multi-view gen)
- **同类「数据多样性 > 保真」哲学**：[[2018_DomainRandomization_Tobin]]
- **被增广对象**：[[2024_OpenVLA]]、[[2023_RT-2]]
- **评测平台**：[[2025_Eva-VLA]]、[[2026_VLA-Risk]]
