---
title: "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"
authors: "Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone"
year: "2023"
venue: "NeurIPS 2023 Datasets & Benchmarks Track"
arxiv: "2306.03310"
status: "已精读"
tier: "⭐⭐⭐ 谱系基石"
tags: [literature, T1D, benchmark, robustness-eval, LIBERO-family, lifelong-learning]
---

# LIBERO (Original) — 谱系基石

> [!info] 元信息
> - **作者**：Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone（UT Austin · Sony AI）
> - **arXiv**：[2306.03310](https://arxiv.org/abs/2306.03310)
> - **官方网站 / GitHub**：[libero-project.github.io](https://libero-project.github.io) · [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
> - **数据集规模**：4 suites × 130 tasks，每任务 50 条人工遥操作演示
> - **rolling leaderboard**：无（学术 benchmark）

## 📄 这个 benchmark 是做什么的

LIBERO（**LI**felong learning **BE**nchmark on **RO**bot manipulation）是 lifelong-learning for decision-making (LLDM) 的标杆 benchmark，研究知识迁移效率（declarative / procedural / mixed）、策略架构设计、算法效果、任务顺序鲁棒性以及预训练影响。后续 LIBERO-Plus / PRO / X / Para 全部以它为基础。

四个 suite：
- **LIBERO-Spatial**：10 tasks——同物体不同空间关系
- **LIBERO-Object**：10 tasks——同空间不同物体
- **LIBERO-Goal**：10 tasks——同环境不同目标
- **LIBERO-Long (LIBERO-100)**：100 tasks 长程组合任务

共 130 tasks × 50 demos = 6500 条人工示范，基于 robosuite + MuJoCo。

## 🧠 我的评估

### 评测维度
- 主要是 **任务级泛化**（任务序列、知识结构），原版**不含视觉/语言扰动**
- 与 VLA-Risk 几乎正交：VLA-Risk 测扰动鲁棒，LIBERO 测任务泛化
- 是「clean baseline」——任何 robustness benchmark 都拿它作 SR-original 锚点

### 任务规模
- 130 tasks / 6500 demos
- Spatial / Object / Goal 各 10 个 short-horizon；Long 100 个 long-horizon
- 无显式难度分级（但 Long suite 是公认最难）

### 评估 baseline 模型
- 原文：ResNet-RNN、ResNet-T、ViT-T 三种策略架构
- 后续社区评测：OpenVLA（多权重 ~95%）、π0（>98%）、FocusVLA（98.7%）、Octo、RDT、CogACT、SmolVLA 等几乎所有主流 VLA

### 关键 finding
- 顺序微调（sequential finetuning）反而**优于** EWC / ER 等 lifelong-learning 方法
- 视觉编码器架构对 LLDM 性能不一致（ViT 不总赢）
- **naive supervised pretraining 反而损害下游 LLDM 性能**（反直觉发现）
- 主流 VLA 在 clean LIBERO 上 SR > 95%，**已经饱和**——这正是 LIBERO-Plus/PRO 出现的原因

### 复现成本
- **GPU**：训练 OpenVLA 全权重需 8×A100；推理评测仅需 1×24GB
- **仿真器**：robosuite + MuJoCo（开源）
- **评测时间**：全 130 tasks × 50 episodes ≈ 6-12 h on 1 GPU
- **开源**：✅ MIT License，数据集 HuggingFace 可下

### 是否适合曦源课题
- ✅ **必备 baseline**（不是主测试场，但任何 robustness 工作都得提供 LIBERO-clean SR）
- 显存友好（46 GB 足够推理任何主流 VLA）
- 但**单跑 LIBERO 已不够**——成绩饱和，无区分度。**必须搭配 LIBERO-Plus 或 LIBERO-PRO**

## 🔗 关联
- [[VLA-Risk_ICLR2026]]：使用 LIBERO 的 Direct Manipulation 子集
- [[2025-10_LIBERO-Plus]]：在原版上加 7 维扰动
- [[2025-10_LIBERO-PRO]]：在原版上加 5 维"PRO"组合扰动
- [[2026-02_LIBERO-X]]：分层 capability 评测扩展
- [[2026-03_LIBERO-Para]]：语言改写专项扩展
