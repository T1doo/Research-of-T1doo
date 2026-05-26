---
title: "LIBERO-X: Hierarchical Capability Benchmark for Vision-Language-Action Models"
authors: "Guodong Wang, Chenkai Zhang, Qingjie Liu, Jinjin Zhang, Jiancheng Cai, Junjie Liu, Xinmin Liu"
year: "2026"
venue: "arXiv preprint (2026-02)"
arxiv: "2602.06556"
status: "已精读"
tier: "⭐⭐ 分层 capability"
tags: [literature, T1D, benchmark, robustness-eval, LIBERO-family, hierarchical]
---

# LIBERO-X — 3 维 capability × 分层难度

> [!info] 元信息
> - **作者**：Guodong Wang 等 7 人
> - **arXiv**：[2602.06556](https://arxiv.org/abs/2602.06556)
> - **核心定位**：把鲁棒性分解为 **3 项核心能力 × 累积难度**

## 📄 这个 benchmark 是做什么的

LIBERO-X 不再笼统测「鲁棒性」，而是把它**结构化拆解**为 3 项核心能力，每项有累积难度（progressive levels），让 fine-grain 诊断 VLA 在哪一层崩。

三项核心能力：
1. **Spatial generalization** — 空间泛化（物体位置/朝向变化）
2. **Object recognition** — 物体识别（外观/类别变化）
3. **Task instruction understanding** — 任务指令理解

每能力沿 hierarchical levels 累积扰动 → 揭示 capability 边界。

## 🧠 我的评估

### 评测维度
- **3 capability × 多 level**（论文未明确说几 level）
- 与 LIBERO-Plus / PRO 对比：
  - Plus 用 dimension 切分扰动，PRO 测组合崩溃
  - X 用 **capability 切分**——更接近"诊断式"评测
- 与 VLA-Risk 关系：VLA-Risk 的 (object/action/space) × (vision/instr) 矩阵其实包含 X 的三项；但 X 强调分层难度，VLA-Risk 强调攻击对抗

### 任务规模
- 论文未明确说总 task 数
- 用人工遥操作收集的多样训练数据「弥合训练-评测分布 gap」（这点不同于 Plus/PRO 直接复用原 LIBERO）

### 评估 baseline 模型
- 论文表述「representative VLA models」，未具名清单
- 应包含 OpenVLA / π0 等主流

### 关键 finding
- VLA 在累积扰动下「显著退化」
- **持续暴露在 scene comprehension 和 instruction grounding 上的不足**
- 揭示 train-eval distribution gap 即使数据多样也仍存

### 复现成本
- 论文是 2026-02 新出，**代码情况未知**（未明确 GitHub 链接）
- 需自行采集新数据 → **门槛比 Plus/PRO 高**
- 显存需求未提，估计与 LIBERO 同级

### 是否适合曦源课题
- ⚠️ **暂不推荐为主测试场**
- 理由：
  1. **开源不确定**——2026-02 新论文，代码未必 release
  2. baseline 信息不充分——很难直接对标
  3. 分层 capability 设计对小课题而言**颗粒度过细**——可能写不完
- **建议**：作为「辅助分析角度」——若 Plus/PRO 评测结果想拆解到 capability 层面，可参考 LIBERO-X 的分类法

## 🔗 关联
- [[2023-06_LIBERO_Original]]：基础
- [[2025-10_LIBERO-Plus]] / [[2025-10_LIBERO-PRO]]：维度切分 vs capability 切分
- [[VLA-Risk_ICLR2026]]：VLA-Risk 矩阵的另一种切分方式
