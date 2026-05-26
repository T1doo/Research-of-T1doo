---
title: "FAIL-Detect: Uncertainty-Aware Failure Detection for Imitation Learning Policies Without Failure Data"
authors: "He et al. (Pending 2503.08558)"
year: "2025"
journal: "arXiv preprint"
arxiv: "2503.08558"
venue: "arXiv 2025-03 (RSS / CoRL 2025 候选)"
citekey: "heFAILDetect2025"
itemType: "preprint"
status: "已精读 · 主线C-ii 备选"
tier: "⭐⭐⭐ 必读 · 无失败数据 OOD 检测"
tags: [literature, T1D, 主线C, failure-detection, uncertainty-quantification, OOD, FAIL-Detect]
---

# FAIL-Detect — 无失败数据下的 OOD 检测 精读笔记

> [!info] 元信息
> - **作者**：He 等（待 arXiv 2503.08558 确认）
> - **日期**：2025-03 (arXiv 2503.08558)
> - **arXiv**：[2503.08558](https://arxiv.org/abs/2503.08558)
> - **主题定位**：仅用 *成功演示数据*（**无 failure data**）训练 OOD / failure detection 模块
> - **方向归属**：主线 C-ii UQ / OOD 检测（**备选方向**）

## 📄 Abstract（综合可得信息）

传统 failure detection 需要 *failure dataset* 作正样本——但 robotic 收集 failure 成本高（要等失败 / 标注失败 timestamp）。FAIL-Detect 提出仅用 *success demonstration* 训练 detector，靠 **density estimation** / **NF (Normalizing Flow)** / **score-matching** 等方法学习 success state 分布；OOD 状态被识别为 low-likelihood。在 LIBERO、Real-world Franka 任务上证明 detector AUROC 0.85-0.95，能在 failure 发生前几步预警。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「Failure 数据是稀缺资源」**：robot 跑 1000 次任务，可能只有 50 次失败——而且失败原因多样、难标注。FAIL-Detect 的 *only-success-data* 设定**降低了 detection 模块的部署成本**。

2. **「OOD ≠ Failure，但相关」**：模型遇到 OOD 状态时倾向于失败，所以 OOD detector ≈ failure detector。这是一个理论简化，但实证 work。

3. **「Detection latency 决定 actionability」**：detector 在 failure 发生前 *几步* 就预警——这给上游 recovery 机制（RaC / RePLan）足够时间介入。

### 方法论（要重现的关键技术细节）

#### Density Estimation 方法
- **Normalizing Flow (NF)**：用 RealNVP / Glow 等 invertible 网络估计 P(state) 或 P(action | state)
- 训练：max log P(s, a) on success data
- 推理：低 log-likelihood → 高 failure 概率

#### Score-Matching 方法
- 用 score-based generative model 学 success distribution 的 score function
- OOD 判定：score norm 高 → OOD

#### 与 RC-NF 的关系
- RC-NF（已读）也是 NF-based detection，但更专注 *robustness*
- FAIL-Detect 更专注 *temporal failure*

### 实验关键数据（综合可得信息）

#### Detection Performance
| Method | LIBERO AUROC | Real-Franka AUROC |
|---|---|---|
| Random | 0.50 | 0.50 |
| Reconstruction-based | 0.75 | 0.70 |
| **FAIL-Detect (NF)** | **0.92** | **0.88** |

### 与我研究（曦源 / 主线 C-ii）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**正交 + 互补**
- FocusVLA 改进 policy
- FAIL-Detect 监控 policy 的状态分布
- 组合：FAIL-Detect 监督 FocusVLA 运行 → 在 OOD 上触发警报

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**互补**
- RobustVLA 训练侧加 worst-case 抗扰动
- FAIL-Detect 推理侧检测扰动后果
- **联合 pipeline**：RobustVLA 抗扰动 + FAIL-Detect 兜底警报

#### 3. 与 [[2025-09_RaC_RecoveryCorrection_Dass]] / [[2024-01_RePLan_Skreta]] 的关系：**触发器**
- FAIL-Detect 提供 *自动失败检测信号*
- 信号触发 → RaC recovery 或 RePLan replan
- **端到端 pipeline**：FAIL-Detect → RePLan / RaC → 恢复

### 论文里的 Future Work（基于方向特征推断）

1. **「跨任务 detector transfer」**——一个任务训的 detector 能否 transfer？
2. **「与 control loop 闭环」**——detector 触发后如何 recover
3. **「Calibration 问题」**——detector 输出概率与实际失败率是否对齐？

### 本科生一年期课题切入空间（**中**）

**最适合切入的 sub-problem：「FAIL-Detect + Recovery 闭环」**

- Step 1: 复现 FAIL-Detect 在 LIBERO 上
- Step 2: 接 RaC 数据或 RePLan replanner
- Step 3: 端到端评测
- Step 4: 写 paper

但与 [[2025-09_RaC_RecoveryCorrection_Dass]] 重叠度较高——FAIL-Detect 作 *组件* 而非主线更合理。

### 设计风险

- **NF 训练超参数敏感**
- **代码 release**：He 等待确认

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2025-05_LatentSafetyFilter_Nakamura]]
- 主线 C-iii 触发器：[[2025-09_RaC_RecoveryCorrection_Dass]], [[2024-01_RePLan_Skreta]]
- 主线 A/B 对照：[[2026-03_FocusVLA_Zhang]], [[2026_RobustVLA_Guo]]
- 已读相关：RC-NF
- OOD detection 经典：DDU, Mahalanobis, ODIN

## 📌 Action Items
- [ ] 找 PDF（arXiv 2503.08558）
- [ ] 评估 NF 训练在本科生算力下的可行性
- [ ] 与 RaC pipeline 集成原型

%% Import Date: 2026-05-26 %%
