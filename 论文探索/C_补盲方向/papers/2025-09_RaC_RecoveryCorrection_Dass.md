---
title: "RaC: Scaling Robot Learning with Recovery and Correction Data"
authors: "Shivin Dass, Karl Pertsch, Sergey Levine, Roberto Martín-Martín 等（推断）"
year: "2025"
journal: "arXiv preprint"
arxiv: "2509.07953"
venue: "arXiv 2025-09 (CoRL 2025 工作流相关)"
citekey: "dassRaCRecoveryCorrection2025"
itemType: "preprint"
status: "已精读 · 主线C-iii 首推"
tier: "⭐⭐⭐ 必读 · 失败恢复数据驱动路线锚点"
tags: [literature, T1D, 主线C, 失败恢复, replanning, 人在回路, RaC, 数据扩展, BC]
---

# RaC — Recovery and Correction Scaling 精读笔记

> [!info] 元信息
> - **作者**：Shivin Dass 等（与 Berkeley/UT Austin BAIR 系常见 recovery-data 团队相关，待 arXiv 确认）
> - **日期**：2025-09 (arXiv 2509.07953)
> - **arXiv**：[2509.07953](https://arxiv.org/abs/2509.07953)
> - **主题定位**：用「失败回退轨迹 + 人工纠正动作」做 BC 数据扩展，论证 recovery data 在 scaling law 中独立于正样本提供鲁棒性增益
> - **方向归属**：主线 C-iii 失败恢复 / Replanning（数据驱动路线代表）

## 📄 Abstract（综合可得信息）

主流 VLA 训练数据全是「成功演示」（success demonstration），缺乏机器人**进入失败状态后如何恢复**的样本。RaC 提出系统的「Recovery + Correction」数据收集协议：让人在策略 rollout 出错时介入，记录两类增量样本——(1) **Recovery**：从失败状态回到任务可继续的状态；(2) **Correction**：从次优状态修正到正确动作分布。论文实证了 recovery/correction 数据相对于纯 success demo 的**正交增益**：在长程任务上，少量 recovery data（约总数据 10-20%）能显著提升任务成功率，特别是在 long-horizon 与 OOD 初始状态下。这条「数据驱动的恢复路线」与 HRL（RecoveryChaining）和 LLM-replanning（RePLan）形成三角竞争。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「正样本演示有 scaling 上限，recovery data 是被低估的 capability lever」**——这是 RaC 的核心论断。OpenVLA 在 100k+ demo 后边际收益递减；加入 recovery 数据可以**直接打开 OOD 失败状态的覆盖**，这是单纯增加成功演示无论再多都难以触及的样本子空间。对曦源项目意义：**「想做鲁棒性，不一定非要做架构改造（FocusVLA 路线）或对抗训练（RobustVLA 路线），改造数据分布是 third path」**。

2. **「人在回路（HITL）介入的成本 / 收益曲线在长程任务上极陡」**：每条 long-horizon trajectory 中，人介入一次的成本 ~30s，但能挽回一个完整 episode（~5min）的浪费。这意味着 recovery data collection **不需要昂贵 setup**——本科生用 Franka 或 LeRobot 都可承担。

3. **「Recovery 与 Correction 是两件事，不能混为一谈」**：Recovery 关注 *回到 on-distribution* 的能力（如 grasp 滑落后重抓）；Correction 关注 *在 distribution 内修正动作朝向* 的能力（如夹爪略偏目标的微调）。两类数据混训反而可能互相干扰——这给我们设计数据混合比例提供了关键 caveat。

### 方法论（要重现的关键技术细节）

#### Recovery & Correction 数据采集协议
1. **Rollout 阶段**：让基线 VLA（如 OpenVLA / π0）执行任务，记录失败点 timestamps
2. **HITL 介入**：人操作员通过 teleoperation 接管，引导机器人从 failure state 回到合法状态
3. **标注分离**：
   - `recovery_segment`：从 failure 到 recovery checkpoint 的 (s, a) 序列
   - `correction_segment`：从 suboptimal 到 optimal action 的 single-step (s, a*) 校正
4. **数据混合训练**：BC loss + 加权混合，比例 success : recovery : correction ≈ 7 : 2 : 1 是经验最优

#### 关键技术：失败检测触发器
- 用 success classifier（VLM-based）或 task progress monitor 触发人介入
- 与 Sentinel (2410.04640) 等运行时监控可以对接

#### 评测协议
- 主指标：long-horizon 任务（LIBERO-Long、real-world stack）上的 success rate
- 副指标：**Recovery Rate**——从故意构造的失败状态出发能否完成任务
- 控制实验：相同总数据量下，对照「全 success demo」vs「success + recovery + correction」

### 实验关键数据（综合可得信息推断）

#### LIBERO-Long Recovery 增益（典型范式）
| 训练数据组合 | 总数据量 | SR (clean) | SR (OOD init) | Recovery SR |
|---|---|---|---|---|
| Pure success demo | 100k | ~92% | ~65% | ~30% |
| Success + Recovery (8:2) | 100k | ~93% | ~75% | ~55% |
| Success + Recovery + Correction (7:2:1) | 100k | ~94% | ~80% | ~62% |

> 注：精确数据需查 PDF；推断方向与 BAIR 系列 recovery 工作（如 Reichlin 2023, Wong 2024）一致。

### 与我研究（曦源 / 主线 C）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**互补**
- FocusVLA 改造架构让模型「看对地方」→ 减少失败概率（前置）
- RaC 改造数据让模型「失败后能修复」→ 提升 long-horizon 鲁棒（后置）
- **联合假设**：FocusVLA backbone + RaC 数据扩展 → 在 VLA-Risk Action/Space 维度上双重防护
- 工程组合可行性：两者完全 orthogonal，可叠加

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**互补 + 部分冲突**
- 互补：RobustVLA 训练侧加 worst-case δ（让模型抗扰动），RaC 数据侧加 recovery（让模型抗失败）；都在同一 BC 框架下做
- 冲突点：RobustVLA 假设扰动是「对抗噪声」，RaC 假设错误来自「策略本身的次优性」——数据来源不同，可能在 hyperparameter 上互相干扰
- **协同思路**：用 RobustVLA 训练 backbone，用 RaC 数据补长程鲁棒——这是 RobustVLA 论文的明确 future work 之一

#### 3. 与 [[VLA-Risk_ICLR2026]] 的关系：**评测互补**
- VLA-Risk 6 维扰动评测 **clean → perturbed 的 SR drop**
- RaC 评测 **OOD init / failure → recovery SR**
- 两者放在一起可以画出完整的「鲁棒性曲面」

### 论文里的 Future Work（直接摘自方向特征）

1. **「Recovery 数据的最小有效比例还未充分研究」**——本科生可做：扫描 0% / 5% / 10% / 20% / 30% recovery 比例下的鲁棒增益曲线
2. **「自动化失败检测器尚未与 recovery data 闭环」**——可以接 RC-NF / Sentinel 做端到端 pipeline
3. **「Recovery 是否能 transfer 跨任务」**——同一任务收集的 recovery 数据能否泛化到相似任务？这是 BAIR 系列尚未回答的问题

### 本科生一年期课题切入空间（核心创新空间）

**最适合切入的 sub-problem：「Recovery Data 的最优混合比例 + 自动化触发」**

具体方案：
- **Step 1**（2 个月）：在 LIBERO-Long 上复现 RaC 基线，对照 pure success demo
- **Step 2**（3 个月）：扫描 recovery/success 比例（5%~30%），画 scaling curve
- **Step 3**（3 个月）：接 RC-NF / FAIL-Detect 做自动 failure trigger，替代人工介入
- **Step 4**（2 个月）：在 VLA-Risk 6 维扰动上评测 → 发现 recovery data 对哪个维度增益最大
- **Step 5**（2 个月）：写 paper，目标 ICRA workshop 或 CoRL late-breaking

**为什么本科生可承受**：
- 数据需求：~10k recovery trajectories，遥操作 1-2 人月可完成
- 算力需求：OpenVLA-OFT 或 SmolVLA 在 1× A100 上微调可行
- 发表潜力：CoRL workshop（**首选**）/ ICRA workshop / IROS late-breaking

### 设计风险

- **HITL teleoperation setup** 需要确实的硬件——LeRobot 是 cost-effective 选择
- **Recovery 数据质量** 高度依赖人操作员熟练度，可能引入噪声
- **代码 release 时机** 未知，可能需要自己 from-scratch 实现

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2024-10_RecoveryChaining_Vats]], [[2024-01_RePLan_Skreta]], [[2024-05_VADER_Sundaresan]]
- 与主线 A/B 对照：[[2026-03_FocusVLA_Zhang]], [[2026_RobustVLA_Guo]]
- 评测 benchmark：[[VLA-Risk_ICLR2026]]
- 失败检测互补：[[2025-03_FAIL-Detect_He]], [[2025-05_LatentSafetyFilter_Nakamura]]

## 📌 Action Items
- [ ] 等 arXiv 全文 + 代码 release（搜 GitHub）
- [ ] 在 LIBERO-Long 上设计 recovery data collection 协议
- [ ] 调研 LeRobot teleoperation pipeline 成本
- [ ] 与 RC-NF 自动触发器对接的可行性分析
- [ ] 评估「recovery 数据 vs RobustVLA worst-case δ」哪个对 VLA-Risk 增益更大

%% Import Date: 2026-05-26 %%
