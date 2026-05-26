---
title: "LIBERO-PRO: Towards Comprehensive Robustness Evaluation of Vision-Language-Action Models"
authors: "Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, Lichao Sun"
year: "2025"
venue: "arXiv preprint (2025-10)"
arxiv: "2510.03827"
status: "已精读"
tier: "⭐⭐⭐ 推荐曦源 TOP-2 候选"
tags: [literature, T1D, benchmark, robustness-eval, LIBERO-family, generalization]
---

# LIBERO-PRO — 5 维「组合扰动」让 SR 从 90% 崩到 0%

> [!info] 元信息
> - **作者**：Xueyang Zhou 等 8 人（华中科大 · Lehigh）
> - **arXiv**：[2510.03827](https://arxiv.org/abs/2510.03827)
> - **GitHub**：[Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)
> - **数据集规模**：原 LIBERO × 5 维扰动 × 50 episodes/task
> - **rolling leaderboard**：GitHub README 维护

## 📄 这个 benchmark 是做什么的

「PRO」隐含 *PRObust* / *PROfessional* 含义，专测 VLA 是否真懂任务而非死记。在 LIBERO 上加 5 维组合扰动，**插件式（plug-and-play）**，下载扰动文件即可在原 LIBERO 评测 pipeline 上跑。

五个泛化维度：
1. **Object** — 目标物外观/类别替换
2. **Position** — 物体空间重定位
3. **Semantic** — 语言改写（paraphrase）
4. **Task** — 任务逻辑改变（动作顺序/组合）
5. **Environment** — 跨环境（不同 suite 之间）

## 🧠 我的评估

### 评测维度（关键！）
- **5 维 + 强调组合**（不是单独，而是叠加）
- 与 LIBERO-Plus 对比：
  - Plus 侧重「单 dim 退化曲线」（diagnostic 风格）
  - PRO 侧重「组合扰动下能否归零」（stress-test 风格）
- 与 VLA-Risk 对比：
  - VLA-Risk 用 GPT-5 生成对抗扰动，PRO 用规则化扰动文件
  - PRO 复现门槛 < VLA-Risk

### 任务规模
- 沿用 LIBERO 130 tasks × 50 evaluation episodes × 5 dim
- 难度暗含：单 dim 已让 SR 暴跌；组合 dim 直接归零

### 评估 baseline 模型
- **OpenVLA**（4 个官方 checkpoint，每 suite 一个）
- **π0**（1 个官方 checkpoint）
- **π0.5**（1 个官方 checkpoint）

仅 3 个家族，**比 Plus 少**，但都是顶级 SOTA。

### 关键 finding
- **OpenVLA & π0：position 扰动下 SR 0.98 → 0.00**——彻底崩溃
- **π0.5 略有救**：0.38（仍很差）
- 结论：「主流 VLA 是在死记 (env layout, action sequence) 配对」
- 失败模式：目标换了仍去抓原物体；指令组合后无法理解

### 复现成本
- **环境**：Python 3.8.13 + PyTorch 1.11.0（**老依赖**，需新建 conda 环境）
- **GPU**：推理需求与原 LIBERO 相同（24GB 足够 OpenVLA 7B）
- **快速并行评测**：支持 BDDL 并行
- **评测时间**：5 dim × 130 task × 50 ep ≈ 32,500 episodes，约 1-2 天/GPU；子集化 1 dim 约 6 h
- **开源**：✅ MIT，HuggingFace 提供全部扰动文件

### 是否适合曦源课题
- ⚠️✅ **推荐 TOP-2/3 候选**（与 SimplerEnv / VLA-Arena 二选一）
- 优点：
  1. **故事最爆炸**：「90%→0%」一句话就是论文摘要 hook
  2. plug-and-play，与已有 LIBERO 评测代码 100% 兼容
  3. 显存友好（46GB 全够）
- 缺点：
  1. 仅 3 个 baseline（OpenVLA / π0 / π0.5），baseline 表不够丰满
  2. 老 PyTorch 1.11 依赖**可能与新 VLA 代码冲突**——需要测试
  3. 与 LIBERO-Plus 有显著重叠（都基于 LIBERO 加扰动），择一即可
- **决策建议**：若曦源故事走「stress-test 范式」走 PRO；若走「细颗粒度分析」走 Plus

## 🔗 关联
- [[VLA-Risk_ICLR2026]]：方法论上互补——VLA-Risk 用 LM 对抗生成，PRO 用规则
- [[2023-06_LIBERO_Original]]：基础
- [[2025-10_LIBERO-Plus]]：同月 sibling，谁强谁弱要看实测
- [[2026-03_LIBERO-Para]]：PRO 的 Semantic 维度被 Para 进一步细化
