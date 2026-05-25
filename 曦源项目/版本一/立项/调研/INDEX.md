---
create time: 2026-05-21
tags:
  - 曦源项目
  - 立项调研
  - INDEX
---

# Efficient VLA 立项调研 · 索引

> [!info] 总览
> 本目录是曦源项目 *Efficient VLA* 立项的完整调研产出，覆盖 3 个研究方向 + 6 个 baseline + 4 个 benchmark，含 6 份综合分析报告 + 9 篇 References 笔记。建议阅读顺序：先看 [[00_总览与立项建议]] 了解推荐主线，再按需展开各方向 / baseline / benchmark 细节。

## 综合分析报告（本目录下）

| 文件 | 主题 | 字数 |
|------|------|------|
| [[00_总览与立项建议]] | 三方向对比 + 推荐主线 + 时间表 + 风险 | ~3500 字 |
| [[01_方向一_动态自适应推理]] | AAC + SmolVLA + A2A 深度解析 + 切入点 | ~4500 字 |
| [[02_方向二_轻量级异常检测]] | RC-NF + π0 + π0.5 深度解析 + 切入点 | ~6500 字 |
| [[03_方向三_鲁棒性微调]] | RobustVLA + ACT + DP 深度解析 + 切入点 | ~6500 字 |
| [[04_Baseline模型对比]] | 6 个 baseline 横向比较 + 选型建议 | ~2500 字 |
| [[05_Benchmark评测体系]] | LIBERO / LIBERO-Plus / CALVIN / RoboTwin + Allen AI Leaderboard | ~2500 字 |

## References 笔记（含 abstract + 「我的思考」）

### 三方向核心论文

- [[liangAdaptiveActionChunking2026]] — AAC: Adaptive Action Chunking at Inference-time（方向一核心，arxiv 2604.04161）
- [[zhouRCNFRobotConditionedNormalizing2026]] — RC-NF: Robot-Conditioned Normalizing Flow（方向二核心，arxiv 2603.11106）
- [[guoRobustnessVisionLanguageActionModel2026]] — RobustVLA: On Robustness against Multi-Modal Perturbations（方向三核心，arxiv 2510.00037）

### 经典 VLA Baseline

- [[black$p_0$VisionLanguageActionFlow2026]] — π0: Vision-Language-Action Flow Model（3.3 B，RSS 2025，arxiv 2410.24164）
- [[intelligence$p_05$VisionLanguageActionModel2025]] — π0.5: Open-World Generalization（3.3 B，CoRL 2025，arxiv 2504.16054）

### 轻量化 VLA Baseline

- [[shukorSmolVLAVisionLanguageActionModel2025]] — SmolVLA: Affordable & Efficient（0.45 B，arxiv 2506.01844）⭐ **推荐主骨干**

### 非 VLA Baseline

- [[zhaoLearningFineGrainedBimanual2023]] — ACT: Action Chunking with Transformers（80 M，RSS 2023，arxiv 2304.13705）
- [[chiDiffusionPolicyVisuomotor2024]] — DP: Diffusion Policy（RSS 2023，arxiv 2303.04137）
- [[jiaActiontoActionFlowMatching2026]] — A2A: Action-to-Action Flow Matching（RSS 2026，arxiv 2602.07322）

## Benchmark（无论文，详见 [[05_Benchmark评测体系]]）

- **LIBERO** — 4 套件 130 任务，单臂 lifelong learning（arxiv 2306.03310）
- **LIBERO-Plus** — LIBERO 的鲁棒性扰动版本（具体地址待立项后落实）
- **CALVIN** — 长程 language-conditioned，4 环境 34 任务（arxiv 2112.03227）
- **RoboTwin 2.0** — 双臂协同，50+ 任务（CVPR 2025 Highlight，arxiv 2504.13059）
- **Allen AI VLA Eval Harness** — 18 benchmark × 1,885 model 聚合 leaderboard（v0.2.0, 2026-05）

## 推荐阅读路径

### 我是首次接触本立项的人
1. [[00_总览与立项建议]]（5 分钟）
2. [[04_Baseline模型对比]] 第 1 节主表（2 分钟）
3. [[05_Benchmark评测体系]] 第 1 节概览（2 分钟）
4. 选感兴趣的方向报告深读

### 我是导师 / 评审委员会
1. [[00_总览与立项建议]] 第 7 节（评审 Q&A）
2. [[00_总览与立项建议]] 第 2 节（推荐主线）
3. [[00_总览与立项建议]] 第 4 节（时间表）

### 我准备启动立项
1. [[04_Baseline模型对比]] 第 4 节（启动配置）
2. [[05_Benchmark评测体系]] 第 4 节（评测基础设施 checklist）
3. [[00_总览与立项建议]] 第 4 节 Phase 0（启动任务）
4. 主线方向报告的「第 5 节切入点」

### 我想找文献综述
- 所有 References/*.md 的「核心观点」「方法论」段都有简洁的论文 takeaway，可作为浏览快照
- 三方向综合报告的「第 2 节 核心论文深度解析」是更详细的版本

## 原始素材

- **立项方向定义**：[../立项方向](../立项方向.md)
- **Zotero 导入的论文 PDF**：见每篇 References 的 zotero-link 字段
- **PDF 高亮与图片**：`E:\仓库\Research-of-T1doo\References\images\`

## 调研元信息

- 调研日期：2026-05-21
- 数据来源：abstract（9 篇 Zotero 导入）+ arxiv HTML 全文 WebFetch（12 篇核心 + 4 个 benchmark）+ GitHub README（Allen AI Harness、LIBERO 等）
- 执行方式：3 个 general-purpose agent 按方向分组并行 + 主线程综合
- 总字数：综合报告约 ~25,000 字（含表格、公式、wikilink），References 笔记总计约 ~30,000 字
