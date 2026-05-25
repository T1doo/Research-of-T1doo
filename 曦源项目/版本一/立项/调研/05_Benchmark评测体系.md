---
create time: 2026-05-21
tags:
  - 曦源项目
  - 立项调研
  - benchmark
---

# Benchmark 评测体系横评

> [!info] 立项导向
> Efficient VLA 立项的三个方向都必须在公开 benchmark 上验证。本节横评 LIBERO / LIBERO-Plus / CALVIN / RoboTwin 四个仿真平台 + Allen AI VLA Evaluation Harness 一个聚合 leaderboard，给出"每方向应跑哪个、为什么"的明确选型建议。结论：**LIBERO 是最便宜的入门 + 三方向 baseline 对齐 benchmark；LIBERO-Plus 是方向三鲁棒性的核心评测；CALVIN 适合方向一长程任务；RoboTwin 是双臂场景的差异化亮点**。

## 1. 概览对比表

| 平台 | 任务数 | 类别 | 仿真器 | SOTA 现状 | 评测协议 | 与本项目方向匹配 |
|------|-------|------|--------|----------|----------|---------------|
| **LIBERO** | 130（4 套件） | 单臂、lifelong learning | robosuite (MuJoCo) | π0/SmolVLA/AAC 均报告 85-95%+ | 4 套件分别评测 + 跨套件 transfer | 全方向 baseline 通用，**必跑** |
| **LIBERO-Plus** | LIBERO 派生 | 鲁棒性扰动版 | 同 LIBERO | RobustVLA 等开始报告 | 注入视觉/物体扰动后看 SR 跌幅 | **方向三核心**、方向一鲁棒性差异化 |
| **CALVIN** | 34 task × 4 环境 (A/B/C/D) | 长程语言-条件 | PyBullet | baseline imitation 表现不佳 | D→D, ABC→D, ABCD→D 三个 split，5-step chain SR | **方向一长程**、方向二任务边界 OOD |
| **RoboTwin** | 50+（2.0 版） | 双臂协同 | 自研仿真 + 真机 COBOT Magic | 70%+ 单臂 / 40%+ 双臂 vs real-only | 自动数据生成 + 仿真到实机 | **双臂差异化亮点**，三方向均可扩展 |
| **Allen AI VLA Eval Harness** | 18 benchmark | 聚合 leaderboard | 多仿真器 | v0.2.0 (2026-05) 已聚合 1,885 model × 18 benchmark | 47× throughput (2000 episode/18 min @ H100) | 不直接评测，提供**模型对照与 SOTA 数据** |

## 2. 逐 benchmark 详解

### 2.1 LIBERO（必跑，三方向通用 baseline）

- **来源**：Lifelong-Robot-Learning 团队，2023，arxiv 2306.03310。
- **4 个任务套件**：
  - **LIBERO-Spatial**（10 任务）：同物体不同空间布局，考察空间泛化；
  - **LIBERO-Object**（10 任务）：同布局不同物体；
  - **LIBERO-Goal**（10 任务）：同物体同布局，目标变化；
  - **LIBERO-Long / LIBERO-10**（10 任务）：长程组合任务；
  - **LIBERO-100 = LIBERO-90 (pretrain) + LIBERO-10 (test)**。
- **评测协议**：默认在训练时 on-the-fly 评测，可单独跑评测脚本；按 lifelong learning 顺序 fine-tune，关注 forward / backward transfer。
- **SOTA 现状**（截至 2026-05）：
  - SmolVLA 平均 87.3%（Spatial 90 / Object 96 / Goal 92 / Long **71**）；
  - π0 论文未报告完整 LIBERO 数字，但 openpi 复现普遍 85%+；
  - AAC + GR00T 平均 95.0%（Spatial 94.4 / Object 98.6 / Goal 94.2 / Long 92.8）。
- **依赖**：torch 1.11 + cu113 + robosuite（具体版本看 requirements.txt）。GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO 。
- **本项目用途**：所有方向的复现起点。**LIBERO-Long 是最有差异化提升空间的子套件**（SmolVLA 仅 71%）—— 三方向技术叠加上去的提升应优先在 Long 上量化。

### 2.2 LIBERO-Plus（方向三核心、方向一差异化）

- **定位**：在 LIBERO 基础上注入鲁棒性扰动（视觉/物体/光照/相机抖动等）。Allen AI VLA Harness 同时列出 LIBERO-Plus 与 LIBERO-Pro / LIBERO-Mem，说明社区已开始把 LIBERO 派生成多个鲁棒性 / 记忆性 benchmark 集群。
- **本项目用途**：
  - **方向三 RobustVLA 复现的天然对位**——把 RobustVLA 在 LIBERO 上 17 扰动的实验直接迁移到 LIBERO-Plus 的扰动协议；
  - **方向一 AAC 的鲁棒性差异化点**——AAC 原文未在 LIBERO-Plus 评测，可填补实证空白（参考方向一报告切入点 C）；
  - **方向二 RC-NF 的误报率验证**——LIBERO-Plus 的合理扰动应被 RC-NF 判为"正常"，否则就是误报。
- **关键风险**：LIBERO-Plus 的官方 paper / 项目页尚未在本次调研中拉到（arxiv 2505.18564 不是它），需要立项后第一周通过 Allen AI 仓库找到具体地址。

### 2.3 CALVIN（方向一长程评测）

- **来源**：U. Freiburg，2021，arxiv 2112.03227。
- **任务**：long-horizon language-conditioned manipulation，4 个环境（A/B/C/D），34 个原子 task，链式拼接成 5-step instruction chain。
- **三个 split**：
  - **D → D**：同环境训练评测，最容易；
  - **ABC → D**：3 环境训 1 环境测，跨环境泛化；
  - **ABCD → D**（zero-shot 语言遵从）：全环境训，测新指令。
- **关键指标**：5-step success rate（连续 5 个子任务全部成功的比例），是 long-horizon 评测的金标准。
- **SOTA 现状**：原始论文 baseline multi-context imitation learning 表现不佳——这意味着 **VLA 方法在 CALVIN 上有显著上升空间**。
- **本项目用途**：
  - **方向一 AAC**：长程任务中 chunk size 决策更敏感，AAC 的 entropy 触发理论上更有用。AAC 原文未在 CALVIN 评测，是新颖点；
  - **方向二 RC-NF**：5-step chain 的子任务边界是 OOD 高发期，RC-NF 应在边界处给出更高的异常分；
  - **方向三 RobustVLA**：把 17 扰动在 5-step chain 上看复合衰减，回答"鲁棒性增益是否随 horizon 衰减"。

### 2.4 RoboTwin（双臂差异化亮点）

- **来源**：CVPR 2025 Highlight，2.0 版本 2025 发布，arxiv 2504.13059。GitHub: https://github.com/robotwin-Platform/RoboTwin 。
- **特色**：
  - **50+ 双臂协作 manipulation tasks**；
  - **自动化数据收集**：用 3D 生成模型 + LLM 生成 expert demonstration（"spatial relation-aware code generation"），从单张 2D 图生成 interactive digital twin；
  - 仿真 + 真机（COBOT Magic Robot）双轨。
- **SOTA**：与 real-only 训练相比，RoboTwin 数据带来**单臂 +70%、双臂 +40%** 成功率提升——这是 sim-to-real 数据合成的重要支撑。
- **本项目用途**：
  - **方向一切入点 D**：AAC 在双臂场景下"独立熵自适应"，左右臂各自决策 chunk 截断；
  - **方向二切入点 C**：RC-NF 双臂异常（左右臂相对位置偏差、力度不匹配）目前文献空白；
  - **方向三切入点 C**：双臂 action 扰动是 RobustVLA 没碰过的新模态。

### 2.5 Allen AI VLA Evaluation Harness（参考 leaderboard，不直接评测）

- **地址**：https://allenai.github.io/vla-evaluation-harness/leaderboard/ + GitHub https://github.com/allenai/vla-evaluation-harness 。
- **规模**：v0.2.0（2026-05-09 release）聚合 **18 个 benchmark × 1,885 个 model 评测结果**。
- **覆盖的 benchmark**：LIBERO, SimplerEnv, CALVIN, ManiSkill2, **LIBERO-Pro, LIBERO-Plus**, RoboCasa, VLABench, MIKASA-Robo, **RoboTwin**, RLBench, RoboCerebra, LIBERO-Mem, BEHAVIOR-1K, Kinetix, RoboMME, MolmoSpaces-Bench, FurnitureBench。
- **支持的官方模型**：OpenVLA, π0, π0-FAST, GR00T N1.6, OFT, X-VLA, CogACT, RTC, VLANeXt；社区模型还有 DB-CogACT、QwenGR00T 等 starVLA 系列。
- **吞吐**：47× throughput（2000 LIBERO episodes 在单卡 H100 上 18 分钟跑完）。
- **本项目用途**：
  - 立项后直接接入 harness，省去搭评测脚本；
  - 提供 **SmolVLA / π0 / OpenVLA 的 SOTA 对照基线**，写论文时引用即可；
  - 注意：当前 leaderboard 是 JS 渲染、调研时拉不到表格内容，需要立项后访问 GitHub 仓库 raw 数据。

## 3. 选型建议（按方向）

| 方向 | 首跑 | 次跑 | 终跑（如时间允许） |
|------|------|------|------------------|
| 方向一（自适应推理） | **LIBERO**（对齐 AAC 95.0% / SmolVLA 87.3%） | **CALVIN**（长程下 chunk 决策） | **RoboTwin**（双臂独立熵） |
| 方向二（轻量异常检测） | **LIBERO-Anomaly-10**（RC-NF 原 benchmark） | **LIBERO-Plus**（验证误报率） | **RoboTwin**（双臂异常） |
| 方向三（鲁棒性微调） | **LIBERO**（17 扰动复现 RobustVLA） | **LIBERO-Plus**（鲁棒性 benchmark） | **CALVIN**（长程下扰动累积） |

## 4. 评测基础设施 checklist（立项第一阶段）

1. 安装 LIBERO + robosuite + MuJoCo（**预计 1-2 天**）；
2. 接入 Allen AI VLA Eval Harness 跑通 SmolVLA baseline（**1 周**）；
3. 拉到 LIBERO-Plus 官方代码（需要邮件作者或 GitHub issue）；
4. CALVIN / RoboTwin 二阶段引入，前者依赖 PyBullet、后者依赖自研仿真包；
5. 真机评测属于 stretch goal，需要实验室 Franka / SO100 / FR5 等硬件支持。

## 参考

- LIBERO: arxiv 2306.03310 / https://github.com/Lifelong-Robot-Learning/LIBERO
- CALVIN: arxiv 2112.03227 / https://calvin.cs.uni-freiburg.de/
- RoboTwin 2.0: arxiv 2504.13059 / https://github.com/robotwin-Platform/RoboTwin
- Allen AI VLA Eval Harness: https://allenai.github.io/vla-evaluation-harness/leaderboard/
- 三方向报告引用：[[01_方向一_动态自适应推理]] [[02_方向二_轻量级异常检测]] [[03_方向三_鲁棒性微调]]
