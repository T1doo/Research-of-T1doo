---
create time: 2026-05-21
tags:
  - 曦源项目
  - 立项调研
  - baseline
---

# Baseline 模型横向对比与选型

> [!info] 立项导向
> 立项书给出 6 个 baseline（经典 VLA π0/π0.5、轻量化 VLA SmolVLA、非 VLA ACT/DP/A2A），核心问题是"本项目应选谁作为起点骨干"。本节先给一张主对比表，再用三条 selection criteria（46 GB 显存约束 / 三方向技术耦合度 / 开源代码与社区活跃度）筛选，**结论：SmolVLA 是 95% 场景下的最优起点**，π0 作为对照与天花板，ACT/DP 作为"鲁棒训练能否下放到小模型"的科学对照，π0.5 与 A2A 视具体方向选择性引入。

## 1. 主对比表

| Model | 参数量 | 训练数据 | 训练显存 (bs=2) | 推理延迟 | LIBERO 平均 | 真机表现 | 开源情况 | 双臂支持 | 本项目优先级 |
|-------|--------|----------|----------------|---------|-----------|---------|---------|---------|------------|
| **π0** | 3.3 B (3B VLM + 300M expert) | 10,000 hr / 903 M timesteps / 68 task / 7 robot | **~46 GB**（立项约束基线） | **73 ms**（14+32+27, RTX 4090） | 原论文未直报；openpi 复现 ~85% | shirt folding ~0.95；bussing ~0.80 | openpi（社区+官方） | 双臂 797M | **对照 baseline + 方向二集成对象** |
| **π0.5** | ~3.3 B（与 π0 同量级） | 280k 预 + 80k post，**97.6% 非移动操作机器人** | 同 π0 | ~73 ms | 未公开报告 | 10-15 分钟家庭长任务 | openpi | 同 π0 | **方向二 hierarchical 嵌入复用对象** |
| **SmolVLA** ⭐ | **0.45 B**（350M VLM + 100M expert） | LeRobot 481 dataset / 22.9K traj / 10.6M frame | **~6 GB**（同步），**异步推理需更高** | 同步 ~50 ms / 异步 ~30 ms | **87.3%**（Spatial 90/Object 96/Goal 92/**Long 71**） | SO100 三任务 78.3% > π0 61.7% | lerobot 完全开源 | 暂主要单臂 | **首选骨干**（三方向通用） |
| **ACT** | **80 M** | 50 demo / 任务，10 分钟采集 | **11 GB**（RTX 2080 Ti，5 小时） | ~0.01 s | 未在 LIBERO 标定 | 6 真机任务 80-100%（除 Thread Velcro 20%） | 开源 (ALOHA) | **双臂原生**（7+7 DoF） | 方向三低算力对照 + 双臂场景候选 |
| **DP（Diffusion Policy）** | ResNet-18 + 1D temporal CNN/Transformer（~20-50 M 量级） | 任务级演示 | < 8 GB | **0.1 s**（DDIM 10 步，RTX 3080） | Push-T 等多 benchmark +46.9% | mug flip / pour / shirt folding 等 | 开源 (Columbia) | 部分实验 | 方向三低算力对照 |
| **A2A** | 与 backbone 相关，CNN encoder kernel=5、潜空间 512 维 | LIBERO 5 任务 30 epoch 100 demo | 与 backbone 同 | **0.56 ms 单步**（vs 标准 FM ~3 ms） | LIBERO 5 任务 90.4% | 暂未真机 | 项目页（无 GitHub） | 暂未 | 方向一 1-step 推理候选 |

⭐ = 本项目推荐主线骨干。

## 2. 三条 selection criteria 详评

### 2.1 Criterion 1 — 46 GB 显存约束（"能不能跑"）

立项书第一条硬约束：batchsize=2 全参数微调 46 GB 显存。本项目典型可用硬件是单卡 80 GB A100 或 48 GB 6000 Ada。

- **舒适区**（显存 < 24 GB）：SmolVLA、ACT、DP、A2A（与 backbone 相关）
- **紧张区**（46-60 GB）：π0、π0.5（全参微调 batch=2 时几乎占满 80 GB A100）
- **不可行**：OpenVLA 7B（仅在 H100 等更大显存上跑）

**结论**：如果选 π0/π0.5 作为主骨干，本项目实质上只能用 **LoRA**（r=16 时显存压至 ~16 GB）做微调，意味着方向三 RobustVLA 的 worst-case 外内双循环优化会进一步紧张。如果选 SmolVLA，三方向技术都能在 24 GB 单卡甚至消费级 GPU 上跑通。

### 2.2 Criterion 2 — 三方向技术耦合度

| Backbone | 方向一（推理调度） | 方向二（异常检测） | 方向三（鲁棒训练） | 三方向同时支持 |
|----------|------------------|------------------|------------------|---------------|
| π0 | ✅ AAC 可挂；A2A 需重训 expert | ✅ RC-NF 原论文实测对象 | ✅ RobustVLA 原 backbone | ✅ 但显存最紧张 |
| π0.5 | ✅ 同 π0 | ✅ hierarchical subtask 可作 τ 嵌入 | 未直接评测，方法论同 π0 | ✅ 显存同 π0 |
| **SmolVLA** | **✅ AAC + 异步推理天然耦合（方向一切入点 A）；A2A 可替换 expert（切入点 B）** | **✅ "轻+轻" 组合（方向二切入点 A）** | **✅ RobustVLA 移植（方向三切入点 A）** | **✅ 显存最舒适** |
| ACT | 部分（无语言、无 flow matching） | 可挂（输入接口相同） | 移植 CVAE worst-case δ（方向三切入点 B） | 部分 |
| DP | 部分（无语言） | 可挂 | 移植 DDPM worst-case δ（方向三切入点 B） | 部分 |
| A2A | ✅ 方向一原研究对象 | 较弱 | 较弱 | 不适合 |

**结论**：SmolVLA 是**唯一**在三个方向上都被 agent 报告独立推荐为"首选切入点骨干"的 baseline。这不是巧合——0.45 B 参数 + 完全开源 + LeRobot 框架统一接口，恰好命中"Efficient VLA"立项主题。

### 2.3 Criterion 3 — 开源代码与社区活跃度

| Baseline | 主仓库 | 是否含 LIBERO 评测 | 社区复现量 |
|----------|--------|------------------|----------|
| SmolVLA | HuggingFace `huggingface/lerobot`（含 `physical-intelligence/libero` 数据集） | ✅ 全套 | 高（HF 官方维护） |
| π0/π0.5 | `Physical-Intelligence/openpi`（社区+官方混合） | 部分 | 中高 |
| ACT | `tonyzhaozh/aloha` | ❌（LIBERO 不是 ACT 的主战场） | 中（ALOHA 圈子） |
| DP | `cheng-chi/diffusion_policy` | ❌ | 高（学术界）|
| A2A | 项目页 https://lorenzo-0-0.github.io/A2A_Flow_Matching/（无明确 GitHub） | LIBERO 5 任务 | 低（新论文） |
| RC-NF（方向二核心） | 项目页 https://heikaishuizz.github.io/RC-NF/ （GitHub 待确认） | LIBERO-Anomaly-10 | 极低（刚发布） |
| AAC（方向一核心） | https://lance-lot.github.io/adaptive-chunking.github.io/ | LIBERO / RoboCasa | 低（刚发布） |
| RobustVLA（方向三核心） | https://github.com/gakakulicc/RobustVLA | ✅ LIBERO 全套 + FR5 真机 | 中（已有 demo 视频） |

**结论**：SmolVLA + RobustVLA 是开源最完整的组合；AAC / RC-NF / A2A 三个新论文存在不同程度的复现风险，立项第一周需要专项跟进代码可获取性。

## 3. 推荐骨干选型

### 主选：**SmolVLA**

**理由汇总**：
1. **显存舒适**：6 GB 训练显存 vs 立项约束 46 GB，余量 87%；
2. **三方向通用**：Agent 1/2/3 独立得出"SmolVLA + 方向技术"为最优切入点；
3. **性能足够**：LIBERO 87.3% 与 π0 3.3 B 持平（86.0%），SO100 真机超过 π0；
4. **开源彻底**：LeRobot 框架（HuggingFace 一键 install），含训练 / 评测 / 异步推理完整 stack；
5. **可与 A2A 叠加** 实现 0.45 B + 1-step 的极致推理（方向一切入点 B）。

**风险**：
- LIBERO-Long 仅 71%，长程任务能力是短板——方向一 + CALVIN 评测时需重点修补；
- 异步推理 stack 在多线程调度上 bug 较多，立项早期可能踩坑；
- 数据规模比 OpenVLA 少一个数量级，open-world 泛化能力未必如 π0.5。

### 备选 / 对照：**π0**

- 不作为主骨干（显存太紧），但**必须**作为对照实验组：
  - 方向二 RC-NF 在 π0 上有原论文数字（plug-and-play），SmolVLA 复现需要新的对照点；
  - 方向三 RobustVLA 在 π0 上 +12.6%，移植到 SmolVLA 后必须报告"是否仍能拿到同等比例增益"；
  - 用 LoRA(r=16) 微调 π0 显存 ~16 GB，可行。

### 低算力对照：**ACT / DP**

- 方向三切入点 B 的核心实验对象：把 RobustVLA 的双正则化移植到 80 M 的 ACT、~20 M 的 DP 上，回答"鲁棒训练增益是否依赖大模型容量"。
- ACT 自带双臂支持，与 RoboTwin 配合是天然组合。

### 选择性引入：**π0.5 / A2A**

- **π0.5**：若做 hierarchical 任务表征实验（方向二 τ 嵌入复用）则引入；
- **A2A**：若做方向一切入点 B（替换 SmolVLA expert flow matching 起点）则引入。

## 4. 推荐启动配置（立项第 1 个月）

```
主线 stack:
  - HuggingFace lerobot                      # SmolVLA / ACT / DP 统一接口
  - SmolVLA pretrain checkpoint              # HF Hub 直接下
  - LIBERO + robosuite + MuJoCo              # 评测环境
  - Allen AI VLA Eval Harness                # 吞吐 47×
对照 stack:
  - openpi (π0 LoRA fine-tune)               # 对照实验
方向特定 stack:
  - https://github.com/gakakulicc/RobustVLA  # 方向三参考实现
  - AAC / RC-NF / A2A 三个项目页代码         # 立项第一周下载备份
```

硬件最低要求：
- 单卡 80 GB A100 / 48 GB A6000 / 24 GB 4090 均可；
- 若仅 24 GB 4090：可全参微调 SmolVLA + LoRA 微调 π0；
- 真机评测属 stretch goal，依赖 ALOHA / SO100 / Franka / FR5 等可选硬件。

## 参考

- 三方向报告：[[01_方向一_动态自适应推理]] [[02_方向二_轻量级异常检测]] [[03_方向三_鲁棒性微调]]
- Benchmark 报告：[[05_Benchmark评测体系]]
- 论文 references：[[shukorSmolVLAVisionLanguageActionModel2025]] [[black$p_0$VisionLanguageActionFlow2026]] [[intelligence$p_05$VisionLanguageActionModel2025]] [[zhaoLearningFineGrainedBimanual2023]] [[chiDiffusionPolicyVisuomotor2024]] [[jiaActiontoActionFlowMatching2026]]
