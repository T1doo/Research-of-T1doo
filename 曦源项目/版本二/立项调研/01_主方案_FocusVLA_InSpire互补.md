---
create time: 2026-05-26T22:10:00
update time: 2026-05-27T01:30:00（v2 修订摘要）
tags:
  - 曦源项目
  - 版本二
  - 主方案
  - FocusVLA
  - InSpire
  - Plan_A
status: v2 修订（详见 08 最终方案）
---

> [!note] v2 修订（2026-05-27）
> 本文档保留作为 v1 历史记录。**v2 后续更新只在 [[08_最终方案_FocusVLA_InSpire互补研究]] 中维护**。核心修订：
> - 4 变体（vanilla / +InSpire / +Focus-style / 叠加）→ **2 变体（vanilla / +InSpire）+ 视觉利用 ablation**
> - 移除 "OpenVLA-OFT 作为 Focus-style 代理" 的 framing（不严谨）
> - H1 从"两类方法强项不重叠"改为"显式提示在多维扰动上的鲁棒性增益"
> - H2（叠加）降级为 discussion-level
> - H3（failure predictor）保留
# 01 · 主方案 · FocusVLA × InSpire 互补研究（Plan A）

> **一句话定位**：用 InSpire 的 zero-data plugin（强制 VLA 先输出方向词再输出 action）作为**显式 spatial grounding**，对照 FocusVLA / OpenVLA-OFT 代表的**隐式 attention 聚焦**，在 LIBERO-Plus 7 维 × VLA-Risk 6 维扰动上系统验证两路 grounding 的鲁棒性差异、互补性、可叠加性。

## 1. 问题陈述

### 1.1 研究背景

VLA（Vision-Language-Action）模型在 LIBERO clean 上已达 95-99% 成功率，但在分布偏移（distribution shift）下成功率剧烈下降。**这是部署到真实场景的最大障碍**。

学界提出了两条主要技术路径来提升 VLA 鲁棒性：

| 路径 | 代表工作 | 核心机制 | 工程开销 |
|---|---|---|---|
| **显式 grounding** | InSpire (arXiv 2505.13888) | 在 prompt 前置方向问题，强制 VLA 先输出 spatial 关系词 | 极低（plugin，零数据） |
| **隐式 attention 聚焦** | FocusVLA (arXiv 2603.28740)、OpenVLA-OFT | 改架构 attention 让模型"被迫看图"或省略视觉特征 | 中-高（架构改造） |

### 1.2 研究空白（立项关键切入点）

1. **FocusVLA 完全没在 robustness benchmark 上量化** — 论文只报 LIBERO clean SR，所有鲁棒性都是 side effect 论断
2. **InSpire 与 FocusVLA 未做对照** — 两路 grounding 的鲁棒性分工、互补性、可叠加性，没有系统化研究
3. **FocusVLA 作者自述局限 #2「对初始状态敏感」**——这恰恰是 InSpire 的方向词最擅长解决的问题（方向词与 robot frame 直接耦合）

### 1.3 一句话研究问题

> 在 LIBERO-Plus 7 维扰动 + VLA-Risk 6 维扰动下，**显式 spatial grounding (InSpire)** 与**隐式 attention 聚焦 (FocusVLA / OpenVLA-OFT)** 是否在不同扰动维度上**强项互补**？两者是否可**无成本叠加**获得进一步增益？

## 2. 核心假设

### H1（互补假设）

> 显式 grounding（InSpire）在**初始状态扰动 / 视角变换 / spatial inconsistency** 上鲁棒增益显著（因为方向词与机器人 frame 直接耦合）；隐式 attention 聚焦类（OpenVLA-OFT / SmolVLA + Focus-style）在**背景纹理 / distractor / light variation** 上鲁棒增益显著。**两类方法的强项维度不重叠**。

**验证**：4 模型变体 × 7 维 LIBERO-Plus 扰动 = 28 实验。计算每模型在每维度的 SR drop，统计哪些维度上哪类方法 drop 更小，做配对 t-test。

### H2（叠加假设）

> InSpire plugin + Focus-style 架构的简单叠加在 LIBERO-Plus 7 维平均 SR 上**优于任一单独使用** ≥ 3pp。

**验证**：直接对比 4 变体的 7 维平均 SR。要求 (InSpire + Focus) > max(InSpire only, Focus only) + 3pp。

### H3（诊断假设）

> 方向词预测准确率与最终 action 成功率的相关系数 r ≥ 0.6 → InSpire 可作为运行时 failure predictor。

**验证**：收集 ~2k 成功/失败 episode，每个 episode 提取方向词预测序列（softmax confidence + top-1 accuracy 时序均值）。训一个 LightGBM 二分类器预测最终成败，要求 AUC ≥ 0.65。

## 3. 与 FocusVLA 的对齐方式（双向）

1. **FocusVLA 是核心对照**（"隐式聚焦"代表）：4 个变体中至少 2 个用 FocusVLA-style 架构
2. **正面回应 FocusVLA 作者自述局限**：
   - 局限 #2「对初始状态敏感」← InSpire 的 6 方向 spatial grounding 直接打这个洞
   - 局限 #4「未做 robustness benchmark」← 本项目核心贡献
3. **故事完整**：用 InSpire 补 FocusVLA 的洞，再实证两者在 7 维上的分工

## 4. 技术路线（5 phase / 12 月）

### Phase 1 · 复现 InSpire on OpenVLA-OFT（M1-M2，6 月-7 月）

**目标**：让 InSpire plugin 在 OpenVLA-OFT + LIBERO-Spatial 上跑通 clean baseline

**关键动作**：
1. 用 OpenVLA-OFT-7B（HF: `openvla/openvla-7b-oft`，代码完整）作主 backbone；SmolVLA-0.5B 作备
2. LoRA(r=16) 微调，节省显存到 < 40GB
3. **改 prompt**：原任务前置 `"In which direction is the [target] relative to the robot? Options: right/left/up/down/front/back/grasped."`
4. **加 7-class 方向词 cls head**（极简：linear layer on hidden states）
5. **方向词 GT 自动推导**：从 LIBERO demo 自带的 robot end-effector pose + object pose，写 6 方向 + grasped 自动标注脚本
6. **损失**：方向词 cross-entropy + 原 action regression 联合（loss weight 0.1:1.0 起步）

**关键文件路径**（推测，需 W1 确认）：
- `openvla-oft/scripts/finetune_libero.py` — LoRA 微调入口
- `openvla-oft/prismatic/models/vlms/prismatic.py` — backbone 模型类，prompt 改造点
- `libero/libero/benchmark/__init__.py` — LIBERO 任务定义

### Phase 2 · clean + 7 维分组评测（M3-M5，8 月-10 月）

**目标**：跑完 4 变体 × 7 维 = 28 实验，验证 H1

**4 个变体**：
1. `vanilla` — OpenVLA-OFT-7B + LoRA 微调，无 InSpire 无 Focus
2. `+InSpire` — vanilla + InSpire plugin
3. `+Focus-style` — Focus-style 代理（如 FocusVLA 已开源就用原版；否则用 OpenVLA-OFT 默认配置，因为它本身就是"省略视觉特征" shortcut 架构的代表 — 见 FocusVLA 论文 §2 对 OpenVLA-OFT 的批判）
4. `+InSpire & Focus-style` — 两者叠加

**LIBERO-Plus 7 维**：layout / viewpoint / state / instruction / light / texture / noise（详见 [[2025-10_LIBERO-Plus]]）

**评测规模**：每维度 500 instance（子集化，控制成本），总 7 × 500 × 4 = 14,000 episode × 30s = ~117 GPU-h

**额外评测**：VLA-Risk 6 维（V_obj/V_act/V_spa/I_obj/I_act/I_spa）作为附加 stretch（如果 P2 末仍有 GPU 预算）

### Phase 3 · 互补性归因分析（M6-M7，11 月-12 月）

**目标**：把 7 维分组并做统计归因，验证 H1

**分组**（基于先验）：
- **「视觉聚焦敏感」组**：texture / light / noise / instruction（如果包含视觉描述）
- **「spatial grounding 敏感」组**：layout / viewpoint / state

**归因方法**：
1. 每模型变体 × 每组 → 平均 SR drop
2. 配对 t-test：要求每组中 InSpire-only 与 Focus-only 的 drop 差异 ≥ 2pp，p < 0.05
3. 画雷达图：4 变体 × 7 维 SR

### Phase 4 · 方向词作为 failure predictor（M8-M9，01 月-02 月）

**目标**：验证 H3，与 attention-based predictor 对比

**步骤**：
1. 从 P2 评测中收集 ~2k 配对 episode（成功/失败各半）
2. 提取方向词预测时序（每步 7-class softmax + top-1 accuracy）
3. 训 LightGBM 二分类器（输入是时序特征，输出是 episode 最终成败）
4. 评测 AUC，与基于 attention entropy 的 baseline predictor 对比
5. 如有时间：交叉验证（在 LIBERO-Spatial 训，在 LIBERO-Long 测）

### Phase 5 · 写作 + 投稿（M10-M12，03 月-05 月）

**目标**：至少投出 1 个 workshop

**关键 deadline**（粗估，需以最新 CFP 为准）：
- CoRL 2027 workshop：通常 8-9 月
- ICLR 2027 Robot Learning workshop：通常 2-3 月
- NeurIPS 2027 Robot Learning workshop：通常 9-10 月
- 国内会议（CCC / CAA）：3-5 月 deadline

**故事结构**：
- §1 Intro：FocusVLA 没在 robustness benchmark 上量化 → 我们补
- §2 Related：grounding 文献（VoxPoser / ReKep / InSpire / FocusVLA）
- §3 Method：InSpire on OpenVLA-OFT + Focus-style 对照设置
- §4 Experiment：4 变体 × 7 维结果 + 互补归因
- §5 Discussion：failure predictor 应用 + 局限

## 5. Go/No-Go 准则

| 节点 | Go 条件 | No-Go 触发 → 应对 |
|---|---|---|
| **P1 末（M2）** | InSpire 在 LIBERO-Spatial clean SR ≥ vanilla baseline（差 ≤ 5pp） | <vanilla - 5pp → 查方向词 GT bug；调 loss weight；若 1 个月内无改善 **→ 切 [[02_备选方案_FocusVLA鲁棒性诊断]]** |
| **P2 末（M5）** | H1 至少部分成立：两类方法在 ≥ 1 维度强项不重叠 | 完全重叠 → 改叙事为「两路 grounding → 同一 grounding」，故事仍可写 |
| **P3 末（M7）** | 互补归因每组 ≥ 2pp 差异，p < 0.05 | 不显著 → 增 episode 数 / 换 baseline 模型 |
| **P4 末（M9）** | failure predictor AUC ≥ 0.65 | < 0.6 → 改描述性可视化，去掉 predictor 卖点 |

详细决策树见 [[05_风险与缓解]]。

## 6. 关键工具 / 代码 / 数据依赖

| 类别 | 依赖 | 状态 | 备注 |
|---|---|---|---|
| 主 backbone | `openvla/openvla-7b-oft` (HF) | ✅ 已开源 | LoRA r=16 显存 < 40GB |
| 备 backbone | `HuggingFaceTB/SmolVLA-base` | ✅ 已开源 | 显存更小，速度更快 |
| 评测主 benchmark | LIBERO（4 suites） | ✅ GitHub 开源 | `Lifelong-Robot-Learning/LIBERO` |
| 鲁棒评测 | LIBERO-Plus（7 维） | ✅ HF 开源 | `senyufei/LIBERO-Plus` |
| 鲁棒评测附加 | VLA-Risk（6 维） | ⏳ 待 release | 论文有 arXiv，代码未释，stretch goal |
| InSpire 代码 | UESTC Lianli Gao 组 | ⏳ 待 release | 论文承诺；如未释，200 行可手写 |
| FocusVLA 代码 | HIT-Shenzhen | ⏳ 待 release | 不强依赖，OpenVLA-OFT 可替代 |
| 模拟器 | robosuite + MuJoCo | ✅ 开源 | LIBERO 自带 |

## 7. 必读 / 强参考资料

- [[2025-05_InSpire_Zhang]] — 主方法 plugin
- [[2026-03_FocusVLA_Zhang]] — 对照 anchor
- [[2025-10_LIBERO-Plus]] — 7 维评测
- [[VLA-Risk_ICLR2026]] — 6 维评测（stretch）
- [[2024_OpenVLA]] — backbone 文档
- [[A6_shortcut_learning_总结]] — shortcut learning 背景
- 完整列表见 [[06_关联资料索引]]

## 8. 实验设计细节

### 8.1 评测指标
- **主**：任务成功率 SR（每任务 50 instance）
- **辅 1**：方向词预测准确率（每步 7-class accuracy）
- **辅 2**：方向词 confidence 时序均值
- **辅 3**：长程任务平均完成步数（衡量稳定性）

### 8.2 统计方法
- 每个 cell（变体 × 维度）报告 mean ± std（3 seed）
- 主对比用配对 t-test（同 instance 不同模型对比）
- 多重比较 Bonferroni 校正

### 8.3 消融实验
- InSpire loss weight 扫描：{0.05, 0.1, 0.2, 0.5}
- 方向词粒度对比：6 方向 vs 8 方向（加左前 / 右前）
- 方向词位置：prompt 前置 vs 后置

## 9. 与综述报告 6 候选方案的关系

本主方案对应原排序中的"候选 1（FocusVLA 鲁棒性显式评测与改进）"的**轻量化变体**：

- 去掉了原方案中的"FocusVLA + RobustVLA 联合训练"（工程量太大，对 2 人本科团队不现实）
- 增加了 InSpire plugin 作为更轻量的方法贡献（与原候选 1 的纯评测路线相比，故事更"硬"）
- 保留了 LIBERO-Plus + VLA-Risk 评测设计

### 升级路径（若 P2 末 H2 成立且团队学有余力）

可以把 [[02_RobustVLA相关工作扩展]] 中的 TRADES-mini consistency loss 嫁接进来，作为 P5 末投稿前的附加贡献（"InSpire + Focus + TRADES 三方组合"）。但这是 stretch goal，不在主路径里。
