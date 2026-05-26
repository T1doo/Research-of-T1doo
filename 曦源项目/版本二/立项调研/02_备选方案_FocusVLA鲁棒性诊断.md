---
create time: 2026-05-26T22:15:00
tags:
  - 曦源项目
  - 版本二
  - 备选方案
  - FocusVLA-Probe
  - Plan_B
---
# 02 · 备选方案 · FocusVLA-Probe 鲁棒性多维诊断（Plan B）

> **一句话定位**：完全不训练 VLA，把 FocusVLA（或其代理 OpenVLA-OFT/SmolVLA）当黑盒，在 LIBERO-Plus × VLA-Risk × LIBERO-Para 三大 benchmark 上做**系统化鲁棒性诊断 + 失败模式画像**，给出第一张「视觉聚焦类 VLA 的失败模式画像」。

## 1. 何时启动 Plan B

仅在以下任一条件触发时切换到 Plan B：

1. **Plan A P1 末（M2，2026-07 末）InSpire 复现卡壳** — 方向词 GT 推导噪声 > 30%（>50/100 样本错误），或 cls head 训不动（准确率 < 30%）
2. **Plan A P2 中（M4 末，2026-09 末）评测进度严重落后** — 28 实验跑完 < 30%
3. **GPU/预算超支** — M4 末预算用掉 > 60% 且评测未完成 70%

切换的**操作成本** < 1 个月，因为 Plan A 的 OpenVLA-OFT + LIBERO-Plus 评测 pipeline 可直接复用，只需关掉 InSpire 模块。

## 2. 与 Plan A 的关系（共用 vs 替换）

| 组件 | Plan A 状态 | Plan B 状态 |
|---|---|---|
| OpenVLA-OFT-7B 环境 | ✅ 用 | ✅ 用（不变） |
| SmolVLA-0.5B 环境 | ✅ 用（备） | ✅ 用（作为对照） |
| LIBERO-Plus 7 维评测脚本 | ✅ 用 | ✅ 用（不变） |
| InSpire plugin | ✅ 主方法 | ❌ 不用 |
| 方向词 GT 标注脚本 | ✅ 用 | ❌ 不用 |
| Attention map 提取（PyTorch hook） | ❌ 不用 | ✅ **主分析手段** |
| Failure predictor (LightGBM) | ✅ H3 验证 | ✅ 主贡献 |

## 3. 核心研究问题

### H1'（维度选择性）

> FocusVLA 的隐性鲁棒增益是**维度选择性**的——对背景纹理 / distractor 鲁棒（"视觉聚焦"奏效），但对**初始状态、视角、语言改写**无显著增益。

### H2'（attention-failure 关联）

> FocusVLA 失败 episode 的 attention map 存在**可诊断的早期信号**——前 N 步 attention entropy / patch top-K 偏离正常分布 > τ → 80%+ 概率最终失败。

## 4. 技术路线（5 phase / 12 月）

| Phase | 月份 | 目标 |
|---|---|---|
| **P1 复现 baseline** | M1-M2 | OpenVLA-OFT-7B + SmolVLA-0.5B 在 LIBERO clean 上跑通至少 65% baseline。**不复现 FocusVLA 全架构**——用 OpenVLA-OFT 作 Focus-style 代理 |
| **P2 单维度扰动评测** | M3-M5 | LIBERO-Plus 7 维 + VLA-Risk 6 维（子集，每维 ≤ 500 instance） |
| **P3 attention-failure 关联诊断** | M6-M8 | 收 ~2k episode 的 attention map（OpenVLA-OFT last-layer），训 LightGBM 二分类器 |
| **P4 失败模式聚类与画像** | M9-M10 | 把失败 episode 按 (扰动维度, 失败阶段, attention pattern) 三元组聚类，给 5-8 类典型失败模式 |
| **P5 写作 + 投稿** | M11-M12 | ICRA 2027 workshop / CoRL 2027 workshop on Robot Learning |

## 5. Go/No-Go 准则

| 节点 | Go 条件 | No-Go 触发 |
|---|---|---|
| P1 末 | OpenVLA-OFT 在 LIBERO-Spatial clean SR ≥ 65% | < 50% → 降级到 LIBERO 原版（不评测扰动）；环境检查 |
| P2 末 | ≥ 2 个维度统计显著（p<0.05）的脆性 + ≥ 1 个维度 FocusVLA 代理显著优于 baseline | 0 显著差异 → 增加 instance 数 / 换 baseline；仍无 → 转向"为什么所有 VLA 都崩"的负面 finding |
| P3 末 | failure predictor AUC ≥ 0.70 | < 0.65 → 降级为「attention 描述性可视化」，去掉 predictor 卖点 |
| P4 末 | ≥ 5 类可解释失败模式 | < 3 类 → 改投技术报告 / arXiv only |

## 6. 关键优势（为什么作为 Plan B 而不是 Plan A）

| 维度 | Plan B 表现 |
|---|---|
| **风险** | 极低（不训练，只评测） |
| **GPU 时长** | ~50-100 GPU-h（仅 inference + LightGBM 训练） |
| **复现依赖** | 不需要 InSpire / FocusVLA 任何代码 release |
| **2 人本科生承受度** | ✅ 非常容易（无架构修改、无新损失） |
| **novelty** | 中（系统化诊断报告本身有价值，作者自述局限 #4 就是这个） |
| **退路** | 已经是最保守路线，再降级只能投技术报告 |

## 7. 预期产出（如成为最终路线）

- **第一目标**：ICRA 2027 workshop / CoRL 2027 workshop on Robot Learning 短稿（4-6 页）
- **第二目标**：arXiv 技术报告 + HuggingFace 数据集 release（attention-failure pair 数据）
- **退路**：国内会议（CCC / AICTC）+ 校内 BC 结项

## 8. 与 Plan A 的可融合性

如果 Plan A 进度顺利，Plan B 的部分组件可以作为 Plan A 的**附加章节**：

- **Plan A §Failure analysis** 可直接复用 Plan B P3-P4 的 attention map 分析方法
- **Plan A §Discussion** 可引用 Plan B 的 attention-failure 关联结果作为「为什么 InSpire 显式 grounding 有效」的解释证据

这意味着 P1-P2 期间，**B 同学的"评测脚本 + attention hook"准备工作可同时服务 Plan A 和 Plan B**——是真正的"共用 pipeline"。

## 9. 必读 / 强参考资料

- [[2026-03_FocusVLA_Zhang]] — 4 大局限是诊断蓝图
- [[2025-10_LIBERO-Plus]] — 7 维评测
- [[VLA-Risk_ICLR2026]] — 6 维评测
- [[2024_OpenVLA]] — backbone 文档
- [[2025-08_ShortcutLearning_Xing]] — shortcut 诊断方法学（如何"找 spurious correlation"）

## 10. 一句话总结

> Plan B 是 Plan A 的"无训练精简版"，工程量减半但 novelty 减一档；作为兜底是合理的，但默认目标仍是 Plan A。
