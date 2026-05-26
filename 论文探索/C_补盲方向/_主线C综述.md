---
title: 主线 C · 补盲方向 综述
status: 框架 · 待精读笔记回流后补全
target_length: 3500 字
deep_dive: C-iii 失败恢复 + C-ix Test-Time Adaptation + C-x 指令扰动
---

# 主线 C · 补盲方向综述

## C.0 一句话定位

在主线 A（视觉利用，FocusVLA 锚）和主线 B（数据训练，RobustVLA 锚）之外，扫描其他鲁棒性子方向，识别 1-2 个最适合本科生一年期课题的补盲方向。

## C.1 10 候选方向初判（已完成）

详见 [[_候选方向扫描]]。摘要：

| 编号 | 方向 | 推荐度 | 理由 |
|---|---|---|---|
| C-i | 对抗攻击 & 防御 | ⚠️ | 与 B1 重叠度高 |
| C-ii | OOD / UQ | ✅ 备选 | RC-NF 已有基础 |
| **C-iii** | **失败恢复 / Replanning** | ⭐ **首推** | 与 RobustVLA 互补完美 |
| C-iv | 物理扰动 | ❌ | 理论门槛过高 |
| C-v | Safety filter | ⚠️ | 与 RC-NF 重叠 |
| C-vi | 触觉融合 | ❌ | 硬件成本高 |
| C-vii | World model 引导 | ⚠️ 备选 | 实现复杂 |
| C-viii | 形式化验证 | ❌ | 高维不可解 |
| **C-ix** | **Test-Time Adaptation** | ⭐ **首推** | 工程化清晰 |
| C-x | 指令扰动 | ✅ 备选 | VLA-Risk 实证驱动 |

## C.2 三个深耕方向的细节（待精读笔记回流后补全）

### C-iii · 长程任务失败恢复 / Replanning Robustness

**核心问题**：当 VLA 主推策略失败（碰撞、卡住、识别错误）时，系统能否检测、回滚、重规划？

**代表工作**：
- **RaC**（2509.07953）：人在回路的失败恢复训练，10× 数据效率
- **RecoveryChaining**（2410.13979）：分层 RL 学习恢复 sub-policies
- **RePLan**（2401.04157）：LLM 闭环 perception + replanning
- **VADER**（2405.16021）：affordance + 多机器人协作恢复

**与曦源研究的关键关联**：
- RC-NF（[[2026_RCNF_Zhou]]，References 已读）已经提供失败检测的"传感器"
- 主线 B 的 RobustVLA 提供鲁棒主推策略
- C-iii 提供"失败时怎么办"——补全了整个 closed-loop
- **未解问题（论文未答）**：恢复策略本身的鲁棒性？恢复失败的二次恢复？

[详细分析待 P3 精读笔记完成]

### C-ix · Test-Time Adaptation / Online Adaptation for VLA

**核心问题**：VLA 在测试时遇到分布偏移（新光照、新物体），能否在线适应而无需重训？

**代表工作**：
- **TT-VLA**（2601.06748）：测试时任务进度 RL 微调，密集奖励代理
- **VLA-RL**（2025）：测试时采样 + 验证
- **RoboMonkey**（2025）：大规模 TT 采样验证

**与曦源研究的关键关联**：
- 与 RobustVLA（训练时静态鲁棒）形成"训练 + 推理双侧"完整体系
- 与 FocusVLA（视觉利用静态架构）也互补（动态适应 vs 静态架构）
- **本科生可行性最高**：不需重训 backbone；OpenVLA 上可快速迭代
- **未解问题（论文未答）**：奖励代理设计的通用性？计算开销与适应速度的 trade-off？

[详细分析待 P3 精读笔记完成]

### C-x · Instruction Perturbation / Prompt Injection on VLA

**核心问题**：VLA-Risk 实证显示**指令侧 ASR > 视觉侧**（63.99% vs 38.91%），且 FocusVLA 视觉聚焦保护不了指令攻击。怎么做指令鲁棒？

**代表工作**：
- **Image-based Prompt Injection**（2603.03637）：图像隐藏恶意提示
- **Annie**（2509.03383）：VLA 安全 benchmark
- **Jones et al.**（2506.03350）：LLM jailbreak 适配 VLA

**与曦源研究的关键关联**：
- VLA-Risk benchmark 直接对接（I_obj / I_act / I_spa）
- LIBERO-Para 提供同义改写的鲁棒训练数据
- 与主线 A（视觉路线）天然互补
- **未解问题**：infeasible instruction detection 是新问题，几乎无现成方案

[详细分析待 P3 精读笔记完成]

## C.3 最终深耕方向推荐

基于初扫，**C-iii + C-ix 双推**：
- C-iii 是"主动恢复" → 层 3 系统级安全的代表
- C-ix 是"动态适应" → 层 1+2 输入扰动鲁棒的延伸
- 两者都与已有 References 笔记（RC-NF / RobustVLA）紧密结合
- 都能在 LIBERO + VLA-Risk benchmark 上系统评测

C-x 作为备选（VLA-Risk 揭示后会是热门方向）。

[详细对比与最终推荐 — 待精读笔记完成]
