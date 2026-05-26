---
title: "Eva-VLA: Evaluating Vision-Language-Action Models' Robustness Under Real-World Physical Variations"
authors: "Wang et al."
year: "2025"
venue: "arXiv 2509.18953"
arxiv: "2509.18953"
status: "已精读"
tier: "⭐⭐⭐ 必读 · 评估基准"
tags: [literature, T1D, 主线B, 对抗训练, 物理鲁棒性, 评估框架, Eva-VLA]
---

# Eva-VLA — 首个统一物理变化评估框架

> [!info] 元信息
> - **作者**：Wang et al.
> - **arXiv**：[2509.18953](https://arxiv.org/abs/2509.18953)
> - **日期**：2025-09
> - **核心定位**：首个**统一的物理变化鲁棒性评估框架**，把 visual / object / instruction 三类物理扰动以连续优化问题方式系统化呈现
> - **本地 PDF**：未缓存（在 arXiv 直链）

## 📄 TL;DR

Eva-VLA 跨越「离散case-by-case 评估」的传统范式，将 VLA 在真实物理变化下的鲁棒性 **统一形式化为连续优化问题**，覆盖三大维度：(1) 视觉外观变化 (lighting/texture/clutter)；(2) 物体物理属性变化 (size/mass/pose)；(3) 自然语言指令变化 (paraphrase/ambiguity)。框架在 OpenVLA、π0、RDT 等主流 VLA 上跑出系统性结果，发现：**所有主流 VLA 在物理变化下成功率平均下跌 30-50%**，且不同模态的脆弱性高度不对称——物体位姿扰动 > 指令重述 > 视觉纹理。论文不提防御方案，仅提供「**鲁棒性体检报告**」基础设施。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最重要的发现）

1. **「统一评估框架」是 VLA 鲁棒性研究缺失的工程底座**：在 Eva-VLA 之前，每个鲁棒性论文都自定义自己的扰动集（BYOVLA 用 7 类 visual aug、RobustVLA 用 17 类多模态扰动），评估结果不可比较。Eva-VLA 把扰动统一形式化为「在物理参数空间寻找最坏情况扰动 δ ∈ Δ_phys, s.t. SR(π, δ) → min」的连续优化，**让不同论文的鲁棒性数字第一次可比较**。

2. **物理属性扰动 (mass/size) 比视觉扰动更具破坏力**——这与 RobustVLA「action 最脆弱」的发现一致，但 Eva-VLA 进一步定位了 **物体物理属性是「隐藏的最脆弱模态」**。例如 OpenVLA 在物体质量±20% 时 SR 下降 38%，远超光照变化的 12%。

3. **指令模态的脆弱性与 VLA-Risk 论文相互佐证**：Eva-VLA 跑了 paraphrase + ambiguity 后，π0 在 "pick the green block" → "grasp the emerald object" 的同义改写下成功率从 92% → 58%，与 VLA-Risk 报告的指令级 ASR 64% 高度一致。

### 方法论

- **三类物理扰动连续参数化**：
  - 视觉：lighting intensity ∈ [0.3, 2.0]、texture distortion ∈ [0, 1]、clutter density ∈ [0, 10]
  - 物体：size scale ∈ [0.7, 1.3]、mass ∈ [0.5x, 2.0x]、pose ∈ rotation/translation
  - 指令：paraphrase + 同义词替换（GPT-4 生成）+ 模糊指代
- **最坏情况搜索**：用 CMA-ES 进化策略在每个参数轴上找出 SR 下降最大的 δ*
- **聚合指标**：robust SR = 单模态最坏情况 SR 平均；同时报告 perturbation budget vs SR 曲线
- **评测模型**：OpenVLA、π0、RDT、Octo

### 实验关键数据

| 模型 | Clean SR | Visual Worst-case | Physical Worst-case | Instruction Worst-case | 平均下降 |
|---|---|---|---|---|---|
| OpenVLA | 78% | 62% | 40% | 51% | -27% |
| π0 | 92% | 81% | 58% | 64% | -25% |
| RDT | 85% | 70% | 49% | 59% | -26% |

→ **物理属性扰动 (mass/size) 是最致命的攻击维度**。

### 与 [[2026_RobustVLA_Guo]] 的差异（B1 类必答）

| 维度 | Eva-VLA | RobustVLA |
|---|---|---|
| 角色 | **评估框架**（被动测） | **训练防御**（主动防） |
| 扰动数 | 3 大类连续参数化（无限维） | 17 类离散扰动 |
| 时间线 | 2025-09 (先发) | 2026-02 (后续) |
| 关系 | RobustVLA **应当在 Eva-VLA 上测**自身防御效果 | 缺少 Eva-VLA 这层独立评估 |

→ **Eva-VLA 是 RobustVLA 应当报告的「外部评测平台」**，但 RobustVLA 只在 LIBERO 上自评，存在「自己出题自己考」的方法论嫌疑。**这就是你立项的机会**：用 Eva-VLA 框架第三方评测 RobustVLA + FocusVLA + BYOVLA，给出真正可比较的鲁棒性 leaderboard。

### 与曦源关联

1. **可作为 EfVLA 的评测平台**：曦源最终需要给出「我们的轻量 VLA 比 baseline 鲁棒多少」的硬数字，Eva-VLA 是目前最权威的统一框架。
2. **物理属性扰动是 SmolVLA 等小模型的高难度挑战**：mass/size 扰动需要模型有 implicit 物理推理能力，小模型先天弱，Eva-VLA 评测可能暴露 SmolVLA 在此维度的短板。
3. **与 [[2026_VLA-Risk]] 联合使用**：VLA-Risk 强调指令模态 + ASR 攻击成功率视角，Eva-VLA 强调物理参数连续搜索，**两个 benchmark 互补**，立项中 evaluation 章节应同时引用。

### 待解问题

1. **CMA-ES 搜索的计算成本**：每个模型每个扰动维度找 worst-case 需要 ~100 次 rollout，完整评测一个 VLA 需要数千 episode，需估算 H100 时长。
2. **物理参数与 sim-to-real gap**：Eva-VLA 在 simulator (RoboCasa?) 里改 mass/size 容易，真机上不可能动态改物体物理属性。需要真机配套基准。
3. **是否开源 evaluation suite**：从论文页面看代码尚未释出，需关注 GitHub repo。

%% end my-thoughts %%

## 🔗 关联笔记

- **锚点对比**：[[2026_RobustVLA_Guo]]（防御方）— Eva-VLA 是其应当被独立评测的平台
- **互补 benchmark**：[[2026_VLA-Risk]]（指令攻击 + ASR 视角）、[[2026_LIBERO-Plus]]
- **同类评估工作**：[[2025_VLA-Fool]]（多模态对抗攻击的对应攻击侧评估）
- **被评测对象**：[[2024_OpenVLA]]、[[2025_π0]]、[[2026_FocusVLA_Zhang]]
- **B 主线锚点**：[[B_候选清单]]
