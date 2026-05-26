---
title: "VLA-Risk: Benchmarking Vision-Language-Action Models with Physical Robustness"
authors: "Anonymous (双盲) — Yanchi Ru, Zhengyue Zhao, Yingzi Ma, Xiaogeng Liu, Chaowei Xiao（OpenReview 公开作者）"
year: "2026"
venue: "ICLR 2026 (under double-blind review)"
openreview: "31EjDFwFEe"
itemType: "conference paper (under review)"
status: "已精读"
tier: "⭐⭐⭐ 必读 · benchmark 基础设施"
tags: [literature, T1D, benchmark, 鲁棒性评估, VLA-Risk, ICLR2026]
---

# VLA-Risk — 精读笔记（ICLR 2026 双盲）

> [!info] 元信息
> - **作者**：Yanchi Ru, Zhengyue Zhao, Yingzi Ma, Xiaogeng Liu, Chaowei Xiao（华盛顿大学 Chaowei Xiao 组，安全方向知名学者）
> - **场所**：ICLR 2026 双盲投稿
> - **OpenReview**：[31EjDFwFEe](https://openreview.net/forum?id=31EjDFwFEe)
> - **本地 PDF**：`论文探索/_pdf_cache/VLA-Risk_OpenReview.pdf`
> - **数据来源**：构建于 LIBERO + VLABench + nuScenes 之上
> - **核心定位**：第一个 VLA 物理鲁棒性的系统 benchmark

## 📄 Abstract & 关键贡献

VLA 在统一感知/语言/动作执行上取得显著进步，但带来新的攻击面：跨指令执行和视觉理解的脆弱性。VLA-Risk 是**第一个系统性 benchmark**，评估 VLA 在 **2 模态（image / instruction）× 3 任务维度（object / action / space）= 6 种扰动类型** 下的鲁棒性。覆盖：
- **296 scenarios / 3784 episodes**
- 3 个场景类别：**Direct Manipulation**（80/1600）+ **Semantic Reasoning**（96/384）+ **Autonomous Driving**（120/1800）
- Victim models：OpenVLA、π0、OpenEMMA

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（最有价值的三个发现）

1. **指令级扰动比视觉扰动破坏力更大**（这是最反直觉的关键发现）：
   - 指令级 ASR 均值 **63.99%** vs 视觉级 38.91%
   - OpenVLA 在 Direct Manipulation 上：视觉 Action 攻击 ASR 62.61% vs 指令 Action 攻击 ASR **80.87%**
   - 含义：当前 VLA 的语言-视觉对齐机制 **更易被语言侧击穿**——对你的研究很关键，**FocusVLA 的视觉聚焦改进可能保护不了指令侧扰动**。

2. **「Infeasible Instruction」是被严重忽略的攻击面**：
   - 三类不可行指令：Attribute Contradiction（红蓝杯）、Action Impossibility（向上倒水）、Space Inconsistency（同时左+右）
   - VLA 优化目标是任务完成，不是任务可执行性判断；面对违反物理常识的指令仍盲目执行
   - **这给安全部署带来严重隐患**：危险指令的语义合理但物理不可行（如"把玻璃杯放进餐桌"——只有纸杯可用）

3. **即使攻击失败，也带来"步数膨胀"风险**（隐性 cost）：
   - OpenVLA 在视觉 Direct Manipulation 上，攻击失败的 episode 平均执行步数 +21.33（如 Action 任务从 ~280 → 300+ 步）
   - 这反映出"决策迟疑"——即使最终完成任务，模型陷入计算开销和潜在安全风险
   - **传统 SR/ASR 指标会掩盖这个问题**——你的研究里如果只看成功率，会漏掉一类失败模式。

### 方法论关键细节

#### 6 维扰动分类
```
                     Vision (V)              Instruction (I)
Object  (obj)        Object Mislabeling      Attribute Contradiction
Action  (act)        Action Contradiction    Action Impossibility  
Space   (spa)        Space Distraction       Space Inconsistency
```

#### 扰动生成 pipeline（3 阶段）
1. **Instruction Parsing**：GPT-5 把自由指令转 `(object, attributes, action, spatial)` 结构化 tuple
2. **Perturbation Planning**：在候选扰动集 $\mathcal{C}_t$ 上最大化 attack success
$$\max_{c_t} \mathbb{E}_{(Img_{ori}, I) \sim S}[f_t(Img_{ori}, Inst_{ori}, c_t)] \quad \text{s.t. } \mathcal{D}_t(c_t) \leq \tau_t$$
3. **Rendering + Stability Augmentation**：
$$Img_{pert} = \text{Norm}\big(R(Img_{ori}, c; A_t(Img_{ori}) + \Delta p, \alpha_0 + \Delta \alpha)\big)$$
   - 用 Grounding DINO 定位目标 + bbox center anchor
   - 加位置抖动 $\|\Delta p\|_\infty \leq \rho$ 和透明度变化 $|\Delta \alpha| \leq \eta$

#### 评测指标
- **Attack Success Rate (ASR)**：$\text{ASR}_{vis} = (\text{SR}(\mathcal{D}_{ori}) - \text{SR}(\mathcal{D}_{pert}))/\text{SR}(\mathcal{D}_{ori})$
- 指令级：$\text{ASR}_{inst} = \text{SR}(\mathcal{D}_{pert})/\text{SR}(\mathcal{D}_{ori})$（错误执行率）
- 自动驾驶用 ADE 阈值：$\mathbb{1}[(ADE_{pert} - ADE_{ori}) > \epsilon] = 1$

### 关键实验结果

#### OpenVLA · Direct Manipulation 性能崩塌表
| 任务 | Benign SR | Vision Attack SR | Instruction Attack SR | V-ASR | I-ASR |
|---|---|---|---|---|---|
| Object | 72.5 | 31.0 | 17.5 | 57.24% | 75.86% |
| Action | 57.5 | 21.5 | 11.5 | 62.61% | **80.87%** |
| Space  | 82.0 | 39.0 | 11.0 | 52.44% | **86.59%** |

#### π0 · Semantic Reasoning
| 任务 | Benign | V-Attack | I-Attack | V-ASR | I-ASR |
|---|---|---|---|---|---|
| Object | 57.81 | 42.19 | 46.88 | 27.02% | **81.09%** |
| Action | 57.81 | 35.94 | 37.83 | 37.83% | 75.68% |
| Space  | 57.81 | 34.38 | 40.53 | 40.53% | 43.25% |

#### OpenEMMA · Autonomous Driving（ADE 单位 m）
- Object: Benign 1.09 / Attacked 1.69 → ASR 视 44.40 / 指 33.30
- Action: Benign 1.09 / Attacked 1.44 → ASR 视 66.70 / 指 55.60
- Space:  Benign 1.09 / Attacked 1.27 → ASR 视 22.20 / 指 55.60

### 与曦源项目（鲁棒性主线）的关联

**这是你立项必须用的 benchmark！**理由：

1. **覆盖 FocusVLA 没评测的所有维度**：
   - FocusVLA 只在 LIBERO clean 上跑（98.7% 多权重），完全没做扰动评测
   - VLA-Risk 的 6 维扰动可以直接用来测 FocusVLA 的"隐性鲁棒性"——验证导师的"视觉聚焦 → 鲁棒"假设
   - **可行实验**：复现 FocusVLA + 在 VLA-Risk 上跑 → 对比 OpenVLA / π0 → 看视觉/指令侧鲁棒增益分布

2. **直接揭示 RobustVLA 的盲区**：
   - RobustVLA 在 17 视觉 + 部分 instruction 扰动上做 worst-case δ；但 VLA-Risk 引入了"infeasible instruction"这个新维度（Attribute/Action/Space Impossibility），RobustVLA 没覆盖
   - **可成为新的研究问题**：能否在 RobustVLA 的多臂 bandit 里加入 infeasibility detection？

3. **数据集都是开源基础**：
   - LIBERO（Direct Manipulation）+ VLABench（Semantic Reasoning）+ nuScenes（AD）都是你已经能用的资源
   - 用 GPT-5 自动生成扰动 → 复现门槛低
   - 但需要等代码 release

4. **明确了"指令侧 > 视觉侧"的攻击优先级**：
   - 你的研究方向选择应该 **重视指令鲁棒性**——单纯做视觉鲁棒（FocusVLA 路线）会留下大半攻击面
   - 这正好为「主线 A 视觉利用 + 主线 C 指令鲁棒性补盲」的组合提供了实证依据

### 待解问题 / 后续追读

#### VLA-Risk 自己留下的研究空白

1. **"开发对应防御机制"是论文 future work**——这正是你的研究机会！
2. **未评测 FocusVLA / VLA-Adapter / SmolVLA 等小模型**——你可以补齐这块对比
3. **没做"双重扰动"（同时视觉+指令）**——多扰动叠加的鲁棒性曲线值得探索
4. **指令侧扰动依赖 GPT-5**——能否用更小的 LM 生成？这关系到 benchmark 的复现成本

#### 必须追读
1. **VLABench**（Zhang 2024）——Semantic Reasoning 任务来源
2. **OpenEMMA / DriveVLM**——AD 域 VLA baseline
3. **Annie: Be careful of your robots**（Huang 2025, 2509.03383）——安全 benchmark 比较对象
4. **Jones et al. 2025**（2506.03350）——adversarial attacks on robotic VLA 同期工作
5. **Wang et al. 2025**——视觉对抗 patch 攻击 VLA

%% end my-thoughts %%

## 🔗 关联笔记
- [[2026-03_FocusVLA_Zhang]]：必须在 VLA-Risk 上跑评测
- [[2026_RobustVLA_Guo]]：互补——RobustVLA 训练侧、VLA-Risk 评测侧
- [[benchmarks/LIBERO-Plus]]：另一个鲁棒性 benchmark，互补
- [[benchmarks/VLA-Arena]]：同期另一个 benchmark
- [[C-x_instruction_perturbation]]：指令侧鲁棒性子方向

## 📌 Action Items
- [ ] 等代码 release（关注 OpenReview / GitHub）
- [ ] 用 VLA-Risk 的 6 维扰动评测 FocusVLA 复现版本
- [ ] 设计「FocusVLA + 指令鲁棒模块」联合训练实验
- [ ] 在综述章节里专门一段讨论「指令鲁棒性的优先级」

%% Import Date: 2026-05-26 %%
