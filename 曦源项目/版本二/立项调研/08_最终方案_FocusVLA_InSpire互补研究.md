---
create time: 2026-05-26T23:00:00
tags:
  - 曦源项目
  - 版本二
  - 最终方案
  - FocusVLA
  - InSpire
  - 工作版v1
status: 工作版v1
audience: 项目组 + 导师 + workshop paper 撰写参考
---

# 08 · 最终方案 · FocusVLA × InSpire 互补研究（深度版）

> [!warning] 使用说明
> 本文档是**项目组内部技术工作版**，含数学命题、effect size 估计与广泛文献综述。
> **不适合直接抄入曦源申请书**——申请书请参考 `版本二/递交材料/1-2.申请书填写草稿.md`，对标 2024 学长样本的简洁版。
> 两份文档承担**完全不同的角色**，互斥使用。

> [!info] 文档定位
> - 整合 [[00_总览与立项决定]] ~ [[07_为什么不选其他方案]] 9 个立项调研文件的核心内容
> - 通过 web 搜索补充 17 篇近期相关工作论文（标注 ⭐ web-new）
> - 形成 6 个月后写 workshop short paper §1-§3 的素材库
> - 长度 7500 字 · 含 § 0 TL;DR · 4 子主题相关工作综述 · H1-H3 形式化论证 + 3 附录

---

## § 0 · TL;DR

VLA（Vision-Language-Action）模型在标准 LIBERO clean SR 已达 95-99%，但在 LIBERO-Plus 等扰动评测下断崖式下跌至 30% 上下。提升鲁棒性目前有两条主要路径：**显式空间提示**（如 InSpire 的方向词 plugin，在 prompt+output 层注入信号）和**隐式注意力聚焦**（如 FocusVLA 的 Cascaded Attention + Focus Attention，在 attention 层切断 shortcut）。两条路径目前各自发展，**缺乏在统一 backbone × 统一评测协议下的系统对照**。本项目以 OpenVLA-OFT-7B 为统一主干，训练 4 个模型变体（vanilla / +InSpire / +Focus-style / +两者叠加），在 LIBERO-Plus 7 维 + VLA-Risk 6 维（stretch）扰动评测下验证 3 个核心假设：**H1（互补）** 两类方法的强项维度不重叠；**H2（叠加）** 两者组合平均 SR > max(单独) + 3pp；**H3（诊断）** 方向词预测可作为运行时 failure predictor（AUC ≥ 0.65）。预期产出 1 篇 4-6 页 workshop short paper（CoRL / ICLR / NeurIPS Robot Learning workshop）+ 开源 plugin 代码 + 方向词标注数据集。

---

## § 1 · 问题陈述与研究空白

### 1.1 VLA 鲁棒性现状

VLA 模型在 LIBERO 4 个 task suite 上的 clean SR 普遍达 95-99%（FocusVLA 报 98.7%，OpenVLA-OFT 报 98.5%，VLA-Adapter-Pro 报 98.5%）。但 LIBERO-Plus（Fei 等 2025，[arXiv:2510.13626](https://arxiv.org/abs/2510.13626)）通过 7 维扰动（layout / viewpoint / state / instruction / light / texture / noise）系统评测后发现：**主流 VLA 在 viewpoint 与 initial state 扰动上的 SR 普遍从 95%+ 跌至 30% 上下**——一个 60pp 的鸿沟。VLA-Risk（Wang 等 2026 ICLR，[arXiv 待补](https://arxiv.org/abs/0000.00000)）进一步在指令侧扰动上揭示了类似断崖：6 维扰动下平均 ASR 高达 63.99%（视觉侧 38.91%）。

**这是 VLA 走出实验室、走入家用 / 医疗辅助等真实场景的最大障碍。**

### 1.2 两类技术路径的对比

学界 2024-2026 年涌现的 VLA 鲁棒性增强方法可大致归为两类。下表对比 8 个代表工作的"信号注入位置"与"工程开销"：

| 方法 | 路径 | 信号注入位置 | 需要新数据 | 工程开销 | 推理 latency 代价 |
|---|---|---|---|---|---|
| **InSpire** (2505.13888) | 显式 | Prompt + output cls head | ❌（自动标注） | 极低 | +1 个 token decode |
| **SG-VLA** (2603.22760) ⭐ web-new | 显式 | 辅助 spatial decoder | ❌ | 中 | 0（仅训练辅助） |
| **Gaze-Reg VLA** (2603 Pani) | 显式 | KL alignment to gaze | 需要 gaze data | 中 | 0 |
| **AutoFocus-IL** (2511 Gong) | 显式 | VLM saliency → attention | ❌ | 中 | 0 |
| **TraceVLA** (2412 Zheng) | 显式 | Visual trace prompt | ❌ | 低 | 0 |
| **FocusVLA** (2603.28740) | 隐式 | Cascaded + Focus attention | ❌ | 高（架构改造） | 0 |
| **OpenVLA-OFT** (2502) | 隐式 | 省略视觉特征 shortcut | ❌ | 低（架构选择） | 0 |
| **Spatial Forcing** (2511) | 隐式 | 3D-aware attention | ❌ | 高 | 0 |

两路差异本质：**显式路径**通过强制模型在 output 层产生中间表示（方向词、辅助标签）来纠正决策；**隐式路径**通过架构改造让 attention 在 feature 层自然聚焦。

### 1.3 三个研究空白

通过对 101 篇精读笔记（[[reference-paper-notes]]）与 17 篇新搜论文（详见 § 2）的整理，识别出三个明确空白：

1. **FocusVLA 完全没在 robustness benchmark 上量化**——其论文只报 LIBERO clean 98.7%，4 大局限中第 #4 项明确写"未在 robustness benchmark 上系统评测"
2. **显式 vs 隐式两类 grounding 缺乏直接对照**——所有相关论文都只评测自家方法 vs 其他 vanilla baseline，**没有任一论文同时在统一 backbone 上跑两条路径**
3. **FocusVLA 局限 #2「对初始状态敏感」与 InSpire 的方向词 grounding 在方法学上正好可补**——方向词与机器人 frame 直接耦合，理论上正是初始状态扰动的天然解药

### 1.4 一句话研究问题

> 在 LIBERO-Plus 7 维 + VLA-Risk 6 维扰动下，**显式 spatial grounding (InSpire)** 与**隐式 attention 聚焦 (FocusVLA / OpenVLA-OFT)** 是否在不同扰动维度上**强项互补**？两者是否可**无成本叠加**获得进一步增益？方向词预测是否可作为**轻量 failure predictor**？

---

## § 2 · 相关工作（4 子主题广覆盖）

### 2.1 显式 spatial grounding 谱系

显式路径的核心思想是**在模型 output 层强制产生空间相关的中间表示**，以此对抗 shortcut。

**InSpire**（Zhang 等 2025，[arXiv:2505.13888](https://arxiv.org/abs/2505.13888)，UESTC Lianli Gao 课题组）是本项目主方法。在原 task instruction 前置 `"In which direction is the [target] relative to the robot?"`，强制 VLA 先输出方向词 ∈ {right, left, up, down, front, back, grasped}，再输出 action。损失为方向词 CE + action regression 联合。**零数据成本**——方向词 GT 可从 demo 自带的 robot/object pose 自动推导。详见 [[2025-05_InSpire_Zhang]]。

**SG-VLA**（Tu 等 2026，[arXiv:2603.22760](https://arxiv.org/abs/2603.22760)）⭐ web-new ⭐ 通过辅助 spatial decoder 在多任务监督下学习中间空间表示，不需要修改 prompt，是 InSpire 的"自动辅助任务版"。

**Gaze-Regularized VLA**（Pani 等 2026-03）通过 KL 散度对齐 VLA attention 与人类 eye-gaze 数据，强制模型看人会看的地方。代价是需要 human gaze 数据。

**AutoFocus-IL**（Gong 等 2025-11）用 VLM 生成的 saliency map 替代人类 gaze，更易扩展。

**TraceVLA**（Zheng 等 2024-12）把过去 N 步的轨迹画在图上作为 visual prompt，相当于把"运动历史"显式注入观察空间。

**新工作（web 搜到）**：

- **ACoT-VLA**（CVPR 2026） ⭐ web-new ⭐：Action Chain-of-Thought，让 VLA 先输出语言级的"动作意图"再输出底层 action。是 InSpire "方向词"的语言级泛化版。代码已开源 [github.com/AgibotTech/ACoT-VLA](https://github.com/AgibotTech/ACoT-VLA)。
- **GraphCoT-VLA**（[arXiv:2508.07650](https://arxiv.org/abs/2508.07650)） ⭐ web-new ⭐：3D 空间感知 CoT，把场景关系编码为图结构再做 CoT 推理。复杂度比 InSpire 高一档。
- **LaST**（[arXiv:2601.05248](https://arxiv.org/abs/2601.05248)） ⭐ web-new ⭐：Latent Spatio-Temporal CoT，在 latent 空间做时空 CoT，是显式 CoT 的"半隐式化"中间路径。
- **ATA**（[arXiv:2603.01490](https://arxiv.org/abs/2603.01490)） ⭐ web-new ⭐：Attention-guided 桥接 implicit 与 explicit reasoning，最接近本项目"对照两路"的思路，但只在单一 benchmark 上做。
- **Hierarchical Language-Action Alignment**（[arXiv:2604.05614](https://arxiv.org/abs/2604.05614)） ⭐ web-new ⭐：04-2026 最新工作，分层把语言指令对齐到 action（high-level intent → low-level command）。

**本项目定位**：以 InSpire 为代表的"方向词 plugin"是最轻量的显式 grounding，工程门槛 < 200 行代码（即使 InSpire 官方代码未释也可手写），是本项目优先选择。

### 2.2 隐式 attention 聚焦谱系

隐式路径的核心思想是**改造 attention 机制让模型在 feature 层自然聚焦**，无需 output 层显式监督。

**FocusVLA**（Zhang 等 2026，[arXiv:2603.28740](https://arxiv.org/abs/2603.28740)，HIT-Shenzhen × DaiMon × NJU × RUC）是导师推荐的导航星论文。核心是两个机制：
- **Modality Cascaded Attention**：把 VLA-Adapter 的并行 mixed attention 拆成串行的 3 次独立 cross-attention（$H_A \to H_{AQ} \to H_V$），切断 action query 抢占视觉细节的 shortcut
- **Focus Attention**：patch-level top-K (K=256) + channel-level element-wise gate

效果：0.5B 参数在 LIBERO 多权重 98.7%，**打败 7B 的 OpenVLA-OFT (98.5%)**；训练 1.5× 加速。详见 [[2026-03_FocusVLA_Zhang]]。

**OpenVLA-OFT**（Kim 等 2025-02）是 FocusVLA 在论文 § 2 明确批判的"shortcut 架构典型"——直接在 action 生成阶段省略视觉特征。但作为对照，**它正是"无显式 grounding + 简化 attention"的代表样本**，本项目用它作为 Focus-style 代理。

**EF-VLA**（Anon 2024）同样属于"省略视觉特征"路径，与 OpenVLA-OFT 同代。

**VLA-Adapter**（FocusVLA 的直接前驱）：使用 mixed parallel attention 同时融合视觉/语言/动作 query，是 FocusVLA 改进的起点。

**Spatial Forcing**（2025-11）：通过 3D 几何先验强制 attention 关注 spatial 关系，是另一条隐式聚焦路径。

**同代 token pruning 工作**：VLA-Pruner / VLA-IAP / SemanticVLA / Compressor-VLA——通过对 visual token 做剪枝实现"聚焦"，与 FocusVLA 思想类似但更激进。

**新工作（web 搜到）**：

- **Evo-0**（[arXiv:2507.00416](https://arxiv.org/abs/2507.00416)） ⭐ web-new ⭐：VLA with Implicit Spatial Understanding，是 FocusVLA 的"隐式 spatial"最近邻竞品。
- **Recurrent-Depth VLA**（[arXiv:2602.07845](https://arxiv.org/abs/2602.07845)） ⭐ web-new ⭐：通过 recurrent depth 实现 implicit test-time compute scaling，是"隐式推理"的另一条路。
- **DepthVLA**（[arXiv:2510.13375](https://arxiv.org/abs/2510.13375)） ⭐ web-new ⭐：显式 depth-aware spatial reasoning，与 InSpire 的方向词形成 3D 空间维度的补充。

**本项目定位**：不强行复现 FocusVLA 的 Cascaded Attention（工程风险高，详见 [[07_为什么不选其他方案]]），而是用 OpenVLA-OFT 作为"隐式 Focus-style"的合法代理——它是 FocusVLA 论文明确批判的对照样本，对照公平性可论证。

### 2.3 Shortcut learning 与 counterfactual 防御

**ShortcutLearning in Generalist Robot Policies**（Xing 等 2025-08 CoRL，UESTC，与 InSpire 同实验室）系统诊断了 VLA dataset fragmentation 与 spurious correlation 根源——这是给"为什么需要显式 grounding"提供理论依据的关键论文。详见 [[2025-08_ShortcutLearning_Xing]]。

**CF-VLA**（Peng 等 2025-12，NVIDIA + UCLA）通过 counterfactual self-reflective 训练消除 shortcut。与 InSpire 思想类似但实现复杂度高一档。

**本项目论证**：ShortcutLearning 论文实证表明 dataset 中存在大量 spurious correlation（"左手拿 = 蓝色背景"等）；显式方向词 grounding 通过给模型一个**与背景无关、与 robot frame 强耦合的标签**，理论上可以抑制 shortcut。这是 H1 互补假设的理论根基。

### 2.4 Failure detection 与运行时不确定性

**SAFE**（Kim 等 2025-06，[arXiv:2506.09937](https://arxiv.org/abs/2506.09937)） ⭐ web-new ⭐：Multitask Failure Detection for VLA，提出针对 VLA 的多任务失败检测框架。是 H3 的直接 baseline，必须引用。

**FPC-VLA**（Sciencedirect 2025） ⭐ web-new ⭐：Supervisor for Failure Prediction & Correction，比 SAFE 更进一步加入修正环节。

**I-FailSense**（[arXiv:2509.16072](https://arxiv.org/abs/2509.16072)） ⭐ web-new ⭐：VLM-based General Failure Detection，是 H3 的最新 baseline，2025-09 发布。

**Averaging Trap (Shifting Uncertainty to Critical Moments)**（[arXiv:2603.18342](https://arxiv.org/abs/2603.18342)） ⭐ web-new ⭐ **极重要** ⭐ ：该论文明确指出"LLM-style entropy 在 VLA failure detection 上有 Averaging Trap 问题"——连续 entropy 在长 episode 上被平均化，无法捕捉关键 moment 的不确定性。**而 InSpire 的方向词预测（离散 7-class 切换）正好是这个问题的天然解药**——离散切换比连续 entropy 更易锁定 critical moment。这给 H3 提供了**最强的理论辩护**，本项目必引。

**Altered Thoughts, Altered Actions**（[arXiv:2603.12717](https://arxiv.org/abs/2603.12717)） ⭐ web-new ⭐：反向论证 InSpire 类 CoT 也存在脆弱性——指令侧的微小扰动可以让方向词预测错乱继而引发 action error。本项目应将其作为 H3 的"安全 baseline"——讨论 InSpire 在 instruction perturbation 下的失败模式。

**RC-NF**（References/ 已读）：normalizing flow based anomaly detection，与本项目 H3 思路正交（基于 latent 分布 vs 基于显式方向词），可在 § Discussion 中比较。

### 2.5 评测 benchmark

**LIBERO**（Liu 等 2023，4 suites × 130 tasks）：基线评测平台
**LIBERO-Plus**（Fei 等 2025，7 维扰动）：主评测，详见 [[2025-10_LIBERO-Plus]]
**LIBERO-Para**（2026）：语言改写专项鲁棒性评测
**LIBERO-PRO**（2025）：generalization 评测
**VLA-Risk**（Wang 等 2026 ICLR，6 维扰动）：stretch goal 评测，详见 [[VLA-Risk_ICLR2026]]

**新工作（web 搜到）**：
- **RADAR**（[arXiv:2602.10980](https://arxiv.org/abs/2602.10980)） ⭐ web-new ⭐：Real-world dynamics benchmark
- **MultihopSpatial**（[arXiv:2603.18892](https://arxiv.org/abs/2603.18892)） ⭐ web-new ⭐：Multi-hop spatial reasoning benchmark
- **SpatialLadder**（[arXiv:2510.08531](https://arxiv.org/abs/2510.08531)） ⭐ web-new ⭐：Progressive spatial reasoning training benchmark

### 2.6 我们的定位

所有上述相关工作中，**没有任一论文同时**：
1. 在统一 backbone 上跑显式 + 隐式两条路径
2. 在 LIBERO-Plus 7 维 + VLA-Risk 6 维上做完整扰动评测
3. 把方向词预测作为 failure predictor（连接 § 2.4 的 Averaging Trap 理论）

本项目填补这三个交集。

---

## § 3 · 三个假设的形式化

### 3.1 H1（互补假设）形式化

**命题**：令 $M \in \{\text{vanilla}, +\text{InSpire}, +\text{Focus}, +\text{两者}\}$，$d \in \{\text{layout, viewpoint, state, instruction, light, texture, noise}\}$（LIBERO-Plus 7 维）。定义 **perturbation drop**：

$$\Delta_{M,d} = \text{SR}_{\text{clean}}(M) - \text{SR}_d(M)$$

**可证伪声明**：存在维度子集 $D_{\text{spatial}} = \{\text{viewpoint}, \text{state}\}$ 与 $D_{\text{visual}} = \{\text{texture}, \text{light}\}$，使得：

$$\Delta_{+\text{InSpire}, d \in D_{\text{spatial}}} < \Delta_{+\text{Focus}, d \in D_{\text{spatial}}} - \tau$$

$$\Delta_{+\text{Focus}, d \in D_{\text{visual}}} < \Delta_{+\text{InSpire}, d \in D_{\text{visual}}} - \tau$$

其中 $\tau = 2$pp，置信度 $p < 0.05$（配对 t-test，Bonferroni 校正，每组 ≥ 2 维度）。

**机制层面的辩护**：
- 显式方向词与 robot frame 直接耦合 → viewpoint / state 变化时方向词仍稳定
- 隐式 attention 聚焦在 feature 层提取 task-relevant patch → texture / distractor 时仍能保持注意力

**预期 effect size**：
- LIBERO-Plus 论文报告 7 维 std 约 4-7pp / dim
- 3 seed × 500 instance/cell，5pp 差异的 statistical power ≈ 0.85
- FocusVLA-style 在 cascaded 上报 +3.4pp，InSpire 在 OpenVLA-7B 上 rev2 报 4 task 平均 +5-8pp

**Fallback 策略**：
- 仅 1 维度显著 → 改叙事为「主要互补维度在 X」
- 完全重叠 → 改叙事为「显式 ≈ 隐式，两路 → 同一种 grounding」，反直觉发现仍可投 workshop

### 3.2 H2（叠加假设）形式化

**命题**：

$$\overline{\text{SR}}(+\text{两者}) > \max(\overline{\text{SR}}(+\text{InSpire}), \overline{\text{SR}}(+\text{Focus})) + 3\text{pp}$$

其中 $\overline{\text{SR}}(M) = \frac{1}{7} \sum_d \text{SR}_d(M)$ 是 LIBERO-Plus 7 维平均 SR。

**不要求超过和**（即不需要 $\overline{\text{SR}}_{\text{两者}} > \overline{\text{SR}}_{\text{InSpire}} + \overline{\text{SR}}_{\text{Focus}}$），只要求 **+3pp 阈值**——这是"非负超线性"，即两者叠加不会拖累任一方且能进一步增益。

**机制层面的辩护**：
- InSpire 在 output cls head + prompt 层注入信号
- Focus-style 在 attention feature 层改造
- 两者作用在 transformer 不同位置 → 理论上正交，无参数共享 → 无干扰

**3pp 阈值的选择依据**：
- LIBERO-Plus 论文中 best vs 2nd-best gap 通常 2-5pp
- 3pp 是"可发表"的最低增益 threshold（small effect at workshop level）

**Fallback 策略**：
- +0~+3pp：写"无显著叠加，但不冲突"——workshop 接受
- -3pp~0pp：写"两者在某些 task 上冲突"——本身是 finding
- < -3pp：彻底冲突 → 切叙事到 H1 单独 + H3 单独

### 3.3 H3（诊断假设）形式化

**命题**：定义每个 episode 的方向词预测时序特征向量

$$\mathbf{x}_e \in \mathbb{R}^k = [\bar{\text{acc}}_e, \text{var}(\text{acc}_e), \bar{\text{conf}}_e, \text{var}(\text{conf}_e), \text{flips}_e, ...]$$

最终成败 $y_e \in \{0, 1\}$。训练分类器 $f: \mathbf{x} \to [0,1]$ 预测最终成败，要求

$$\text{AUC}(f) \ge 0.65 \quad \text{(5-fold CV)}$$

**特征工程**：
- top-1 accuracy 时序均值 $\bar{\text{acc}}_e$
- top-1 accuracy 时序方差 $\text{var}(\text{acc}_e)$
- softmax confidence 时序均值 $\bar{\text{conf}}_e$
- softmax confidence 时序方差 $\text{var}(\text{conf}_e)$
- 方向词翻转次数 $\text{flips}_e$（连续步间标签变化次数）
- 抓取词 "grasped" 出现位置（episode 中的相对时间）

**Baseline 对比**：
- attention entropy-based predictor（基于 OpenVLA-OFT last-layer attention）
- 随机预测（AUC = 0.5）

**理论辩护（Averaging Trap）**：
InSpire 方向词预测作为 lightweight failure signal 是对 [Averaging Trap 论文](https://arxiv.org/abs/2603.18342)指出的 entropy averaging 问题的**离散化解药**——方向词的离散切换（discrete flip）比连续 entropy 更易锁定 critical moment。本项目实证检验这一理论预测。

**预期 effect size**：
- InSpire 论文未直接报这个数（H3 是本项目原创扩展）
- SAFE (2506.09937) 报 AUC 0.71（不同任务、不同 setup）
- 我们的离散方向词预测应在 0.65-0.75 range
- AUC < 0.65 → 改 "descriptive visualization" 卖点（attention map + 方向词时序的定性分析）

---

## § 4 · 实验设计

### 4.1 4 模型变体详细配置

| 变体 | Backbone | Prompt | Cls head | Loss | Note |
|---|---|---|---|---|---|
| **vanilla** | OpenVLA-OFT-7B + LoRA(r=16) | 原 LIBERO instruction | ❌ | action regression | 基线 |
| **+InSpire** | 同上 | 前置 "In which direction is the [target]..." | 7-class direction head | action reg + 0.1 × direction CE | 显式 grounding |
| **+Focus-style** | 同上 | 原 instruction | ❌ | action regression | 用 OpenVLA-OFT 默认配置作为 "隐式聚焦" 代理（详见 § 2.2） |
| **+InSpire & Focus** | 同上 | 前置方向词问题 | 7-class direction head | action reg + 0.1 × direction CE | 叠加 |

**LoRA 配置**：r=16，alpha=32，dropout=0.05，target_modules = q_proj, k_proj, v_proj, o_proj。显存预算 < 40GB（单卡 A100）。

**训练数据**：LIBERO-Spatial / Object / Goal / Long 4 suites，每 task 50 demo，共 ~26000 trajectory。LIBERO 4 suite 单卡 8h 完成（参考 OpenVLA-OFT 原论文）。

### 4.2 数据流 pipeline

```mermaid
graph LR
    A[LIBERO demos hdf5] -->|robot eef pose + object pose| B[方向词 GT 标注脚本]
    B -->|7-class label per step| C[训练 data]
    C --> D[4 变体 LoRA 训练]
    D --> E[checkpoint]
    E --> F[LIBERO-Plus 7 维评测]
    F --> G[28-cell SR 矩阵]
    G --> H[H1 互补归因 / H2 叠加检验 / H3 failure predictor]
```

**方向词 GT 标注脚本核心逻辑**（附录 A 给出完整伪代码）：
1. 读取 demo 每步 `observations/robot0_eef_pos` 与 `observations/object_pos`
2. 计算方向向量 `dir = object_pos - robot_eef_pos`（3D）
3. 找绝对值最大的轴 → 主方向（6 种）
4. 检查 grasped 状态（距离 < ε + gripper closed）→ 标 "grasped"
5. 滑动平均 3 步去抓取/释放瞬间的方向词跳变

### 4.3 评测协议

**主评测**：LIBERO-Plus 7 维 × 4 变体 = 28 cell。每 cell 跑 **3 seed × 500 instance** = 1500 episode。

**总成本估算**：28 × 1500 × 30s = ~117 GPU-h（单 A100，inference 速度参考 OpenVLA-OFT 报告）。

**统计方法**：
- 每 cell 报告 mean ± std（3 seed）
- 主对比用配对 t-test（同 instance 不同模型）
- 多重比较 Bonferroni 校正（family-wise error rate ≤ 0.05）
- 报告 effect size（Cohen's d）

**附加评测（stretch goal）**：
- LIBERO-Para 2 子集（语言改写）：补强 instruction 维度
- VLA-Risk 6 维子集（如代码 release）：附加 attention 评测

### 4.4 消融实验

| 消融项 | 扫描值 | 目的 |
|---|---|---|
| InSpire loss weight | {0.05, 0.1, 0.2, 0.5} | 找最优联训权重 |
| 方向词粒度 | 6 vs 8（加左前 / 右前） | 验证粒度对 SR 的影响 |
| 方向词位置 | prompt 前置 vs 后置 | 验证 prompt engineering 的鲁棒性 |
| LoRA r | {8, 16, 32} | 显存 vs 性能权衡 |

### 4.5 关键工程依赖

| 类别 | 依赖 | 状态 |
|---|---|---|
| 主 backbone | `openvla/openvla-7b-oft` (HF) | ✅ 已开源 |
| 备 backbone | `HuggingFaceTB/SmolVLA-base` | ✅ 已开源 |
| 模拟器 | robosuite + MuJoCo | ✅ 开源（LIBERO 自带） |
| 主评测 benchmark | LIBERO-Plus（`senyufei/LIBERO-Plus`） | ✅ HF 开源 |
| 鲁棒评测附加 | VLA-Risk | ⏳ 待 release（stretch） |
| InSpire 代码 | UESTC | ⏳ 待 release；200 行可手写 |
| FocusVLA 代码 | HIT-Shenzhen | ⏳ 待 release；OpenVLA-OFT 代理替代 |

---

## § 5 · 时间线与决策点

### 5.1 5 phase Gantt

| Phase | 月份 | 核心动作 | 预期产出物 |
|---|---|---|---|
| **P1 复现 InSpire on OpenVLA-OFT** | M1-M2 (06-07) | 环境搭建 + 方向词标注脚本 + InSpire plugin sanity check | OpenVLA-OFT clean baseline 数字 + 方向词预测准确率 > 50% |
| **P2 4 变体 × 7 维评测** | M3-M5 (08-10) | 28 cell 完整跑通 | 7×4 SR 矩阵 + 配对 t-test 结果 |
| **P3 互补性归因分析** | M6-M7 (11-12) | 维度分组 + 统计显著性 | H1 验证 + 雷达图 + 4 变体在每维度上的箱图 |
| **P4 Failure predictor** | M8-M9 (01-02) | 收集 2k episode + LightGBM | H3 验证 + AUC ≥ 0.65 |
| **P5 写作 + 投稿** | M10-M12 (03-05) | workshop short paper 撰写 + 投稿 | 至少 1 个 workshop 投出 |

### 5.2 4 个关键 Go/No-Go 节点

详细决策树见 [[05_风险与缓解]] § 3。简要 4 节点：

| 节点 | Go 条件 | No-Go 应对 |
|---|---|---|
| **M2 末** | 方向词准确率 > 50% & action SR ≥ vanilla - 10pp | < 30% → 切 [[02_备选方案_FocusVLA鲁棒性诊断]] |
| **M5 末** | H1 至少 1 维度强项不重叠 | 完全重叠 → 改"两路等价"叙事 |
| **M7 末** | 互补归因每组 ≥ 2pp，p<0.05 | 不显著 → 增 episode 数 / 换 baseline |
| **M9 末** | failure predictor AUC ≥ 0.65 | < 0.6 → 改描述性可视化 |

### 5.3 Plan A → Plan B 切换决策树

详见 [[05_风险与缓解]] § 3。

---

## § 6 · 与导师推荐叙事的对齐

导师明确推荐 FocusVLA 作为"导航星论文"。本项目对该论文作者自述的 **4 大局限**逐一回应：

| FocusVLA 自述局限 | 本项目如何回应 |
|---|---|
| **#1「鲁棒性对背景 + 纹理同时变化仍有限」** | § 4.4 消融中包含多扰动叠加专项；H1 的 $D_{\text{visual}}$ 组直击 |
| **#2「对初始状态敏感」** | H1 的 $D_{\text{spatial}}$ 组**核心切入点**——验证 InSpire 方向词是否能补这个洞 |
| **#3「VLM 内部的视觉利用没涉及」** | § 7 未来工作明确列入，但不在本项目主路径 |
| **#4「未在 robustness benchmark 上系统评测」** | § 4 整个评测设计的**核心贡献**——首次在 LIBERO-Plus 7 维 + VLA-Risk 6 维上量化 Focus-style 的鲁棒增益 |

**叙事一致性**：本项目不是"否定 FocusVLA"，而是"补齐 FocusVLA 自己留的洞 + 找一条同 backbone 不同思路的对照"。导师推荐的导航星定位被严格尊重。

---

## § 7 · 局限与未来工作

### 7.1 当前局限

1. **LIBERO 是仿真环境**——sim-to-real gap 未在本项目验证（团队无真机预算）
2. **方向词粒度粗**（6 + grasped = 7 类）——无法表达 "前左方 30°"等细粒度方向
3. **LIBERO-Plus 中 instruction 维度对所有 VLA 都不显著**（论文已报）——本项目无法在该维度区分变体
4. **OpenVLA-OFT 作为 "Focus-style" 代理是论证性合法但非 100% 复现**——若 FocusVLA 6 月放代码可升级
5. **2 人本科生团队 + ¥10K 预算限制了 GPU 时长 + episode 数**——单 cell 500 instance 在 noise / texture 等高方差维度可能 power 不足

### 7.2 未来工作

1. **嫁接 RobustVLA TRADES-mini 一致性损失**：把对抗训练的思想加入 InSpire+Focus 叠加路径（详见版本一 [[02_RobustVLA相关工作扩展]]）
2. **方向词作为 FocusVLA attention 的弱监督**：reverse use of InSpire——用方向词反向监督 attention 应该看的位置
3. **真机评测（SO-100 / Franka）**：若团队后续获得真机经费
4. **细粒度方向词**：扩展到 12 方向（含对角）或连续 azimuth+elevation 角度
5. **Adversarial perturbation vs natural perturbation 对比**：本项目以自然扰动为主，未来可加入 PGD 等对抗扰动作为 stress test

---

## § 8 · 投稿计划

### 8.1 投稿 venue 优先级

| Venue | 时间窗 | 适配度 |
|---|---|---|
| **CoRL 2027 workshop** on Robot Learning | 投稿 ~8-9 月 | ⭐⭐⭐⭐⭐ 主投 |
| **ICLR 2027 Robot Learning workshop** | 投稿 ~2-3 月 | ⭐⭐⭐⭐ 备投 |
| **NeurIPS 2027 Robot Learning workshop** | 投稿 ~9-10 月 | ⭐⭐⭐ 备投 |
| **国内会议（CCC / CAA）** | 3-5 月 deadline | ⭐⭐⭐ 保底 |

### 8.2 paper 章节与本文档映射

| Workshop paper §  | 本文档对应 § | 字数估计 |
|---|---|---|
| § 1 Intro | § 0 TL;DR + § 1 | 400 字 |
| § 2 Related | § 2 | 600 字（压缩 4 子主题） |
| § 3 Method | § 4.1 + § 4.2 | 800 字 |
| § 4 Experiment | § 4.3 + § 4.4 + 实测结果 | 1500 字 |
| § 5 Discussion | § 3 fallback + § 7 局限 | 500 字 |
| 参考文献 | § 9（精简到 20-25 篇） | — |

**总计 ~3800 字**，符合 workshop 4-6 页限制。

---

## § 9 · 参考文献

### 9.1 必读 5 篇（核心引用）

[1] Zhang, Y., Yuan, W., Zhang, Y., Zhang, X., Wan, J. (2026). FocusVLA: Focused Visual Utilization for Vision-Language-Action Models. *arXiv:2603.28740*.

[2] Zhang, J., Wu, S., Luo, X., Wu, H., Gao, L., Shen, H. T., Song, J. (2025). InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning. *arXiv:2505.13888*.

[3] Fei, S., et al. (2025). LIBERO-Plus: In-Depth Robustness Analysis of Vision-Language-Action Models. *arXiv:2510.13626*.

[4] Kim, M. J., et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*.

[5] Liu, B., Zhu, Y., Gao, C., et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. *arXiv:2306.03310*.

### 9.2 强参考（Plan A 设计依据）15 篇

[6] Xing, et al. (2025). Robotic Manipulation Generalizes Better via Bidirectional Cross-Modal Counterfactuals. *CoRL 2025*. arXiv:2508.06426.

[7] Peng, et al. (2025). CF-VLA: Counterfactual Self-Reflective Vision-Language-Action Models. *arXiv:2512.xxxxx*.

[8] Tu, et al. (2026). SG-VLA: Spatial Grounding for Vision-Language-Action Models. *arXiv:2603.22760*. ⭐ web-new

[9] Pani, et al. (2026). Gaze-Regularized Vision-Language-Action Models. *arXiv:2603.xxxxx*.

[10] Gong, et al. (2025). AutoFocus-IL: Salient-Aware Imitation Learning. *arXiv:2511.xxxxx*.

[11] Zheng, et al. (2024). TraceVLA: Visual Trace Prompting for Generalizable VLA. *arXiv:2412.10345*.

[12] AgibotTech (2026). ACoT-VLA: Action Chain-of-Thought for VLA. *CVPR 2026*. ⭐ web-new

[13] Anonymous (2025). GraphCoT-VLA: 3D Spatial-Aware Chain-of-Thought VLA. *arXiv:2508.07650*. ⭐ web-new

[14] Anonymous (2026). LaST: Latent Spatio-Temporal Chain-of-Thought. *arXiv:2601.05248*. ⭐ web-new

[15] Anonymous (2026). ATA: Attention-Guided Bridging Implicit and Explicit Reasoning in VLA. *arXiv:2603.01490*. ⭐ web-new

[16] Anonymous (2026). Hierarchical Language-Action Alignment in VLA. *arXiv:2604.05614*. ⭐ web-new

[17] Anonymous (2025). Evo-0: VLA with Implicit Spatial Understanding. *arXiv:2507.00416*. ⭐ web-new

[18] Anonymous (2026). Recurrent-Depth VLA: Implicit Test-Time Compute Scaling. *arXiv:2602.07845*. ⭐ web-new

[19] Anonymous (2025). DepthVLA: Depth-Aware Spatial Reasoning for VLA. *arXiv:2510.13375*. ⭐ web-new

[20] Kim, et al. (2025). OpenVLA-OFT: Optimized Fine-Tuning. *arXiv:2502.19645*.

### 9.3 Failure detection 与不确定性 5 篇（H3 关键）

[21] Kim, et al. (2025). SAFE: Multitask Failure Detection for Vision-Language-Action Models. *arXiv:2506.09937*. ⭐ web-new

[22] FPC-VLA: Supervisor for Failure Prediction and Correction. *Sciencedirect 2025*. ⭐ web-new

[23] Anonymous (2025). I-FailSense: VLM-based General Failure Detection. *arXiv:2509.16072*. ⭐ web-new

[24] Anonymous (2026). Shifting Uncertainty to Critical Moments: Reliable UQ for VLA. *arXiv:2603.18342*. ⭐ web-new ⭐ **理论辩护神器**

[25] Anonymous (2026). Altered Thoughts, Altered Actions: Probing CoT Vulnerabilities in VLA. *arXiv:2603.12717*. ⭐ web-new

### 9.4 评测 benchmark 5 篇

[26] Wang, et al. (2026). VLA-Risk: Multi-Modal Adversarial Robustness Benchmark for VLA. *ICLR 2026*.

[27] Anonymous (2026). RADAR: Real-World Dynamics Benchmark for VLA. *arXiv:2602.10980*. ⭐ web-new

[28] Anonymous (2026). MultihopSpatial: Multi-hop Spatial Reasoning Benchmark. *arXiv:2603.18892*. ⭐ web-new

[29] Anonymous (2025). SpatialLadder: Progressive Spatial Reasoning Training Benchmark. *arXiv:2510.08531*. ⭐ web-new

[30] LIBERO-Para Benchmark 2026. *arXiv:xxxx*.

### 9.5 基础方法 / 理论 5 篇

[31] Black, K., et al. (2024). π₀: A Vision-Language-Action Flow Model. *arXiv:2410.24164*.

[32] Shukor, M., et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. *arXiv:2506.01844*.

[33] Madry, A., et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR 2018*. arXiv:1706.06083.

[34] Yin, D., Kannan, R., Bartlett, P. (2019). Rademacher Complexity for Adversarially Robust Generalization. *ICML 2019*.

[35] Guo, J., et al. (2026). On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations (RobustVLA). *arXiv:2510.00037*.

---

## 附录 A · 方向词 GT 推导算法伪代码

```python
def annotate_directions(demo_trajectory):
    """
    输入：LIBERO demo (hdf5)，含 robot0_eef_pos 时序与 object_pos 时序
    输出：每步的方向词标签 ∈ {right, left, up, down, front, back, grasped}
    """
    direction_labels = []
    EPS_GRASP = 0.03  # 抓取距离阈值 (m)

    for t in range(demo_trajectory.length):
        eef_pos = demo_trajectory['observations/robot0_eef_pos'][t]    # (3,)
        object_pos = demo_trajectory['observations/object_pos'][t]      # (3,)
        gripper_closed = demo_trajectory['observations/gripper'][t] < 0.05

        # 1. 检查 grasped 状态
        dist = np.linalg.norm(eef_pos - object_pos)
        if dist < EPS_GRASP and gripper_closed:
            direction_labels.append('grasped')
            continue

        # 2. 计算方向向量并取主轴
        dir_vec = object_pos - eef_pos      # 物体相对 eef 的方向
        abs_dir = np.abs(dir_vec)
        main_axis = np.argmax(abs_dir)      # 0=x, 1=y, 2=z

        # 3. 映射到 6 方向（robot frame 约定）
        if main_axis == 0:
            label = 'right' if dir_vec[0] > 0 else 'left'
        elif main_axis == 1:
            label = 'front' if dir_vec[1] > 0 else 'back'
        else:
            label = 'up' if dir_vec[2] > 0 else 'down'

        direction_labels.append(label)

    # 4. 滑动平均 3 步（多数表决）去抖动
    smoothed = sliding_majority_vote(direction_labels, window=3)
    return smoothed
```

**注**：此伪代码基于 LIBERO 的 robot frame 约定（x = right, y = forward, z = up）。不同模拟器需调整轴映射。

## 附录 B · 4 变体 prompt 模板对比表

```
[vanilla]
"In: A image of a robot arm tasked with picking up the [target].
Task: {task_instruction}.
Out: Action: [action_tokens]"

[+InSpire]
"In: A image of a robot arm tasked with picking up the [target].
Question: In which direction is the [target] relative to the robot?
  Options: right, left, up, down, front, back, grasped.
Answer: {direction_word}.
Task: {task_instruction}.
Out: Action: [action_tokens]"

[+Focus-style]
（与 vanilla 同 prompt，仅 backbone 配置不同——OpenVLA-OFT 默认就是 Focus-style 代理）

[+InSpire & Focus]
（与 +InSpire 同 prompt，backbone 同 +Focus-style）
```

## 附录 C · 全部 28 cell 实验运行检查清单

| Cell | 变体 | 维度 | 状态 | 备注 |
|---|---|---|---|---|
| C01 | vanilla | layout | ⏳ |  |
| C02 | vanilla | viewpoint | ⏳ |  |
| C03 | vanilla | state | ⏳ |  |
| C04 | vanilla | instruction | ⏳ |  |
| C05 | vanilla | light | ⏳ |  |
| C06 | vanilla | texture | ⏳ |  |
| C07 | vanilla | noise | ⏳ |  |
| C08 | +InSpire | layout | ⏳ |  |
| C09 | +InSpire | viewpoint | ⏳ |  |
| ... | ... | ... | ... |  |
| C28 | +InSpire & Focus | noise | ⏳ |  |

完整 28 行清单在 P1 末期填充。每 cell 需记录：3 seed SR、std、与 vanilla baseline 的 paired t-test p 值。

---

## 关联文档

- [[INDEX]] — 立项调研目录索引
- [[00_总览与立项决定]] — 立项决定与项目命名
- [[01_主方案_FocusVLA_InSpire互补]] — Plan A 完整设计（本文档的"短版本"）
- [[02_备选方案_FocusVLA鲁棒性诊断]] — Plan B 兜底
- [[03_第一步两周任务清单]] — W1-W2 执行
- [[04_团队分工与预算]] — 2 人分工 + ¥10K
- [[05_风险与缓解]] — Top 3 风险 + 切换决策树
- [[06_关联资料索引]] — 论文笔记导航
- [[07_为什么不选其他方案]] — 决策记录
- **本文档（08）** — 整合 + 深化 + 17 篇新论文综述

%% Updated: 2026-05-26 %%
