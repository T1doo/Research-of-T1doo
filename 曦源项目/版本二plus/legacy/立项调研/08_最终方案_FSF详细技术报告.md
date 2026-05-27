---
create time: 2026-05-27T16:50:00
tags:
  - 曦源项目
  - 版本二plus
  - 最终方案
  - FSF
  - 完整技术报告
status: 工作版v1
audience: 项目组 + 师哥 + workshop paper 撰写主参考
---

# 08 · 最终方案 · FocusedSpatialForcing 详细技术报告

> [!info] 文档定位
> 本文档是 v2plus **完整技术工作版**，含数学命题、effect size 估计与广泛文献综述。**不适合直接抄入曦源申请书**——申请书请参考 [[递交材料/1-2.申请书填写草稿]]，对标 v2 申请书简洁版。两份文档承担**完全不同的角色**，互斥使用。
> 
> 本文档整合 [[00_总览与立项决定]] ~ [[07_为什么不选其他方案]] 9 个立项调研文件的核心内容 + [[09_与版本二差异化说明]]，并通过 web 搜索补充 31 篇引用论文（详见 [[06_关联资料索引]]），形成 6 个月后写 workshop short paper §1-§5 的素材库。

## § 0 · TL;DR

VLA 模型在标准 LIBERO clean SR 已达 95-99%，但在 LIBERO-Plus 等扰动评测下断崖式下跌至 30% 上下。**版本二**已确认 OpenVLA-OFT + InSpire 显式方向词提示插件路线可行（师哥确认）。**版本二plus**在此基础上新增 **Focused Spatial Forcing (FSF) 通道**——以冻结的 FastVGGT 为 teacher，对 OpenVLA-OFT Layer 24 视觉 token 做余弦相似度对齐；对齐 loss 的 per-token 权重由 **方向词 GT × DINOv2 CLS attention 双源融合的 focus mask**（**v2 纯净路径，不引入 SAM2**）调制（前景 1.0 / 上下文 0.5 / 背景 0.1），对应师哥提示的"focus 在重要位置，背景监督可以弱点"。本项目以 OpenVLA-OFT-7B 为统一主干，训练 **3 个模型变体**（V1: vanilla / V2: +InSpire / V3: +InSpire+FSF），在 LIBERO-Plus 7 维扰动 + 真机 3 维扰动两个轴下验证 **5 个核心假设**：H1 仿真鲁棒性增益、H2 InSpire×FSF 叠加效应、H3 failure predictor（继承 v2）、H4 真机鲁棒性增益、H5 双源 focus mask 优于单源/均匀。预期产出 1 篇 4-6 页 workshop short paper（CoRL/ICLR/NeurIPS Robot Learning workshop）+ 开源 plugin 代码 + 方向词标注/Focus mask/VGGT 缓存/真机数据 4 个 HuggingFace 数据集。

## § 1 · 问题陈述与研究空白

### 1.1 VLA 鲁棒性现状

VLA 模型在 LIBERO 4 个 task suite 上的 clean SR 普遍达 95-99%（FocusVLA 报 98.7%，OpenVLA-OFT 报 98.5%，Spatial Forcing 报 98.5%，VLA-Adapter-Pro 报 98.5%）。但 **LIBERO-Plus**（Fei 等 2025，[arXiv:2510.13626](https://arxiv.org/abs/2510.13626)）通过 7 维扰动（layout / viewpoint / state / instruction / light / texture / noise）系统评测后发现：**主流 VLA 在 viewpoint 与 initial state 扰动上的 SR 普遍从 95%+ 跌至 30% 上下**——一个 60pp 的鸿沟。

**这是 VLA 走出实验室、走入家用/医疗辅助等真实场景的最大障碍。**

### 1.2 三类技术路径的对比

学界 2024-2026 年涌现的 VLA 鲁棒性增强方法可大致归为三类。下表对比 9 个代表工作的"信号注入位置"与"工程开销"：

| 方法 | 路径 | 信号注入位置 | 需要新数据 | 工程开销 | 推理 latency |
|---|---|---|---|---|---|
| **InSpire** (2505.13888) | **显式 output** | Prompt + cls head | ❌（自动标注） | 极低 | +1 token decode |
| SG-VLA (2603.22760) | 显式 output | 辅助 spatial decoder | ❌ | 中 | 0 |
| Gaze-Reg VLA (2603.xxxx) | 显式 attention | KL alignment to gaze | 需 gaze data | 中 | 0 |
| AutoFocus-IL (2511.xxxx) | 显式 attention | VLM saliency → attention | ❌ | 中 | 0 |
| **FocusVLA** (2603.28740) | 隐式 attention | Cascaded + Focus attention | ❌ | **高**（架构改造） | 0 |
| **OpenVLA-OFT** (2502) | 隐式 architecture | 省略视觉特征 shortcut | ❌ | 低（架构选择） | 0 |
| **Spatial Forcing** (2510.12276) | **隐式 encoder align** | VGGT teacher 中间层对齐 | ❌ | **低**（~30 行） | 0 |
| **FSF（本项目）** | **InSpire + Spatial Forcing + Focus Mask 三层叠加** | output + encoder + mask 调制 | ❌（继承 v2 + DINOv2 自动；不依赖 SAM2） | 中 | +1 token decode |
| FocusVLA-VGGT | 隐式 encoder concat | policy 阶段并行 encoder | ❌ | 中（FocusVLA 自述"梯度弱"） | 0 |

**三路差异本质**：
- **显式路径**通过强制模型在 output 层产生中间表示（方向词、辅助标签）来纠正决策
- **隐式 attention 路径**通过架构改造让 attention 在 feature 层自然聚焦
- **隐式 encoder align 路径**通过冻结 teacher 监督让 student feature 学到 3D 信息（**SF 与 v2plus FSF 都属此类**）

### 1.3 5 个研究空白

通过对 101 篇精读笔记（[[reference-paper-notes]]）与 31 篇引用论文（详见 [[06_关联资料索引]]）的整理，识别出 5 个明确空白：

1. **显式方向词提示插件缺乏多维扰动下的系统验证 + 真机验证**（v2 已部分填补，v2plus 补足真机）
2. **显式提示对模型视觉利用机制的影响缺乏实证**（v2 已部分填补，v2plus 继承）
3. **隐式 3D 表征对齐（Spatial Forcing）尚未与显式空间提示（InSpire）叠加验证**——本项目 H2 填补
4. **3D 监督的 focus mask 来源融合策略尚未系统消融**——本项目 H5 填补
5. **隐式 3D 对齐在真机扰动下的鲁棒性尚未验证**（SF 原论文仅 RoboTwin 测试且无系统扰动）——本项目 H4 填补

### 1.4 一句话研究问题

> 在 LIBERO-Plus 7 维扰动 + 真机 3 维扰动下，**显式方向词提示插件（InSpire）与聚焦式 3D 表征对齐（Focused Spatial Forcing）叠加**，能否进一步系统提升 VLA 模型的扰动鲁棒性？**双源融合的 focus mask 是否优于单源或均匀监督？**

## § 2 · 相关工作（5 子主题广覆盖）

### 2.1 显式 spatial grounding 谱系

详见 v2 文档 [[../../版本二/立项调研/08_最终方案_FocusVLA_InSpire互补研究]] § 2.1。本项目 V2/V3 共用 InSpire 路线，方法学完整复用。新增的相关工作：
- **ACoT-VLA** (CVPR 2026)、**GraphCoT-VLA** (2508.07650)、**LaST** (2601.05248)、**ATA** (2603.01490)、**Hierarchical Language-Action Alignment** (2604.05614)

### 2.2 隐式 3D 表征对齐谱系（v2plus 核心新增）

**Spatial Forcing**（Li 等 2025，[arXiv:2510.12276](https://arxiv.org/abs/2510.12276)，OpenHelix Team）是本项目 FSF 通道的直接父本。详见 [[研究报告/01_Spatial_Forcing_深度精读]]。核心：
- VLA 中间层视觉 token 与冻结 VGGT 的对应 token 做余弦相似度对齐
- Loss: $\mathcal{L}_{align} = -\frac{1}{N}\sum_i \cos(\text{MLP}(\Gamma(x_i^V)), f_i^{3D}+E)$, $\alpha=0.5$
- Layer 24 是消融最优层
- 训练加速 3.8×；数据效率 5.9×

**FocusVLA 中的 VGGT 用法对比**（详见 [[研究报告/03_FocusVLA_中_VGGT_用法剖析]]）：
- FocusVLA：VGGT 作 policy 阶段并行 encoder；作者自述"梯度弱不稳定，性能上限受限"
- Spatial Forcing：VGGT 作 Teacher 在中间层对齐；性能充分（+1.4pp）
- **v2plus 采纳 SF 优等做法 + 增加 focus mask 调制**

**3D-aware VLA 全景**（详见 [[研究报告/06_3D-aware_VLA_相关工作综述]]）：
- 显式 3D 输入路线：PointVLA、SpatialVLA (2501.15830)
- 多视图 3D 表征：GP3 (2509.15733)、VIPA-VLA (2512.13080)
- 隐式 3D 对齐：**Spatial Forcing**（本项目父本）、Evo-0 (2507.00416)、Recurrent-Depth VLA (2602.07845)
- 显式深度感知：DepthVLA (2510.13375)、QDepth-VLA (2510.14836)、AugVLA-3D (2602.10698)
- 多 token 互补：FocusVLA、ST-VLA (2603.13788)、GraphCoT-VLA

### 2.3 VGGT 与续作生态（详见 [[研究报告/02_VGGT_及后续工作综述]]）

VGGT（CVPR 2025 Best Paper，[arXiv:2503.11651](https://arxiv.org/abs/2503.11651)）后已有 12+ 个续作：
- **FastVGGT**（2509.02560，ICLR 2026）—— **本项目主选 teacher**，4× 加速训练无关
- StreamVGGT (2507.11539)、π³ (2507.13347, ICLR 2026)、HD-VGGT (2603.27222)
- VGGT-Long (2507.16443)、InfiniteVGGT (2601.02281)、IncVGGT、FrameVGGT、XStreamVGGT、Evict3R
- VGGT-DP (2509.18778)、3D-Mix for VLA (2603.24393)、SceneVGGT (2602.15899)、AugVLA-3D (2602.10698)

### 2.4 Focus mask 与 attention guidance 谱系

详见 [[研究报告/04_Focus_Mask_设计调研]]。本项目 Focus mask 3 个来源的相关工作：
- Source A（方向词 GT）→ 继承 v2 InSpire 工具
- Source B（SAM2/GroundedSAM）→ **不引入**（v2 纯净路径；P3 stretch ablation 可选）
- Source C（DINOv2 CLS attention）→ AttentionVoxel (2509.20579) 直接证明此 saliency 思路 + Gaze-Reg VLA + AutoFocus-IL

### 2.5 Shortcut Learning、Failure Detection 与 Benchmark

详见 [[06_关联资料索引]] § 4-5。核心：
- ShortcutLearning in VLA (2508.06426) — VLA 鲁棒性根源
- CF-VLA、RobustVLA (2510.00037) — counterfactual 防御
- SAFE (2506.09937) — H3 直接 baseline
- Averaging Trap (2603.18342) — H3 理论辩护（极重要）
- I-FailSense、FPC-VLA、VLA-Risk (ICLR 2026)

## § 3 · 5 个核心假设的形式化

完整公式与 fallback 详见 [[01_主方案_FocusedSpatialForcing]] § 6。本节简述：

### H1（仿真鲁棒性增益，继承 v2 但更强版）

$$\Delta_{V3, d} < \Delta_{V1, d} - 5\text{pp}, \quad p < 0.05$$ 

在至少 2 个扰动维度上（预期 viewpoint 与 state）。

### H2（InSpire×FSF 叠加效应）

$$\text{SR}_{V3} > \max(\text{SR}_{V2}, \text{SR}_{+\text{FSF only}}) + 1.5\text{pp}$$

紧约束下用 A3 中 M-None（=SF 原版均匀监督）作为 +FSF only 代理。

### H3（Failure Predictor，继承 v2）

$$\text{AUC}(f) \ge 0.65 \quad \text{(5-fold CV)}$$

### H4（真机鲁棒性增益，v2plus 新增）

$$\overline{\Delta_{V3}^{real}} < \overline{\Delta_{V1}^{real}} - 5\text{pp}, \quad p < 0.10$$

### H5（双源 focus mask 优于单源/均匀，v2plus 核心创新）

$$\text{SR}_{M\text{-}ABC} > \max(\text{SR}_{M\text{-}A}, \text{SR}_{M\text{-}B}, \text{SR}_{M\text{-}C}) + 1\text{pp}$$
$$\text{SR}_{M\text{-}ABC} > \text{SR}_{M\text{-}None} + 1.5\text{pp}, \quad p < 0.05$$

## § 4 · 实验设计

### 4.1 模型变体与 ablation 配置（紧约束 3 变体）

| 变体 | Backbone | Prompt | Cls head | Loss | Note |
|---|---|---|---|---|---|
| **V1: vanilla** | OpenVLA-OFT-7B + LoRA(r=16) | 原 LIBERO instruction | ❌ | $\mathcal{L}_{action}$ | 基线 |
| **V2: +InSpire** | 同上 | 前置方向词 prompt | 7-class direction head | $+0.1\mathcal{L}_{direction}$ | 继承 v2 |
| **V3: +InSpire+FSF** (主创新) | 同上 + **FSF projector** | 同 V2 | 同 V2 | $+0.5\mathcal{L}_{FSF}$ | **v2plus 主创新** |

**LoRA 配置**：r=16，alpha=32，dropout=0.05，target_modules = q_proj, k_proj, v_proj, o_proj。显存 < 40GB。

**训练数据**：LIBERO 4 suites，~26000 trajectory。单卡 8-12h / variant / seed。

### 4.2 视觉利用 ablation（推理时，继承 v2）

| Ablation | 操作 | Setup | 目的 |
|---|---|---|---|
| **Token Dropout** | 推理时随机 mask N% visual tokens | N ∈ {10,30,50} × 3 变体 = 9 组 | SR 退化曲线对 V1/V2/V3 差异 |
| **Patch Mask** | 用方向词 GT 倒推 task-relevant patch 定向 mask | 3 变体 × 2 maskpattern = 6 组 | task-relevant patch 依赖度 |
| **Attention 可视化** | 对比 V1/V2/V3 在同一 episode 上 attention 分布 | 定性 + 量化 entropy | 探究 FSF 是否改变 attention |

### 4.3 评测协议（紧约束版）

**LIBERO-Plus 主评测**：7 维 × 3 变体 = 21 cell；每 cell 3 seed × 300 instance = 900 episode（紧约束 500→300）；总 18,900 episode；**158 GPU-h**。

**真机评测**：3 任务 × 3 扰动 + 3 clean = 12 cell × 30 episode = 360 episode；本地 RTX 4090；不进云 GPU 账户。

**Ablation 总 GPU-h**：A1-A3 必做 70 h、A4-A5 粗扫 10 h、A6 视觉利用 25 h；A7-A8 stretch。

**统计方法**：
- 每 cell 报告 mean ± std（3 seed）
- 主对比用配对 t-test（同 instance 不同模型）
- 多重比较 Bonferroni 校正（family-wise error rate ≤ 0.05）
- 报告 effect size（Cohen's d）

### 4.4 关键工程依赖

| 类别 | 依赖 | 状态 |
|---|---|---|
| 主 backbone | `openvla/openvla-7b-oft` (HF) | ✅ 已开源 |
| **3D teacher** | **`mystorm/FastVGGT` (HF)** | ✅ **已开源** |
| 模拟器 | robosuite + MuJoCo | ✅ 开源（LIBERO 自带） |
| 主评测 benchmark | LIBERO-Plus (`senyufei/LIBERO-Plus`) | ✅ HF 开源 |
| Focus mask 工具 | **DINOv2**（已在 OpenVLA-OFT 中）+ 方向词 GT 投影（v2 复用）| ✅ 全部开源；**不依赖 SAM2** |
| 真机平台 | LeRobot SO-100 / Franka（实验室已有） | ✅ 用户确认 |
| InSpire 代码 | UESTC | ⏳ 待 release；200 行可手写 |
| Spatial Forcing 代码 | OpenHelix Team | ✅ 已开源 (github.com/OpenHelix-Team/Spatial-Forcing) |

## § 5 · 时间线与决策点（详见 [[03_第一步两周任务清单]]、[[05_风险与缓解]]）

### 5.1 6 phase Gantt 简表

| Phase | 月份 | 核心动作 | 预期产出物 |
|---|---|---|---|
| **P0 环境与真机准备** | M0 (06-初) | 4 仓库装通 + FastVGGT sanity + 真机平台 + 方向词标注 + Focus mask pipeline | N0 决策达标 |
| **P1 复现 + Sanity** | M1-M3 (06-08) | VGGT 离线缓存 + V1/V2/V3 LoRA × 3 seed + 真机 150 demo 采集 | clean SR 矩阵 + 方向词准确率 > 60% |
| **P2 主评测 + 真机标注** | M4-M6 (09-11) | 21 cell × 3 seed × 300 inst LIBERO-Plus + A1/A2/A3 + 真机标注 | H1 + H5 验证 |
| **P3 机制分析 + 真机评测** | M7-M9 (12-02) | A6 视觉利用 ablation + A4/A5 粗扫 + 真机 360 episode 评测 | H4 验证 + attention 可视化 |
| **P4 Failure predictor + 写作** | M10-M11 (03-04) | LightGBM + §1-§3 论文初稿 | H3 验证 + AUC ≥ 0.65 |
| **P5 投稿冲刺** | M12 (05) | §4 + §5 + 全文 5-6 页 + 投稿 | 至少 1 个 workshop 投出 |

### 5.2 5 个 Go/No-Go 节点决策准则

详见 [[05_风险与缓解]] § 3。

| 节点 | Go 条件 | No-Go 应对 |
|---|---|---|
| **N0** (M0 末) | 4 仓库装通 + FastVGGT mIoU ≥ 0.5 + 真机就绪 | 切 v2 only / VGGT 原版 fallback |
| **N1** (M3 末) | V3 clean SR ≥ V1 - 3pp & VGGT 缓存完整 | V3 退化 > 10pp → 切 v2 only |
| **N2** (M6 末) | H1 ≥ 2 维度显著 & H5 M-AC > M-None | 完全无增益 → negative finding workshop |
| **N3** (M9 末) | H4 ≥ 5pp & 视觉 ablation 显示 V3 对前景 patch 依赖更高 | 退化 → 真机降级为 case study |
| **N4** (M11 末) | AUC ≥ 0.65 & 论文数据齐全 | 不齐 → 转 ICLR/CCC（更晚 deadline） |

## § 6 · 与师哥推荐叙事的对齐

师哥明确推荐"沿 FocusVLA 方向深挖鲁棒性 + 加 3D + 参考 Spatial Forcing + focus 在重要位置背景弱 + VGGT 续作借鉴"。详细对应见 [[09_与版本二差异化说明]] § 6。本节简述：

| 师哥语言 | v2plus 对应 |
|---|---|
| "加 3D 是好想法" | FSF 通道 + FastVGGT teacher + Layer 24 对齐 |
| "按 InSpire 的会，要在真机上验证" | 3 任务 × 50 demo × 3 扰动维度 = 360 episode 真机评测 |
| "参考 arXiv:2510.12276 隐式 3D 表征对齐" | FSF loss 直接采用 SF 论文公式 |
| "FocusVLA 也用了 VGGT，看做法是否有区别" | 详述 SF（中间层对齐优等）vs FocusVLA（policy 并行劣等） |
| "Spatial forcing 但需要 focus 在重要位置，背景可弱点" | 双源融合 focus mask（前景 1.0 / 上下文 0.5 / 背景 0.1） |
| "VGGT 续作可以大致看下" | FastVGGT 主选 + π³/StreamVGGT/HD-VGGT 续作综述 |

## § 7 · 局限与未来工作

### 7.1 当前局限

1. **LIBERO 是仿真环境** + 实验室真机 3 任务，sim-to-real gap 未完整验证
2. **方向词粒度粗**（6 + grasped = 7 类）——无法表达 "前左方 30°" 等细粒度
3. **LIBERO-Plus 中 instruction 维度对所有 VLA 都不显著**（论文已报）
4. **FastVGGT 与 VGGT 原版的精度差异**：在 LIBERO 合成图上需验证
5. **真机数据 50 demo × 3 任务 = 150 demo 偏少**，fine-tune 风险
6. **2 人本科生团队 + ¥10K 预算限制了 GPU 时长 + episode 数**

### 7.2 未来工作

1. **Multi-layer SF**：扫描 OpenVLA-OFT Layer {8, 16, 24, 32}，研究多层 SF 联合监督
2. **真机 scope 扩大**：5+ 任务 × 100+ demo × 5+ 扰动维度
3. **跨 backbone 验证**：在 SmolVLA、π0、π0.5 上验证 FSF 通用性
4. **细粒度方向词**：扩展到 12 方向（含对角）或连续 azimuth+elevation
5. **真实 RGB-D 输入 + 联合训练**：FSF 与 SpatialVLA 路线融合
6. **VGGT 续作 ablation**：FastVGGT vs π³ vs HD-VGGT 作 teacher 的对比

## § 8 · 投稿计划与 paper 章节映射

### 8.1 投稿 venue 优先级

| Venue | 时间窗 | 适配度 |
|---|---|---|
| **CoRL 2027 workshop** on Robot Learning | 投稿 ~8-9 月 | ⭐⭐⭐⭐⭐ 主投 |
| **ICLR 2027 Robot Learning workshop** | 投稿 ~2-3 月 | ⭐⭐⭐⭐ 备投（赶上 M12 deadline） |
| **NeurIPS 2027 Robot Learning workshop** | 投稿 ~9-10 月 | ⭐⭐⭐ 备投 |
| **CoRL 2027 main 短稿** | 同 CoRL workshop | ⭐⭐⭐ delight goal（H1+H4+H5 全成立时冲） |
| 国内会议（CCC / CAA / NCRR） | 3-5 月 deadline | ⭐⭐⭐ 保底 |

### 8.2 Paper 章节与本文档映射

| Paper § | 字数 | 对应本文档 §  | 内容要点 |
|---|---|---|---|
| § 1 Intro | 500 | § 0 TL;DR + § 1 | (1) VLA 鲁棒性 30% 鸿沟；(2) 显式 vs 隐式 grounding；(3) 3D 监督的必要性；(4) focus 思想；(5) 5 个核心假设 |
| § 2 Related | 700 | § 2 + [[06_关联资料索引]] | (1) 3D-aware VLA；(2) Focus/attention guidance；(3) Explicit grounding；(4) Benchmark |
| § 3 Method | 1000 | § 4.1 + [[01_主方案_FocusedSpatialForcing]] | (1) 架构图；(2) Loss 公式；(3) Focus mask 双源融合；(4) VGGT 缓存；(5) 训练协议 |
| § 4 Experiment | 1500 | § 4.3 + § 4.4 + ablation | (1) Setup（LIBERO-Plus + 真机）；(2) 主结果 21 cell；(3) 真机 360 episode；(4) Ablation A1-A6；(5) H3 failure predictor |
| § 5 Discussion | 500 | § 7 + [[05_风险与缓解]] | (1) FSF 增益机制讨论；(2) 局限；(3) 未来工作 |
| 参考文献 | — | [[06_关联资料索引]] 精简到 25-30 篇 | — |

**总计 ~4200 字**，符合 workshop 4-6 页限制。

## § 9 · 参考文献（精简版，详见 [[06_关联资料索引]]）

### 9.1 必读 5 篇（核心引用）

1. Li, F., et al. (2025). Spatial Forcing. arXiv:2510.12276. **FSF 直接父本**
2. Anonymous (2025). FastVGGT. arXiv:2509.02560. **主选 teacher**
3. Wang, J., et al. (2025). VGGT. CVPR 2025. arXiv:2503.11651.
4. Zhang, Y., et al. (2026). FocusVLA. arXiv:2603.28740. **关键对比**
5. Zhang, J., et al. (2025). InSpire. arXiv:2505.13888. **v2 父本**

### 9.2 强参考 10 篇（详见 [[06_关联资料索引]] § 2）

OpenVLA-OFT、LIBERO-Plus、LIBERO、StreamVGGT、π³、VGGT-DP、3D-Mix for VLA、AttentionVoxel、AugVLA-3D、DepthVLA

### 9.3 Focus mask & attention guidance 8 篇（详见 [[06_关联资料索引]] § 3）

Gaze-Regularized VLA、AutoFocus-IL、Grounded SAM、SAM2、DINOv2、GroundingDINO、SAM2Act、SlotVLA

### 9.4 Failure Detection & Benchmark 5 篇（详见 [[06_关联资料索引]] § 4）

SAFE、Averaging Trap、FPC-VLA、I-FailSense、VLA-Risk

### 9.5 Shortcut Learning & Robustness 3 篇（详见 [[06_关联资料索引]] § 5）

ShortcutLearning in VLA、CF-VLA、RobustVLA

## 附录 A · 方向词 GT 推导算法伪代码（继承 v2，FSF Source A 复用）

详见 [[03_第一步两周任务清单]] 附录 A。

## 附录 B · FSF Projector 模块代码骨架（~30 行）

详见 [[01_主方案_FocusedSpatialForcing]] § 4.2。

## 附录 C · Focus Mask 双源融合伪代码（v2 纯净路径）

详见 [[01_主方案_FocusedSpatialForcing]] § 3 与 [[研究报告/04_Focus_Mask_设计调研]]。

## 附录 D · 真机数据采集与标注协议

详见 [[研究报告/05_真机平台与数据标注方案]]。

## 附录 E · 21 + 12 = 33 主评测 Cell 检查清单

### LIBERO-Plus（21 cell）

| Cell | 变体 | 维度 | 数量 | 状态 |
|---|---|---|---|---|
| C01-C07 | V1 (vanilla) | layout/viewpoint/state/instruction/light/texture/noise | 7 | ⏳ |
| C08-C14 | V2 (+InSpire) | 同 7 维 | 7 | ⏳ |
| C15-C21 | V3 (+InSpire+FSF) | 同 7 维 | 7 | ⏳ |

每 cell：3 seed × 300 instance = 900 episode。每 cell 记录 (mean ± std, paired t-test p vs V1, Cohen's d)。

### 真机（12 cell = 3 任务 × (clean + 3 扰动)）

| Task | Clean | D1 光照 | D2 背景 | D3 初始位置 |
|---|---|---|---|---|
| T1 Pick-place block | C-T1 | C-T1-D1 | C-T1-D2 | C-T1-D3 |
| T2 Push block | C-T2 | C-T2-D1 | C-T2-D2 | C-T2-D3 |
| T3 Pick-place mug occluded | C-T3 | C-T3-D1 | C-T3-D2 | C-T3-D3 |

每 cell 30 episode（V1/V2/V3 各跑），共 12 × 3 × 30 = 1080 episode；约 18 hr 真机 + 标注 2 天，总 1 周完成。

%% Updated: 2026-05-27 %%
