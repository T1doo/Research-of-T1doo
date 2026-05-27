---
create time: 2026-05-26T22:10:00
tags:
  - 曦源项目
  - 版本三
  - 主方案
  - SAM-as-Probe
  - FocusVLA
  - GroundedSAM
  - SAM2
  - Plan_A
---
# 01 · 主方案 · SAM-as-Probe 纯诊断式实证研究（Plan A）

> **一句话定位**：以 SAM2/GroundedSAM 物体 mask 作为「外部视觉真值」，对主流 VLA（OpenVLA-OFT-7B / SmolVLA-0.5B）在 LIBERO clean + LIBERO-Plus 7 维扰动下做"视觉聚焦质量 × 扰动鲁棒性"的**纯诊断式实证研究**——不训练 VLA，把 SAM 当探针，所有"变长 token / 物体分割"想法收敛到评测侧。

## 1. 问题陈述

### 1.1 研究背景

VLA（Vision-Language-Action）模型在 LIBERO clean 上已达 95-99% 成功率（FocusVLA 98.7%、OpenVLA-OFT 98.5%、VLA-Adapter-Pro 98.5%）。但 LIBERO-Plus（Fei et al. 2025，arXiv:2510.13626）通过 7 维扰动（layout / viewpoint / state / instruction / light / texture / noise）系统评测后发现：**主流 VLA 在 viewpoint 与 initial state 扰动上的 SR 普遍从 95%+ 跌至 30% 上下**——60pp 鸿沟。

学界对这种"鸿沟"提出了一类有影响力的解释：**VLA 鲁棒性差的根因是"视觉利用机制"**——架构 shortcut 让 action query 绕过视觉细节、过多视觉 token 稀释 attention 焦点、任务无关视觉信息引入噪声。FocusVLA（Zhang et al. 2026，arXiv:2603.28740）是这类解释的代表，提出 Modality Cascaded Attention + Focus Attention 两个机制让模型"被迫看图"，0.5B 参数在 LIBERO 多权重 98.7% 打败 7B OpenVLA-OFT。

**但这类工作有一个共同的方法论盲点**：它们用 *模型自己学到的 attention* 来证明 *模型自己学到的 attention 是对的*——缺乏外部参考真值。FocusVLA 自述局限 #3 「VLM 内部的视觉利用没涉及」与 #4 「未在 robustness benchmark 上系统评测」即是这一盲点的直接体现。

### 1.2 三个研究空白（立项关键切入点）

通过对 `论文探索/` 101 篇精读笔记 + 调研期 web 搜索的 15+ 篇相邻工作整理，识别出三个明确空白：

1. **没有任何论文用 SAM/GroundedSAM mask 作为外部参考真值**系统评估 VLA 视觉聚焦的正确性与鲁棒性。Oat-VLA（arXiv:2509.23655，object-agent-centric tokenization）、SlotVLA（arXiv:2511.06754，slot attention）用了 object-level token，但都是**改架构 + 重训**路线，没在 robustness benchmark 上做 attention 质量量化。

2. **FocusVLA / Oat-VLA / TokenFLEX 等"视觉聚焦类 VLA" 完全没在 LIBERO-Plus 等扰动 benchmark 上做 attention-mask IoU 量化**。所有方法都自证"我让模型聚焦"，但没人回答"模型聚焦到哪里 = 任务物体所在的真实区域吗？扰动下这种聚焦还稳吗？"

3. **没有论文用物体 mask 计算"理论最优 K\*"**（覆盖任务物体所需的最小 patch 数），实证检验 FocusVLA / VLA-Pruner 等工作的固定 K=256 是否对多场景过度/不足。

### 1.3 项目内的预先工作证据

本项目研究方向并非临时拍脑袋——`论文探索/A_视觉利用鲁棒性/papers/2024-01_GroundedSAM_Ren.md` 末尾的 Action Item 早在 2026-05 调研初期就明确记录：

> **「评估：FocusVLA attention heatmap vs Grounded SAM mask 的 IoU 在 LIBERO 上量化」**

版本三正是把这条 Action Item 系统化、扩展到 LIBERO-Plus 7 维扰动并加上 H2 K\* 与 H3 联动崩塌两个衍生维度，形成完整研究问题。

### 1.4 一句话研究问题

> 在 LIBERO clean + LIBERO-Plus 7 维扰动下，**VLA 的视觉聚焦质量（attention-mask IoU）是否真的与扰动鲁棒性（SR）有因果关系**？**固定 K=256（FocusVLA 等工作的默认）是否在多场景下过度/不足**？**SAM 也认错时（共同失败）vs SAM 认对但 VLA 仍崩（看错了）的失败模式比例是多少**？

## 2. 核心假设（H1-H3）

### H1 · 对齐假设

> 失败 episode 的 VLA 注意力 × SAM 物体 mask IoU **显著低于**成功 episode：ΔIoU ≥ 0.10（绝对值），配对 t-test p < 0.05。

**验证**：
- 在 LIBERO clean + 7 维扰动下收集 ~2k 成功/失败 episode 配对
- 每 episode 取 5 个 keyframe，对 VLA attention map（last-layer cross-attention，patch-grid 归一化）× SAM 物体 mask（patch-grid 投影）算 IoU
- 配对 t-test（同任务、同 prompt、同 instance 的成功 vs 失败）
- 报告 Cohen's d effect size

**预期 effect size**：先验估计 ΔIoU ∈ [0.08, 0.20]，p < 0.05 可达。如 ΔIoU < 0.05 → 反直觉发现（视觉聚焦质量与成功率无关），仍可投 negative-finding workshop。

### H2 · 理论最优 K 假设

> 每场景"任务物体 mask 覆盖所需的最小 patch 数" K\* **中位数 ≤ 80**（远小于固定 K=256），且 K\* 在 LIBERO-Plus 7 维扰动下分布漂移 ≥ 30%。

**验证**：
- 对所有 keyframe 计算 K\* = "覆盖 SAM 任务物体 mask ≥ 90% 的最小 patch 数"
- 分维度统计 K\* 中位数、IQR、扰动前后漂移
- 与固定 K=256（FocusVLA）/ K=64,144,256（TokenFLEX）/ K=16（Oat-VLA）对比，算"过度选取率"（K_fixed - K\*) > 50% 的 episode 占比）与"不足覆盖率"（K_fixed < K\* 的 episode 占比）

**预期分布**：基于 LIBERO 单任务通常 1-3 个相关物体、ViT 14×14 patch grid（196 patch），物体级 mask 覆盖 patch 数预期在 20-80 区间。固定 K=256（>所有 patch）实际是"全保留"，K=16（Oat-VLA）则可能严重不足。

**Fallback 策略**：
- K\* 中位数 ≥ 200 → 改叙事为"固定 K=256 合理性的反证不充分"
- 扰动下漂移 < 10% → 改叙事为"任务相关区域大小对扰动不敏感"——这本身仍是有价值的发现

### H3 · 联动崩塌假设

> SAM mIoU 在某扰动维度上 drop ≥ 15pp 时，VLA SR drop 与 SAM mIoU drop **显著正相关** Pearson r ≥ 0.5（n=7 维 × 2 backbone = 14 点）；同步失败 episode（SAM 错且 VLA 错）占比 ≥ 30%。

**验证**：
- 7 维 × 2 backbone = 14 个数据点，每点是 (SAM mIoU drop, VLA SR drop) 配对
- Pearson / Spearman 相关
- 失败 episode 聚类：(SAM-fail, VLA-fail) / (SAM-ok, VLA-fail) / (SAM-fail, VLA-ok) / (SAM-ok, VLA-ok) 四象限

**衍生贡献**：训一个轻量 LightGBM 二分类器 `f: (attention entropy, attention-mask IoU, SAM mIoU) → episode 成败`，目标 AUC ≥ 0.70（5-fold CV）。

**理论辩护**：与版本二的 H3（方向词预测时序 → failure predictor）思路一致，但本项目的输入是**模型外部可观测的离散值**（SAM mIoU 是黑盒输入），更稳健。

### 三个假设的工程同构

所有验证只需 3 步：
1. VLA 前向 + attention dump（PyTorch hook）
2. SAM2/GroundedSAM 前向 + mask
3. 离线对齐计算（IoU、K\*、配对 t-test、Pearson、LightGBM）

**零 VLA 训练**。所有结果可在 ¥10K 预算 + 1 年内完成。

## 3. 与相邻工作的差异化（关键 novelty 论证）

| 工作 | 路径 | 是否在扰动 benchmark 评测 | 是否做 attention-mask IoU | 是否做 K\* 分布 | 与本项目的关系 |
|---|---|---|---|---|---|
| **FocusVLA** (2603.28740) | 改架构、重训 | ❌（只报 LIBERO clean） | ❌ | ❌（固定 K=256） | 本项目直接回应其局限 #3 #4 |
| **Oat-VLA** (2509.23655) | object-centric 重训 | ❌（只在 LIBERO） | ❌ | ❌（固定 16 token） | 用通用检测器、训练时固定数量；本项目用 SAM、评测时统计 K\* 分布 |
| **SlotVLA** (2511.06754, ICRA 2026) | slot attention 重训 | ❌ | ❌ | ❌ | 发布 LIBERO+ mask 数据集——**可作为本项目轨道 A 的备用 mask GT 来源** |
| **SAM2Act** (2501.18564) | SAM2 做 spatial memory | ❌ | ❌ | ❌ | 用了 SAM2 但非 token 替代，与本项目"探针"姿态不同 |
| **TokenFLEX** (2504.03154) | 训练时随机 K | ❌ | ❌ | ❌ | 真正的变长 K 但通过训练；本项目通过评测统计验证 K\* 应该变长 |
| **VLA-Cache** (2502.02175, NeurIPS 2025) | 静态 token 缓存 | ❌ | ❌ | ❌ | 隐式变长（缓存粒度），与本项目正交 |
| **VLA-Pruner / VLA-IAP / Compressor-VLA / SemanticVLA / LUVC / TPRL** | 各种 pruning | ❌（都未在扰动 benchmark） | ❌ | ❌ | 都"减少 token"但都固定 K，本项目通过 K\* 揭示其合理性 |
| **Gaze-Regularized VLA** (2603.23202) | gaze 作 attention oracle | ❌ | 间接（用 KL 散度对齐） | ❌ | 用 gaze 当 attention oracle 的同思路，但 gaze 需采集成本高；本项目用 SAM 替代 |
| **AttentionVoxel** (2509.20579) | DINOv2 attention 作 saliency | ❌ | ❌ | ❌ | 同思路（外部 attention 当 saliency oracle），但 DINOv2 attention 不是 mask-level |
| **本项目 SAM-as-Probe** | **纯评测、SAM mask + attention IoU + K\*** | **✅ LIBERO-Plus 7 维** | **✅ H1 主指标** | **✅ H2 主指标** | **填补 5 个空白** |

**关键 novelty 论证**：本项目**没有任何一项技术原创**（attention hook、SAM2、IoU 都是现成的），但**没有任何一篇论文同时**：(i) 用 SAM mask 作外部真值评估 VLA 视觉聚焦；(ii) 在 LIBERO-Plus 7 维扰动下系统化；(iii) 同时报告 IoU + K\* + 联动崩塌三个互补维度。这是个"组合空白"，对 2 大二本科生来说工程量友好且故事完整。

## 4. 技术路线（5 phase / 12 月）

### Phase 1 · 探针管线打通（M1-M2，2026-06 ~ 2026-07）

**目标**：让 4 个工具（OpenVLA-OFT-7B + SmolVLA-0.5B + SAM2 + GroundedSAM）在团队环境跑通，**确认轨道 A 是否可行**。

**关键动作**：
1. 4 仓库装通 + checkpoint 下载
2. 写 `probe/attention_hook.py`——hook OpenVLA-OFT last-layer cross-attention 输出
3. 写 `probe/sam_pipeline.py`——GroundedSAM 输入 (image, instruction)，输出 box → SAM mask → SAM2 时序传播
4. 写 `probe/iou.py`——mask → patch grid (14×14 / 16×16) 投影 + attention-mask IoU 计算
5. 在 5 张 LIBERO-Spatial 任务图上做 **GroundedSAM mIoU 校准实验**：与 LIBERO 自带 `observations/object_pose` 投影 mask 对比，量化 GroundedSAM 在 LIBERO 渲染图上的真实精度

**关键 P1 末决策（轨道 A vs 轨道 B）**：

| GroundedSAM mIoU on LIBERO | 决策 |
|---|---|
| ≥ 0.6 | ✅ **走轨道 A**（GroundedSAM 主路线）|
| 0.4 ~ 0.6 | ⚠️ 调 GroundingDINO box threshold 等参数；2 周内若未达标 → 切轨道 B |
| < 0.4 | 🛑 直接切 [[02_备选方案_仿真GT_mask兜底]] |

**关键文件路径**（推测，需 W1 确认）：
- `openvla-oft/prismatic/models/vlms/prismatic.py` — backbone 模型类，attention hook 注入点
- `Grounded-Segment-Anything/grounded_sam_demo.py` — pipeline 入口
- `sam2/sam2_video_predictor.py` — 视频版 mask 传播

### Phase 2 · 大规模 IoU 评测（M3-M5，2026-08 ~ 2026-10）

**目标**：跑完 2 backbone × 8 条件（clean + 7 维）× 500 instance × 3 seed ≈ 24K episode，验证 H1。

**核心评测设计**：
- 8 条件：LIBERO clean + LIBERO-Plus 7 维（layout / viewpoint / state / instruction / light / texture / noise）
- 2 backbone：OpenVLA-OFT-7B（主）+ SmolVLA-0.5B（辅，速度 5×）
- 每 cell：500 instance × 3 seed = 1500 episode
- 每 episode：取 5 个 keyframe（动作开始 / 接近物体 / 抓取瞬间 / 移动中 / 释放）算 attention-mask IoU

**评测规模**：8 × 2 × 1500 = 24K episode × 30s ≈ 200 GPU-h（SmolVLA 主跑 + OpenVLA-OFT 关键 cell 校验）

**关键产出**：
- 主表：2 backbone × 8 条件 × 4 指标（SR / mean-IoU / IoU drop vs clean / Cohen's d）
- 雷达图：每 backbone 在 7 维上的 SR vs IoU 退化轨迹
- 散点图：(IoU, SR) 配对 + 配对 t-test 显著性

**附加评测**（stretch goal）：
- LIBERO-Para 2 子集（语言改写，对照 LIBERO-Plus instruction 维度）
- VLA-Risk 6 维子集（若代码 release）

### Phase 3 · 理论最优 K 分布画像（M6-M7，2026-11 ~ 2026-12）

**目标**：把 P2 已收集的所有 keyframe 算 K\*，验证 H2。

**关键动作**：
1. 对每个 keyframe 算 K\* = "覆盖 SAM 任务物体 mask ≥ 90% 的最小 patch 数"
2. 分维度统计 K\* 分布：中位数 / IQR / 扰动前后漂移
3. 对比固定 K：
   - K=256（FocusVLA / VLA-Pruner / 多数 token pruning 工作）
   - K=144 / 64（TokenFLEX 推理时选项）
   - K=16（Oat-VLA）
4. 计算"过度选取率" P_over = P(K_fixed > 2 K\*)
5. 计算"不足覆盖率" P_under = P(K_fixed < K\*)

**关键产出**：
- K\* 分布直方图 × 7 维（8 张图）
- 论证表："为什么固定 K 在扰动下次优"
- 给社区的诊断证据：建议 K 设计应该 task/scene-aware

### Phase 4 · SAM-VLA 联动诊断（M8-M9，2027-01 ~ 2027-02）

**目标**：验证 H3，与 SAFE / I-FailSense 等 failure predictor baseline 对比。

**关键动作**：
1. 对每个扰动维度算 SAM mIoU drop 与 VLA SR drop 的 Pearson / Spearman 相关
2. 同步失败 episode 四象限聚类：
   - (SAM-fail, VLA-fail)：**共同失败**——视觉退化超出 SAM 与 VLA 共同能力
   - (SAM-ok, VLA-fail)：**VLA 看错**——视觉信息完整但 VLA 注意力没看到，最有诊断价值
   - (SAM-fail, VLA-ok)：**SAM 错但 VLA 仍对**——VLA 用了 SAM 看不到的视觉线索
   - (SAM-ok, VLA-ok)：**双双成功**
3. 失败模式画像：5-8 类典型 "SAM 与 VLA 都看错了什么"
4. 训轻量 failure predictor：LightGBM on (attention entropy, attention-mask IoU, SAM mIoU)，目标 AUC ≥ 0.70（5-fold CV）

**关键产出**：
- 联动相关系数表 + 失败模式 case study 图谱
- 轻量 failure predictor（LightGBM 100 行代码 / < 1MB 模型）

### Phase 5 · 写作 + 投稿（M10-M12，2027-03 ~ 2027-05）

**目标**：至少投出 1 个 workshop。

**关键 deadline**（粗估，需以最新 CFP 为准）：
- CoRL 2027 workshop：通常 8-9 月（在 P5 后，可作下一年准备）
- ICLR 2027 Robot Learning workshop：通常 2-3 月（**主要 target**）
- NeurIPS 2027 Robot Learning workshop：通常 9-10 月
- 国内会议（CCC / CAA）：3-5 月 deadline（保底）

**故事结构**：
- §1 Intro：FocusVLA 等工作明确把 VLA 鲁棒性差归因于视觉利用机制，但都没用外部真值系统验证 → 我们补
- §2 Related：FocusVLA / Oat-VLA / TokenFLEX / VLA-Pruner / Gaze-Reg / AttentionVoxel（约 20 篇）
- §3 Method：SAM-as-Probe pipeline（attention dump + SAM mask + IoU + K\*）
- §4 Experiment：H1 + H2 + H3 三大发现
- §5 Discussion：对"变长 token / 物体级 grounding"未来工作的启示

## 5. Go/No-Go 准则

| 节点 | Go 条件 | No-Go 触发 → 应对 |
|---|---|---|
| **P1 末（M2）** | (a) attention hook 成功提取 last-layer cross-attention；(b) GroundedSAM 在 LIBERO 上 mIoU ≥ 0.6 或确认走轨道 B；(c) 5 任务 × 10 episode 跑通 IoU pipeline | mIoU < 0.4 → **切轨道 B（仿真 GT mask）**；attention hook 困难 → 限定 last-layer / cls token attention 简化 |
| **P2 末（M5）** | H1 至少 1 维度显著（ΔIoU ≥ 0.05，p < 0.1） | 完全否定 → 改叙事为"VLA 视觉聚焦质量与扰动鲁棒性无因果关系——反直觉负面 finding workshop"（仍可投） |
| **P3 末（M7）** | H2 成立（K\* 中位数 ≤ 100 或扰动漂移 ≥ 20%） | 不成立 → 改 P4 为"固定 K 合理性的反证补充实验" |
| **P4 末（M9）** | H3 成立（r ≥ 0.4）+ failure predictor AUC ≥ 0.65 | < 0.6 → 改 descriptive case studies，去掉 predictor 卖点 |

详细决策树见 [[05_风险与缓解]]。

## 6. 关键工具 / 代码 / 数据依赖

| 类别 | 依赖 | 状态 | 备注 |
|---|---|---|---|
| 主 backbone | `openvla/openvla-7b-oft` (HF) | ✅ 已开源 | 推理 < 24GB；不训练 |
| 备 backbone | `HuggingFaceTB/SmolVLA-base` | ✅ 已开源 | 速度 5× 用于大规模评测 |
| 评测主 benchmark | LIBERO-Plus（`senyufei/LIBERO-Plus`） | ✅ HF 开源 | 7 维 × 4 suites |
| 鲁棒评测附加 | VLA-Risk（6 维） | ⏳ 待 release | stretch goal |
| 鲁棒评测附加 | LIBERO-Para（语言改写） | ✅ 已开源 | stretch instruction 维度补强 |
| 物体分割主 | Grounded-Segment-Anything（IDEA-Research）| ✅ 工程级 | 15K+ stars |
| 物体分割时序 | SAM2（facebookresearch/sam2）| ✅ 已开源 | 视频版 mask 传播 |
| 物体分割备用 | LIBERO+ mask 标注（SlotVLA 发布）| ⏳ 待确认 release | 轨道 A 备用 mask 来源 |
| 仿真器 | robosuite + MuJoCo | ✅ 开源 | LIBERO 自带 |
| FocusVLA 代码 | HIT-Shenzhen | ⏳ 待 release | 不强依赖，stretch 加入第 3 backbone |

## 7. 必读 / 强参考资料

- [[2026-03_FocusVLA_Zhang]] — 导航星，本项目直接回应其局限 #3 #4
- [[2024-01_GroundedSAM_Ren]] — **关键**：Action Item 是版本三的预先工作证据
- [[2025-10_LIBERO-Plus]] — 主评测
- [[2025-09_AttentionVoxel_Yurchyk]] — DINOv2 attention 当 saliency oracle 同思路
- [[2026-03_Gaze-Regularized-VLA_Pani]] — gaze 当 attention oracle 同思路
- [[2026-03_ST-VLA_Wu]] — 4D mask 作 attention prior 同思路
- [[2025-08_ShortcutLearning_Xing]] — shortcut 诊断方法学
- [[2025-11_VLA-Pruner_Liu]] / [[2025-11_Compressor-VLA_Gao]] / [[2025-11_SemanticVLA_Li]] / [[2025-12_LUVC_Zheng]] / [[2026-03_VLA-IAP_Cheng]] / [[2026-03_TPRL_Cao]] — H2 K\* 的对照基线

新论文（调研期发现，待补本地笔记）：Oat-VLA (2509.23655) / SlotVLA (2511.06754) / SAM2Act (2501.18564) / TokenFLEX (2504.03154) / VLA-Cache (2502.02175) / DyVTE (2411.19628)

完整列表见 [[06_关联资料索引]]。

## 8. 实验设计细节

### 8.1 评测指标

| 指标 | 定义 | 用于 |
|---|---|---|
| **任务成功率 SR** | LIBERO 标准协议（is_success at episode end） | 所有 H 假设的对照量 |
| **mask-attention IoU** | `|attn-topK ∩ mask-patches| / |attn-topK ∪ mask-patches|`，K = attention 概率前 25% patch | H1 主指标 |
| **mask coverage @K** | `|attn-topK ∩ mask-patches| / |mask-patches|` | 召回率视角 |
| **理论最优 K\*** | 最小 K 使 attn-topK 按 attention 概率排序后覆盖 ≥ 90% mask patches | H2 主指标 |
| **SAM mIoU** | SAM 输出 mask 与 GT mask（仿真投影或人工标注）的 IoU | H3 输入 |
| **co-failure rate** | (SAM-fail ∧ VLA-fail) episode 占比 | H3 辅指标 |
| **failure predictor AUC** | LightGBM on (entropy, IoU, SAM mIoU) → 成败 | H3 应用价值 |

### 8.2 attention 提取约定

- **层选择**：last-layer cross-attention（action query × visual key），稳定可靠
- **head 聚合**：head-wise 求均值（vs head-wise 最大）—— P1 末做 ablation 决定
- **patch grid**：与 ViT 输入 patch 对齐（OpenVLA-OFT 通常 14×14 / 16×16）

### 8.3 mask grid 投影

SAM 输出二值 mask（H × W 像素级）→ patch grid（14×14）的投影逻辑：

```
patch_mask[i,j] = 1 if (mask 在第 (i,j) patch 上的像素占比 ≥ 0.3) else 0
```

阈值 0.3 是工程默认，在 P1 末做 ablation（0.2 / 0.3 / 0.5）。

### 8.4 统计方法

- 每个 cell（变体 × 维度）报告 mean ± std（3 seed）
- H1 主对比用配对 t-test（同 instance 不同模型对比）
- 多重比较 Bonferroni 校正
- 报告 effect size（Cohen's d）

### 8.5 与版本二 v2 的脚本复用

- LIBERO-Plus 评测脚本骨架：复用 `版本二/立项调研/03_第一步两周任务清单.md` § W2 中 `eval/eval_libero.py` 的设计
- LoRA 微调相关代码：**不复用**（版本三不训练）
- 方向词 GT 标注脚本：**不复用**（版本三用 SAM mask 替代显式方向词监督）

## 9. 与版本二的互斥关系

版本二（InSpire 显式提示插件 + 视觉利用 ablation）与版本三（SAM-as-Probe）研究问题不重叠，但在 **OpenVLA-OFT 主 backbone + LIBERO-Plus 主评测平台**上完全共用。这意味着：

1. **同一团队不能同时跑两条路线** —— GPU 时长不够。
2. **递交材料层可以同时提交两份申请书草稿** —— 交给老师 + 院系评审决定。
3. **写作侧可以"两路并行"**：版本三主投 IoU 故事，版本二可作为 future work / appendix（"如果未来想训练 VLA，可以加 InSpire 提示插件"）。
4. **资源选择上版本三更稳**：零训练、零方向词 GT 噪声、零 cls head 收敛风险。这是导师建议"在 FocusVLA 方向深挖鲁棒性"的最低风险落地方式。

详见 [[09_与版本二差异化说明]]。

%% Updated: 2026-05-26 %%
