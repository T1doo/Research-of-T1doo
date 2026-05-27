---
create time: 2026-05-27T17:50:00
tags:
  - 曦源项目
  - 版本二plus
  - 可行性
  - novelty
  - 师哥讨论
status: 工作版v1
audience: 与师哥/导师讨论时的独立分析文档
---

# 版本二plus · 可行性与 Novelty 分析

> **本文档定位**：与师哥/导师讨论 v2plus 立项时使用的独立分析文档，浓缩可行性论证与 novelty 论证。2 页内说清楚。
> **配套**：[[../立项调研/09_与版本二差异化说明]]（v2 vs v2plus 对比）+ [[../立项调研/00_总览与立项决定]]（总体定位）

## A. 一句话总结

**v2plus 在 v2 (OpenVLA-OFT + InSpire) 基础上新增 Focused Spatial Forcing 通道（FastVGGT teacher + 多源融合 focus mask），并增加真机扰动评测；可行性高（核心组件均有开源 + 实验室真机硬件），novelty 集中在"显式提示 × 隐式 3D 对齐 × 多源 focus mask × 真机"四点叠加。**

## B. Novelty 分析

### B.1 v2plus 的 4 项核心 novelty

| # | Novelty | 文献中是否空白 | 师哥提示对应 |
|---|---|---|---|
| **1** | 显式方向词提示（InSpire）与隐式 3D 表征对齐（Spatial Forcing）的**叠加效应** | **空白**——InSpire 与 SF 从未在同一项目中叠加验证 | ✅ "加 3D 是好想法" |
| **2** | **多源融合 focus mask**（方向词GT + SAM2 + DINOv2 CLS attention）调制 SF loss | **完全空白**——SF 原论文是均匀监督；其他工作（AttentionVoxel/Gaze-Reg）单源 | ✅ "spatial forcing 但需要 focus 在重要位置，背景可弱" |
| **3** | **三段量化 focus weight**（前景 1.0 / 上下文 0.5 / 背景 0.1）的实证 | **空白**——SF 原论文 $m_i \equiv 1$；其他工作多用硬 mask（0/1） | ✅ "不相关背景监督可以更弱点" |
| **4** | **真机扰动评测**（3 任务 × 3 扰动维度 × 360 episode） | **部分空白**——SF 仅 RoboTwin 仿真，InSpire 仅小规模真机 | ✅ "按 InSpire 的会，要在真机上验证" |

### B.2 与 4 个最相关工作的对比

| 维度 | InSpire (2505.13888) | Spatial Forcing (2510.12276) | FocusVLA (2603.28740) | **v2plus FSF-VLA** |
|---|---|---|---|---|
| 显式方向词 prompt | ✅ | ❌ | ❌ | ✅ **(继承)** |
| 隐式 3D 表征对齐 | ❌ | ✅ | ❌ | ✅ **(继承)** |
| VGGT 用作 teacher | ❌ | ✅（中间层对齐） | ❌（policy 阶段并行 encoder，劣等）| ✅ **(SF 优等做法)** |
| Focus mask 调制 | ❌ | ❌（均匀） | ❌（patch top-K，不调 loss） | ✅ **(v2plus 创新)** |
| 多源 mask 融合 | ❌ | ❌ | ❌ | ✅ **(v2plus 创新)** |
| 三段量化（fg/ctx/bg） | ❌ | ❌ | ❌ | ✅ **(v2plus 创新)** |
| LIBERO-Plus 7 维评测 | ❌（仅 clean OOD） | ❌（仅 RoboTwin clean） | ❌（仅 LIBERO clean） | ✅ **(继承 v2)** |
| 真机扰动评测 | ✅（小规模） | ❌ | ✅（背景/空间/目标）| ✅ **(继承 InSpire 协议)** |
| 视觉利用 ablation | ❌ | ❌ | ❌（间接通过 Focus Attention）| ✅ **(继承 v2)** |
| Failure predictor | ❌ | ❌ | ❌ | ✅ **(继承 v2 H3)** |

**结论**：v2plus 在以上 10 个维度上**全部具备**，且 4 项核心 novelty 不与任一现有工作重叠。

### B.3 与师哥推荐的 FocusVLA 故事的对应

师哥推荐 FocusVLA 作为"导航星论文"。v2plus 与 FocusVLA 的关系：

- **同方向**：都关注 VLA 视觉利用机制（FocusVLA 自述局限 #3）
- **不同实现**：
  - FocusVLA：架构改造（Cascaded Attention + patch-level top-K + channel gate）
  - v2plus：**轻量 monitoring + 隐式 3D 监督**（不改架构，仅加 projector + 多源 mask loss）
- **互补**：v2plus 的视觉 ablation 提供 FocusVLA 自述局限 #3 的实证证据
- **回应**：v2plus 直接回应 FocusVLA 4 大局限的 #1（背景纹理）、#2（初始状态）、#3（VLM 视觉利用）、#4（robustness benchmark 评测）

详见 [[../立项调研/09_与版本二差异化说明]] § 3。

## C. 可行性分析

### C.1 工程可行性

**v2plus 的工程组件全部有开源**：

| 组件 | 仓库 | 状态 |
|---|---|---|
| OpenVLA-OFT 主干 | github.com/openvla/openvla | ✅ HuggingFace 已发权重 |
| FastVGGT teacher | github.com/mystorm16/FastVGGT | ✅ ICLR 2026，4× 加速 |
| Spatial Forcing 参考代码 | github.com/OpenHelix-Team/Spatial-Forcing | ✅ 已发，~30 行核心修改 |
| SAM2 + GroundedSAM | facebookresearch/sam2 + IDEA-Research/Grounded-SAM | ✅ 已发 |
| DINOv2 | facebookresearch/dinov2 | ✅ 已发 |
| LeRobot 真机 SDK | huggingface/lerobot | ✅ 已发 |
| LIBERO-Plus 评测 | huggingface.co/senyufei/LIBERO-Plus | ✅ 已发 |
| **真机硬件平台** | **实验室已有**（用户确认） | ✅ |

**核心代码增量**：
- FSF projector 模块：~30 行（参考 Spatial Forcing 原论文）
- Focus mask 多源融合管线：~150 行（Source A/B/C 各 ~50 行）
- VGGT 离线缓存 pipeline：~100 行（WebDataset 格式）
- 真机数据采集协议：~200 行（基于 LeRobot 改）

**总增量代码 ≈ 480 行**，对 2 大二本科生（已熟悉 PyTorch + LoRA）可控。

### C.2 算力可行性

| 项目 | GPU-h | 说明 |
|---|---|---|
| VGGT 离线缓存（一次性） | 11 | FastVGGT 4× 加速 |
| V1/V2/V3 LoRA 训练 × 3 seed | 95 | 单 seed 8-12h |
| LIBERO-Plus 主评测（21 cell × 300 inst × 3 seed） | 158 | 紧约束 300 inst |
| 核心 ablation（A1/A2/A3） | 70 | 必做 |
| 视觉 ablation + 粗扫 ablation | 35 | A4/A5/A6 |
| 真机评测 | 0 | 本地 RTX 4090 推理，不进云 GPU 账户 |
| **总计** | **~370** | 单维度紧约束版下可降到 230 |

云 GPU 预算 ¥6000 / 230 GPU-h × ¥26/h（学校共享优先）→ **预算可行**。

### C.3 时间可行性

| Phase | 时长 | 主任务 |
|---|---|---|
| P0 | 2 周（M0） | 环境 + 真机准备 |
| P1 | 3 月（M1-M3） | 复现 + sanity + 真机采集 |
| P2 | 3 月（M4-M6） | 主评测 + 标注 |
| P3 | 3 月（M7-M9） | 机制分析 + 真机评测 |
| P4 | 2 月（M10-M11） | Failure predictor + 论文 |
| P5 | 1 月（M12） | 投稿 |
| **总计** | **12 月** | |

每人时间投入：每周 14-18 小时（与 v2 持平 +20%）。

### C.4 真机可行性

实验室已有真机硬件（用户确认）：
- 机械臂 6+ DoF
- 第三视角相机
- 工作台 + 物体 + 标定板
- 主机 + GPU

3 任务 × 50 demo 采集 = 5-7 天人工。
360 episode 评测 = 5 天人工。

**总真机人工预算 ≈ 12-13 天**（分散在 P1-P3 中）。

详见 [[../研究报告/05_真机平台与数据标注方案]]。

### C.5 与 v2 的工程量对比

| 维度 | v2 | v2plus | 增量 |
|---|---|---|---|
| 训练变体数 | 2（V1/V2） | 3（V1/V2/V3） | +50% |
| 训练数据 | LIBERO 26000 demo | LIBERO 26000 demo + 真机 150 demo | +0.6% |
| 评测 cell（仿真）| 14 | 21 | +50% |
| 评测 cell（真机）| 0 | 12 | +12 cell |
| 总 GPU-h | 175 | 230 | +31% |
| 新增代码量 | 200 行 | 680 行 | +240% |
| 新增组件 | InSpire plugin | InSpire + FSF projector + 多源 mask + VGGT 缓存 + 真机 | +4 组件 |
| 团队人月 | 24 | 24（紧约束）| +0% |
| 预算 | ¥10K | ¥10K | +0% |

**结论**：v2plus 工程量增加 ~30-50%，但与 v2 共享 100% 代码框架；新增组件均有清晰 fallback；总体在 2 人 1 年 ¥10K 紧约束下可行。

## D. 风险与可控性

### D.1 5 个最大风险（概率排序）

| 风险 | 概率 | 影响 | 切换成本 |
|---|---|---|---|
| LIBERO-Plus 评测超 GPU-h 预算 | 50% | 低（降 instance 即可） | < 1 天 |
| 真机数据 50 demo 不够 fine-tune | 45% | 中 | 1-3 天调 fine-tune 策略 |
| Focus mask 多源无差异（H5 失败）| 40% | 中（novelty 降级） | 1-2 天改叙事 |
| FSF Loss 在 OpenVLA-OFT 上退化 SR | 30% | 高 | < 1 周（切 v2 only） |
| VGGT 离线缓存超磁盘 | 30% | 中 | 1-3 天（fp8 量化或 streaming） |

**所有风险都有明确 fallback，最坏退到 v2 only（已稳定方案）**。详见 [[../立项调研/05_风险与缓解]]。

### D.2 与 v2 风险的对比

v2 也有 5 个 top 风险（InSpire 代码未释、方向词 GT 噪声、cls head 不学、LoRA 工程坑、FocusVLA 代码未释）。v2plus 新增 4 个风险（VGGT 缓存、FSF 收敛、Focus mask 多源、真机不够），但**所有 v2plus 新风险都可通过 fallback 退到 v2**。

**等价于 v2plus = v2 + 一些可选的额外组件**，组件失败时退到 v2 完整方案，总体风险可控。

## E. 与师哥提示的全面对齐

师哥 2026-05-27 给出 7 条具体提示，v2plus 全部响应：

| # | 师哥提示 | v2plus 对应章节 |
|---|---|---|
| 1 | "加 3D 是个好想法" | FSF 通道（[[../立项调研/01_主方案_FocusedSpatialForcing]] §1） |
| 2 | "按 InSpire 的会，要在真机上验证" | 真机 3 任务 × 50 demo × 3 扰动维度（[[../研究报告/05_真机平台与数据标注方案]]） |
| 3 | "参考 arXiv:2510.12276，隐式 3D 表征对齐" | FSF loss = SF loss + focus mask 调制（[[../立项调研/01_主方案_FocusedSpatialForcing]] §2） |
| 4 | "FocusVLA 中也用了 VGGT，看做法是否有区别" | SF（中间层对齐优等）vs FocusVLA（policy 并行劣等）对比（[[../研究报告/03_FocusVLA_中_VGGT_用法剖析]]） |
| 5 | "spatial forcing 需要 focus 在重要位置" | 多源融合 focus mask（[[../研究报告/04_Focus_Mask_设计调研]]） |
| 6 | "不相关背景的监督可以更弱点" | 三段量化 fg=1.0 / ctx=0.5 / bg=0.1（[[../立项调研/01_主方案_FocusedSpatialForcing]] §3） |
| 7 | "vggt 续作可以大致看下" | FastVGGT 主选 + π³/StreamVGGT/HD-VGGT 续作综述（[[../研究报告/02_VGGT_及后续工作综述]]） |

**对齐度 7/7 = 100%**。

## F. 投稿 venue 适配度

| Venue | 期望产出 | v2 | v2plus |
|---|---|---|---|
| CoRL 2027 main | 4-6 页 main 短稿 | ⭐⭐ 低 | ⭐⭐⭐ 中（H1+H4+H5 全成立时） |
| **CoRL 2027 workshop** | workshop short | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ **极高（真机加分）** |
| ICLR 2027 Robot Learning workshop | workshop short | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ 极高 |
| NeurIPS 2027 Robot Learning workshop | workshop short | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ 极高 |
| 国内会议 CCC/CAA | 保底短稿 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**v2plus 的真机扰动评测使其在机器人会议上的接受度提高一档**。

## G. 与师哥讨论的关键问题（建议会议提问）

1. **v2plus 替换 v2 是否合适？**——师哥是否认为 plus 的增量足够，可以替换 v2 作为最终立项？
2. **VGGT teacher 选择 FastVGGT 是否合适？**——师哥是否同意 FastVGGT > VGGT 原版？
3. **真机 3 任务 × 50 demo × 3 扰动维度的 scope 是否合适？**——师哥是否认为这与 InSpire 真机协议一致？
4. **¥10K 预算是否足够？**——是否需要申请额外资助？
5. **2 人团队是否够？**——师哥是否认为应该招第 3 人专责真机？

师哥回答后，相应修订本文档与 [[../立项调研/]] 各文件。

%% Updated: 2026-05-27 %%
