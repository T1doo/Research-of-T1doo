---
create time: 2026-05-26T22:15:00
tags:
  - 曦源项目
  - 版本三
  - 备选方案
  - SAM-as-Probe
  - 仿真GT
  - Plan_B
---
# 02 · 备选方案 · 仿真 GT mask 兜底（Plan B）

> **一句话定位**：若 P1 末（M2，2026-07 末）GroundedSAM 在 LIBERO 渲染图上的 mIoU < 0.5，**把外部视觉真值从 SAM 替换为 LIBERO 仿真器原生 GT mask**（通过 robosuite camera API + URDF 渲染）。H1 / H2 完全保留，H3 从"SAM-fragility"改写为"GT-mask drift vs VLA SR drop"。**与 Plan A 共用 attention dump + IoU 框架，切换成本 < 2 周。**

## 1. 何时启动 Plan B

仅在以下任一条件触发时切换到 Plan B：

1. **P1 末（M2，2026-07 末）GroundedSAM 在 LIBERO 校准实验上 mIoU < 0.5** —— LIBERO 是仿真渲染图，风格与 GroundedSAM 训练分布（COCO 真实图像）有 gap，可能出现物体识别错位、box 框过大/过小、SAM 在低纹理表面失效等问题。
2. **SAM2 视频传播在 LIBERO 长 episode 上失稳** —— 物体被遮挡、抓取后形态变化、释放后位置变化等场景，SAM2 可能丢失追踪。
3. **GroundedSAM + SAM2 端到端推理速度过慢**（单 episode > 30s 额外开销）—— 24K episode × 30s = 200 GPU-h 已经是极限，再 +30s/episode 直接超预算。

切换的**操作成本** < 2 周（≈ Plan A 总工程量的 10%），因为：
- Plan A 的 `probe/attention_hook.py` 完全保留
- Plan A 的 `probe/iou.py` 完全保留（IoU 计算与 mask 来源无关）
- 仅替换 `probe/sam_pipeline.py` → `probe/sim_gt_mask.py`

## 2. 与 Plan A 的关系（共用 vs 替换）

| 组件 | Plan A 状态 | Plan B 状态 |
|---|---|---|
| OpenVLA-OFT-7B + SmolVLA-0.5B 环境 | ✅ 用 | ✅ 用（不变） |
| LIBERO-Plus 7 维评测脚本 | ✅ 用 | ✅ 用（不变） |
| Attention hook（PyTorch）| ✅ 主分析手段 | ✅ 主分析手段（不变） |
| `probe/iou.py` IoU 计算器 | ✅ 用 | ✅ 用（不变） |
| Mask → patch grid 投影 | ✅ 用 | ✅ 用（不变） |
| **GroundedSAM 推理管线** | ✅ **主方法** | ❌ **不用** |
| **SAM2 视频传播** | ✅ 用 | ❌ 不用 |
| **robosuite camera API + URDF 渲染** | ❌ 不用 | ✅ **主方法（mask 来源）** |
| Failure predictor (LightGBM) | ✅ H3 验证 | ✅ H3 验证（输入特征略变） |

**共用组件**占 Plan A 总工程量的 ~75%，所以切换成本极低。

## 3. 核心研究问题（与 Plan A 的差异）

### H1（对齐假设）—— **完全保留**

> 失败 episode 的 VLA 注意力 × **仿真 GT mask** IoU 显著低于成功 episode，ΔIoU ≥ 0.10，配对 t-test p < 0.05。

仿真 GT mask 是物理意义上的"完美 mask"（mIoU = 1.0 对真实物体边界），H1 验证更严格——如果用 GT mask 都验证不出对齐效应，说明 attention 与任务物体没有任何关系。

**预期 effect size 比 Plan A 更大**：因为 mask 完美，attention-mask IoU 的 signal/noise 比更高。

### H2（理论最优 K\*）—— **完全保留**

> 每场景"任务物体 GT mask 覆盖所需的最小 patch 数 K\*" 中位数 ≤ 80，K\* 在 LIBERO-Plus 7 维扰动下分布漂移 ≥ 30%。

GT mask 完美但**物体在不同扰动下位置/姿态变化** —— GT mask 的 patch 覆盖数仍会随场景变化，H2 验证仍成立。

### H3（联动崩塌）—— **重新表述**

Plan A 的 H3 是 "SAM mIoU drop（SAM 失效）与 VLA SR drop 联动"，依赖 SAM 有"失效"行为。Plan B 用仿真 GT mask，**mIoU 恒为 1.0**，所以"SAM-fragility"概念不存在。

**改写为 H3' · GT-mask drift 假设**：
> VLA SR drop 与 **GT mask 在扰动维度上的几何 drift**（位置/大小/可见性的统计变化）显著正相关 Pearson r ≥ 0.5，n=7 维 × 2 backbone = 14 点。

具体的"GT mask drift"指标：
- **位置 drift**：mask 中心在 patch grid 上的移动距离均值
- **大小 drift**：mask 覆盖 patch 数的标准差
- **可见性 drift**：mask 被遮挡的 patch 数占比

**衍生 failure predictor**：LightGBM on (attention entropy, attention-mask IoU, GT mask drift metrics) → episode 成败，目标 AUC ≥ 0.70。

## 4. LIBERO GT mask 实现细节

### 4.1 数据源

LIBERO 基于 robosuite + MuJoCo。每个 episode 的 `observations` 自带：
- `robot0_eef_pos`、`robot0_eef_quat`：机器人末端位置 + 朝向
- `object`：任务物体的位置（部分任务给出 quaternion）
- `agentview_image`：第三人称视角图
- `eye_in_hand_image`：手腕相机图

### 4.2 渲染 GT mask 的 2 个路径

**路径 A · robosuite 内置 segmentation rendering**（推荐）：
- robosuite 的 `OffscreenRenderer` 支持 `segmentation_mode='instance'` 或 `'class'`，可直接输出像素级 instance ID map
- 对应代码：`robosuite/utils/camera_utils.py` 中的 `get_real_depth_map` 等工具函数（具体 API 需 W1 末确认）

**路径 B · URDF + 相机内外参手动投影**（备用）：
- 从 LIBERO `object_pose` + URDF 几何模型（`.obj` mesh）取物体 3D 顶点
- 用相机内参 K 与外参 [R | t] 把 3D 顶点投影到 2D
- 取投影点的 convex hull 作为 mask 近似

**路径 A 更准确、路径 B 更通用**。W1 末根据 robosuite 文档情况决定。

### 4.3 多物体场景的处理

LIBERO 单任务通常涉及 1-3 个物体（如"pick alphabet soup and place in basket" 涉及 2 个物体）。GT mask 输出应为多 channel（每物体一个 mask），attention-mask IoU 默认对**任务相关物体的 mask 并集**计算。

### 4.4 与 SlotVLA LIBERO+ 数据集的关系

SlotVLA（arXiv:2511.06754，ICRA 2026）发布了 LIBERO+ 数据集，含 box-/mask-level 标注 + instance-level temporal tracking。**如该数据集已 release，Plan B 可直接复用**，避免自己写仿真渲染脚本（节省 1-2 周工程）。

W1 期间 B 同学的并行任务之一：**查 SlotVLA 数据集 release 状态**（搜 GitHub `slotvla` / OpenReview / 作者主页）。

## 5. Plan A → Plan B 切换 48 小时操作清单

| 操作 | 时间 | 备注 |
|---|---|---|
| 团队会议确认切换 | 1h | A + B + 导师（如可联系） |
| 关闭 GroundedSAM / SAM2 推理管线 | 2h | git branch + 注释相关代码 |
| 切换到 robosuite GT mask 渲染 | 8h | 写 `probe/sim_gt_mask.py` |
| 调试 GT mask 与 attention 的 patch 对齐 | 8h | 复用 Plan A 的 `probe/iou.py` |
| 更新 H3 表述为 "GT-mask drift" | 4h | 改 `01_主方案` § 2 & § 4 |
| 跑首批诊断实验（1 维度 × 2 backbone × 50 episode）| 12h | 验证 pipeline 通 |
| 更新立项文档（备注切换原因） | 2h | 写入 `notes/plan_b_switch_log.md` |
| **合计** | **37h** | < 48 小时内完成 |

## 6. 与 Plan A 共用的写作框架

切换到 Plan B 后，workshop short paper 的故事仅在 § 3 Method 改 mask 来源描述，其他章节几乎无需重写：

| §  | Plan A 内容 | Plan B 改动 |
|---|---|---|
| § 1 Intro | "FocusVLA 等工作没用外部视觉真值..." | "我们用仿真 GT mask 作为外部视觉真值..." |
| § 2 Related | FocusVLA / Oat-VLA / ... | 不变 |
| § 3 Method | GroundedSAM + SAM2 管线 | robosuite GT mask 渲染管线 |
| § 4 Results | H1 + H2 + H3 | H1 + H2 + H3'（GT-mask drift） |
| § 5 Discussion | "未来工作真机扩展用 SAM" | 增加 "GT mask 仅适用于仿真，真机场景需要 SAM/检测器"——但**这本身是个清晰的 future work** |

**Plan B 的故事缺点**：
- 真机泛化承诺减弱（GT mask 不可移植）
- novelty 在审稿人眼中可能稍弱（"用仿真器自带 mask 不算新方法"）

**Plan B 的故事优点**：
- mask 精确，H1 / H2 验证更可信
- 完全免去 SAM 推理成本
- 工程量更小

## 7. 关键 Go/No-Go（切换后）

| 节点 | Go 条件 | 应对 |
|---|---|---|
| 切换后第 1 周末 | GT mask 渲染脚本跑通，10 episode 上 attention-mask IoU 计算正确 | 不通 → 检查 robosuite API；调用 Habitat3 / Holodeck 等其他仿真器 mask 工具 |
| P2 末（M5） | 与 Plan A 同 Go 条件（H1 至少 1 维度显著） | 同 Plan A |
| P3-P5 | 与 Plan A 完全相同 | 同 Plan A |

## 8. 必读 / 强参考资料

- [[2025-10_LIBERO-Plus]] —— 主评测，无论 Plan A/B 都用
- [[2025-08_ShortcutLearning_Xing]] —— 诊断方法学
- robosuite 文档（https://robosuite.ai）—— GT mask 渲染 API
- LIBERO GitHub（github.com/Lifelong-Robot-Learning/LIBERO）—— 数据结构
- SlotVLA arXiv:2511.06754 —— LIBERO+ mask 数据集（如 release 可直接复用）

## 9. 一句话总结

> Plan B 是 Plan A 的"无外部依赖精简版"——SAM/GroundedSAM 失效时把 mask 来源换成仿真 GT，研究问题与方法学几乎不变。真机泛化承诺减弱，但研究严谨性增强。默认目标仍是 Plan A（更通用、更接近真实场景应用）。

%% Updated: 2026-05-26 %%
