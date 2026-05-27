# 曦源项目 v2plus 立项材料系统性审查报告

**审查范围**：`/home/user/Research-of-T1doo/曦源项目/版本二plus/` 下 19 份立项材料（11 立项调研 + 6 研究报告 + 2 递交材料）  
**审查日期**：2026-05-27  
**审查标准**：复旦曦源项目（2 人 / 1 年 / ¥10000 本科生学术研究资助）评审尺度。学术规范（特别是文献引用真实性）一票否决。  
**文献核查覆盖**：L1 重点文献 18 篇（全核）+ L2 次要文献 6 篇（抽检）+ GitHub 仓库 5 个

---

## 0. 一句话总评

**推荐立项决定：🟡 需重大修改后立项**。项目科研思路有实质价值（"InSpire 显式 grounding + Spatial Forcing 隐式 3D 对齐 + 双源 focus mask 调制"在文献中确为空白点，工程组件全部有真实开源可用），核心方法学差异化论证方向在数据层面也站得住；**但学术规范问题严重——`研究报告/03_FocusVLA_中_VGGT_用法剖析.md` 中 4 段标为"FocusVLA 原文引用"的引文中 3 段在原文中不存在，4 项数字声明全错（LIBERO 成功率、Cascaded Attention 结构、最佳模型是否用 VGGT、真机实验设置），且关联资料索引中至少 10 篇论文标"Anonymous"实际作者真实可查。** 这些问题在曦源审查中属于一票否决范畴，必须在递交前完成修订。

---

## 1. 致命问题清单（按严重度排序）

### 1.1 🔴 P0 · FocusVLA 论文引文 3/4 段虚构（学术规范一票否决项）

**位置**：`研究报告/03_FocusVLA_中_VGGT_用法剖析.md` §2.1 / §2.2 / §2.3 / §4.5

**项目报告引用情况**：4 段带引号"FocusVLA 原文"，每段都明确标注章节归属。

**核查方法**：用 `web_fetch` 抓 `https://arxiv.org/abs/2603.28740` 与 `https://arxiv.org/html/2603.28740v1`，对论文全文做关键词搜索与逐字比对。

**核查结果**：

| 引文 | 项目标注位置 | 引号原文 | 论文实际内容 | 判定 |
|---|---|---|---|---|
| 引文 1 | §2.1 架构偏差 | "Existing VLA models inherit the parallel multimodal attention from VLMs, where text queries and visual tokens attend to each other symmetrically..." | 论文 §1 实际是讨论 **VLA-Adapter 的 mixed attention mechanism "structural shortcut"**，对象是 VLA-Adapter 而非泛 VLA；"parallel multimodal attention"、"symmetrically" 等字眼**不存在** | 🔴 **虚构** |
| 引文 2 | §2.2 过多视觉 token | "A typical VLA forward pass processes 1000+ visual tokens, of which fewer than 20% are semantically relevant to the current action step." | 论文 §3.2 实际写 "The excessive quantity of visual tokens dilutes the model's attention, making it difficult for the policy to concentrate on critical manipulation regions."；**"1000+" 与 "<20%" 两个具体数字都不存在** | 🔴 **虚构（含编造数字）** |
| 引文 3 | §2.3 任务无关噪声 | "Training signals are diluted by task-irrelevant background tokens, leading to spurious correlations and reduced generalization." | 论文 §1 实际写 "Abundant background information results in a low signal-to-noise ratio (low quality) during cross-modal interactions, where meaningful task-relevant signals are buried under environmental noise."；**"spurious correlations" 与 "reduced generalization" 等字眼不存在** | 🔴 **虚构** |
| 引文 4 | §4.5 Discussion | "Due to architectural constraints, VGGT features can only be injected at the policy stage to avoid disrupting the pretrained VLM, and their gradients are relatively weaker and less stable, which limits their performance upper bound." | **逐字匹配**（除标点与从句顺序微调），实际位于 **§4.3 Ablation Study "4) Visual Representation"** 而非项目报告标注的 §4.5 Discussion | ✅ 内容真实 / ⚠️ 章节标错 |

**严重度评估**：
- 引文 1–3 共同构成项目报告 03 §2 "三大瓶颈"叙事的全部支柱，全部带引号且声称引用论文章节——这是**学术写作中最严重的不规范类型**（伪造引文）。
- 引文 4 是 v2plus 论证 FocusVLA 路线劣等的"核心证据"，内容真实，但章节位置标错（§4.3 而非 §4.5）。
- 这条问题已被外部预审查发现，本审查独立验证确认其属实。

**对项目论证的影响**：
- v2plus 在递交材料 `1-2.申请书填写草稿.md` §4.2(5) 中转述 "FocusVLA 也使用了 VGGT，但其作法是把 VGGT 作为策略阶段的并行编码器，**作者自述梯度较弱且不稳定、性能上限受限**"——这部分内容**站得住**（依赖引文 4，引文 4 真实）。
- 但项目报告 03 §2 完整论证链（3 个瓶颈 → FocusVLA 设计动机 → v2plus 共鸣点）依赖虚构引文，**论证链需要重建**（不依赖虚构原文）。

### 1.2 🔴 P0 · FocusVLA 数字声明全部错误

**位置**：`研究报告/03_FocusVLA_中_VGGT_用法剖析.md` 多处 + 附录 A.1 LIBERO 主表 + §7.2.1 对比表

**核查项与结果**：

| 项目报告声明 | 标注位置 | 论文实际值 | 判定 |
|---|---|---|---|
| FocusVLA LIBERO 平均 SR = **97.4%** | 03 §6 第 5 行 + 附录 A.1 表 "FocusVLA (with VGGT)" 行 + §7.2.1 表 | Table 1 multi-weights 主结果 **98.7%**；single-weight **97.0%**；Table 2 "with VGGT only" 配置 **96.8%**（比 OpenVLA-OFT baseline 97.1% **低 0.3pp**） | 🔴 97.4% 在论文中找不到对应行 |
| FocusVLA "without VGGT" = **97.2%** | 03 附录 A.1 表 | Table 2 中 VLM+DS（DINOv2+SigLIP）= **98.4%**；VLM+DS+VLM（最佳）= **98.7%** | 🔴 错（"without VGGT"是 98.4% 不是 97.2%）|
| FocusVLA 加 VGGT 提升 **+0.2%**（97.2 → 97.4）| 03 附录 A.1 + §7.2.1 | 实际"加 VGGT"是 **−1.6%（98.4 → 96.8）或 −1.9%（98.7 → 96.8）**——加 VGGT 反而**降** | 🔴 方向反了——但实际结论"VGGT 在 FocusVLA 中无贡献"反而更强 |
| Modality Cascaded Attention 是 **"串行 H_A → H_AQ → H_V"** | 03 §3.1 公式 (3.1.1) + §6 8 维对比表 | 论文公式 (6)–(7) 显示三个 attention（H_A, H_AQ, H_V）**并行**计算然后**拼接**，不是串行管线 | 🔴 结构性误读——TL;DR + §3.1 + §6 对比表都需重写 |
| FocusVLA 最佳模型**用 VGGT**（默认假设）| 03 §A.1 主表 + §A.4 ablation | Table 2 最佳配置（98.7%）= **VLM+DS+VLM**，**不使用 VGGT**；VGGT 单独使用反而是最差配置（96.8%）| 🔴 论证反转：原叙事"FocusVLA 用 VGGT 提 +0.2%"误导；实际应叙事为"FocusVLA 自家最佳设置主动舍弃 VGGT，因为 policy 阶段并联无效——正是 v2plus 论点" |
| FocusVLA 真机：**ALOHA 双臂 6 任务 +8%** | 03 §A.5 | 论文 §5.1 实际为 **Realman 平台、3 任务**（Grasping fruit / Stacking cups / Placing left block），**每 variation 25 trials**（每任务总共 100 trials）| 🔴 平台 / 任务数 / 数量全错 |

**严重度评估**：这些数字遍布项目报告 03 全文，且部分（如 97.4%）已迁移到 `1-2.申请书填写草稿.md` 引文转述中。LIBERO 表数字错误对项目的"FSF SR vs FocusVLA SR"对比叙事影响重大。

**反转点（这是好消息）**：实际 LIBERO 数据**比项目原叙事更强地支持 v2plus 的差异化论证**：FocusVLA 加 VGGT 反而降 0.3pp（vs OpenVLA-OFT），而 SF 加 VGGT teacher 升 1.4pp——同一个 VGGT 用法不同**效果差距 1.7pp**（项目原文说"差 7×"的方向是对的，但具体数字应改）。

### 1.3 🔴 P0 · 关联资料索引中多篇论文作者错标"Anonymous"

**位置**：`立项调研/06_关联资料索引.md` § 1–§ 5 与 § 6.2

**问题**：项目把多篇有真实作者的论文标为 "Anonymous"，可能是因为项目从 OpenReview / ICLR-投稿期看到时是匿名状态，但论文已在 arXiv 公开发布作者信息。这是学术索引规范问题。

| 项目索引中标"Anonymous" | 论文 | 实际作者（前 3 位）| 来源 |
|---|---|---|---|
| [2] FastVGGT | arXiv:2509.02560 | You Shen, Zhipeng Zhang, Yansong Qu 等 | arXiv abstract page |
| [9] StreamVGGT | arXiv:2507.11539 | Dong Zhuo, Wenzhao Zheng, Jiahe Guo 等 | arXiv abstract page，论文实际标题为 "Streaming **4D** Visual Geometry Transformer" |
| [10] π³ | arXiv:2507.13347 | Yifan Wang, Jianjun Zhou, Haoyi Zhu 等（10 人）| arXiv abstract page |
| [11] VGGT-DP | arXiv:2509.18778 | Shijia Ge, Yinxin Zhang, Shuzhao Xie 等 | arXiv abstract page，实际标题 "VGGT-DP: Generalizable Robot Control via Vision Foundation Models" |
| [12] 3D-Mix | arXiv:2603.24393 | Bin Yu, Shijie Lian, Xiaopeng Lin 等（11 人）| arXiv abstract page |
| [14] AugVLA-3D | arXiv:2602.10698 | Zhifeng Rao, Wenlong Chen, Lei Xie 等 | arXiv abstract page，ICRA 2026 |
| [15] DepthVLA | arXiv:2510.13375 | Tianyuan Yuan, Yicheng Liu, Chenhao Lu 等 | arXiv abstract page |
| [22] SAM2Act | arXiv:2501.18564 | （未独立核查，但 ICRA 2026 接收的论文通常有作者）| —— |
| [23] SlotVLA | arXiv:2511.06754 | （未独立核查）| —— |
| [25] Averaging Trap | arXiv:2603.18342 | Yanchuan Tang, Taowen Wang, Yuefei Chen 等 | arXiv abstract page，实际标题 "Shifting Uncertainty to Critical Moments: Towards Reliable Uncertainty Quantification for VLA Model" |
| [27] I-FailSense | arXiv:2509.16072 | Clemence Grislain, Hamed Rahimi, Olivier Sigaud 等 | arXiv abstract page |

**另一个错标**：
- [24] SAFE：项目索引标作者 **"Kim, M. J., et al."**（这是 OpenVLA-OFT 的第一作者 Moo Jin Kim）。SAFE 实际作者：Qiao Gu, Yuanliang Ju, Shengxiang Sun 等（NeurIPS 2025）。**完全错标。**

**严重度评估**：申请书 §4.7 参考文献的 6 篇都有正确作者，但**研究报告 06 综述 + 索引 06 中至少 11 处错标作者**——这些错误如果迁移到论文写作时直接 cite，会产生新的学术规范问题。需要在阶段 2 修订时全部更正。

### 1.4 🔴 P0 · 申请书参考文献超过曦源限制

**位置**：`递交材料/1-2.申请书填写草稿.md` §4.7

**问题**：曦源申请书限参考文献 3-5 篇，当前 6 篇（[1] FocusVLA / [2] InSpire / [3] Spatial Forcing / [4] OpenVLA / [5] VGGT / [6] LIBERO-Plus）。

**核查项与建议**：6 篇引用的 arXiv ID + 作者 + 内容全部**真实有效**，但必须按曦源格式压到 5 篇。删除候选（按"对项目论证不可或缺度"从低到高）：

| 删除候选 | 引用次数 | 删除影响 | 建议优先级 |
|---|---|---|---|
| [4] OpenVLA（2406.09246）| §4.2(1) 仅 1 处 | 项目主 backbone 是 OpenVLA-OFT（arXiv:2502.19645，**未在参考文献中**），引用 OpenVLA 原版有点偏离 | ⭐ 推荐删 |
| [6] LIBERO-Plus | §4.2(6) | 主评测平台，若删则评测背景描述弱 | 不推荐删 |
| [1] FocusVLA | §4.2(5) 反例对照 | 当前是差异化论证核心，但**如阶段 2 改用 SF + M-None 代理路径，可弱化甚至删除 FocusVLA 引用** | 待阶段 2 决策 |

**进一步问题**：[4] OpenVLA 引用如果删除，应该补 [4'] OpenVLA-OFT（arXiv:2502.19645）才合理——因为申请书 §4.2(1) 实际描述的是 OpenVLA-OFT 的特性（"优化微调方式，并行解码，连续动作表示，速度与成功率提升"）。所以**建议将 [4] OpenVLA 替换为 OpenVLA-OFT**，仍 5 篇内。

### 1.5 🟡 P1 · 项目内部数字不一致

| 不一致项 | 出现位置 | 数值 | 应该统一为 |
|---|---|---|---|
| 总 GPU-h | 用户主任务 prompt | **370** | 阶段 1 独立估算：紧约束版 **~230**（04 §3）；全量需求 **~327**（04 §5 末小计）；370 是**全量需求 + ~13% buffer**。**用户 prompt 数字与 04 §5 全量数字大致一致**——并非矛盾，而是不同 scope 的预算。建议三份新文档统一表述为"紧约束版 230 GPU-h、全量版（含 stretch）370 GPU-h"。|
| 总 GPU-h | 04 §3 表格 | **~230** | 同上 |
| 总 GPU-h | 04 §5 末 小计 | **327** | 同上 |
| 总 GPU-h | 可行性 §C.2 | **~370**（紧约束可降到 230）| 同上 |
| 真机评测 cell 数 | 00 §6 H4 命题 | **9 cell** | 12 cell（3 任务 × (3 扰动 + 1 clean)）|
| 真机评测 cell 数 | 05 §6.1 + §6.4 | **12 cell** | 12 cell |
| 真机评测 cell 数 | 可行性 §C.5 表格 | **12 cell** | 12 cell |
| 真机评测 cell 数 | 01 §7.2 评测计数 | "9 cell ... + clean 3 cell × 30 episode = 360 episode" | 实际 12 cell 才对 |
| Focus mask 来源 | 00 §1（项目代号一句话）| "**双源**：方向词GT × DINOv2 CLS attention" | 阶段 3 新文档统一**直接定义 A+C**（不论述 B/SAM2）|
| Focus mask 来源 | 01 §3（Focus Mask Design）| 先列 **A/B/C 三源**再舍弃 B | 同上 |
| FocusVLA "局限数量" | 可行性 §B.3 末 | **"4 大局限"** | 论文 abstract 实际是 **3 个 bottlenecks** |

### 1.6 🟡 P1 · 项目对其他论文的方法描述偏差

**位置**：`立项调研/06_关联资料索引.md` 与 `研究报告/06_3D-aware_VLA_相关工作综述.md`

| 论文 | 项目描述 | arXiv abstract 实际内容 | 判定 |
|---|---|---|---|
| Evo-0（2507.00416）| "动态层选蒸馏，98.3%" | "implicit 3D spatial understanding via off-the-shelf visual geometry foundation model"——**未提"动态层选蒸馏"** | 🟡 项目方法描述偏差 |
| GP3（2509.15733）| "DUSt3R-based spatial encoder，97.8%" | "leverages multi-view input ... spatial encoder to infer dense spatial features"——**abstract 未明确提 DUSt3R** | 🟡 需查全文确认 |
| ShortcutLearning（2508.06426）| 索引 06 [29] 写"Robotic Manipulation Generalizes Better via Bidirectional Cross-Modal Counterfactuals" | 实际标题 "**Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation**"——**完全不同的标题** | 🔴 论文张冠李戴 |
| StreamVGGT（2507.11539）| "Streaming Visual Geometry Transformer" | 实际标题 "Streaming **4D** Visual Geometry Transformer" | 🟢 略偏 |
| FastVGGT（2509.02560）| "Training-Free Acceleration of Visual Geometry **Grounded** Transformer" | 实际标题 "Training-Free Acceleration of Visual Geometry Transformer" | 🟢 略偏 |

### 1.7 🟢 P2 · 项目反复声称"实验室已有真机"但全文档无型号

**位置**：项目几乎所有文档（00 §10 / 04 §7 / 05 §1.1 / 可行性 §C.4）

**问题**：项目反复声明"实验室已有真机硬件"，但所有文档都**没有指明具体型号**。`研究报告/05_真机平台与数据标注方案.md` §1.1 列出 6 种候选（SO-100 / Franka / AgileX Piper / xArm / UR5 / WidowX），但没说"实验室具体是哪一种"。

**核查**：用 web_fetch 访问 huggingface/lerobot，发现 LeRobot 主线 SDK 仅原生支持 **SO-100 / LeKiwi / Koch / HopeJR / OMX / Reachy2 / OpenARM / Unitree G1**，**不直接支持** Franka / xArm / UR / AgileX——后者需要团队自实现 adapter。

**严重度评估**：这是项目 P0 阶段（M0 D8）最大的单点风险。如果实验室真机是非 LeRobot 原生支持的型号，整个真机部分的工程预算（项目报告 05 估的 12-13 天人工）**会显著低估**——adapter 实现可能需要额外 1-2 周。这一信息应在 P0 D1 之前确认。

---

## 2. 文献核查总表

**覆盖**：L1 重点 18 篇全核查 + L2 次要 6 篇抽检（覆盖率约 8% L2）+ GitHub 仓库 5 个。每条文献列：编号 / 文献（中英文）/ arXiv ID / **项目内引用位置（文档+章节）**/ 真实性 / 转述准确性 / 关键问题 / 检索源。

### 2.1 L1 核心方法 5 篇

| # | 文献 | arXiv | 项目引用位置 | 真实性 | 转述准确性 | 关键问题 | 检索源 |
|---|---|---|---|---|---|---|---|
| L1-1 | Spatial Forcing: Implicit Spatial Representation Alignment for VLA | 2510.12276 | 申请书 §4.7 [3]；研究报告 01 全文；索引 06 § 1[1] | ✅ | ✅ | 作者、submitted Oct 14 2025、ICLR 2026、3.8× 加速均一致 | arxiv.org/abs/2510.12276 |
| L1-2 | FastVGGT: Training-Free Acceleration of Visual Geometry Transformer | 2509.02560 | 索引 06 § 1[2]；主方案 01 § 4；递交可行性 §C.1 | ✅ | ⚠️ | 作者非匿名（项目索引标 Anonymous 错）；标题略偏（论文无 "Grounded"）；4× 加速一致 | arxiv.org/abs/2509.02560 |
| L1-3 | VGGT: Visual Geometry Grounded Transformer | 2503.11651 | 申请书 §4.7 [5]；研究报告 02；索引 06 § 1[3] | ✅ | ⚠️ | 作者 6 人一致；CVPR 2025 一致；**"Best Paper" 标识 abstract 页未明确显示，需进一步外部核查 CVPR 颁奖记录** | arxiv.org/abs/2503.11651 |
| L1-4 | FocusVLA: Focused Visual Utilization for VLA | 2603.28740 | 申请书 §4.7 [1]；研究报告 03 全文；索引 06 § 1[4] | ✅ | 🔴 | **见 §1.1–§1.2 详细问题**：3/4 段引文虚构、LIBERO 数字错、Cascaded Attention 结构反、真机实验全错、最佳模型用 VGGT 假设反向 | arxiv.org/abs/2603.28740 + arxiv.org/html/2603.28740v1 |
| L1-5 | InSpire: VLA with Intrinsic Spatial Reasoning | 2505.13888 | 申请书 §4.7 [2]；索引 06 § 1[5] | ✅ | ✅ | 作者 7 人（Ji Zhang, Shihan Wu, Xu Luo, Hao Wu, Lianli Gao, Heng Tao Shen, Jingkuan Song）与项目索引一致；UESTC（项目所写"Lianli Gao 课题组"对应——abstract 页未明确机构，但 Lianli Gao 任职 UESTC 是公开信息）；方向词提示机制描述一致 | arxiv.org/abs/2505.13888 |

### 2.2 L1 强参考 13 篇

| # | 文献 | arXiv | 项目引用位置 | 真实性 | 转述准确性 | 关键问题 | 检索源 |
|---|---|---|---|---|---|---|---|
| L1-6 | OpenVLA-OFT（Fine-Tuning VLA Models: Optimizing Speed and Success）| 2502.19645 | 申请书 §4.7 [4]（**项目实际引的是 [4] OpenVLA 2406.09246，但 §4.2(1) 描述的是 OpenVLA-OFT**）；研究报告 01 § 0；索引 06 § 2[6] | ✅ | ⚠️ | 标题非 "OpenVLA-OFT" 而是 "Fine-Tuning VLA Models..."（OpenVLA-OFT 是方法名而非论文标题）；作者 Kim/Finn/Liang 与项目索引"Kim, M. J., et al."一致；LIBERO 76.5% → 97.1% 完全一致；RSS 2025 | arxiv.org/abs/2502.19645 |
| L1-7 | LIBERO-Plus: In-Depth Robustness Analysis of VLA | 2510.13626 | 申请书 §4.7 [6]；索引 06 § 2[7] | ✅ | ✅ | 7 维扰动一致（objects layout / camera viewpoints / robot initial states / language instructions / light conditions / background textures / sensor noise）；性能下降 95% → <30% 一致 | arxiv.org/abs/2510.13626 |
| L1-8 | LIBERO: Benchmarking Knowledge Transfer | 2306.03310 | 索引 06 § 2[8] | ✅ | ✅ | 作者 7 人（Liu, Zhu, Gao 等）一致；4 task suites × 130 tasks 一致 | arxiv.org/abs/2306.03310 |
| L1-9 | StreamVGGT（Streaming 4D Visual Geometry Transformer）| 2507.11539 | 索引 06 § 2[9]；研究报告 02 | ✅ | ⚠️ | 标题项目缺 "4D"；作者非匿名（Dong Zhuo et al.）；200ms/frame 数字 abstract 未提，但 "online manner"、"causal transformer"、KD 一致 | arxiv.org/abs/2507.11539 |
| L1-10 | π³: Permutation-Equivariant Visual Geometry Learning | 2507.13347 | 索引 06 § 2[10]；研究报告 02 | ✅ | ⚠️ | 作者非匿名（Yifan Wang et al. 10 人）；标题正确（项目报告 02 中也用 π³ 符号）；置换等变机制一致 | arxiv.org/abs/2507.13347 |
| L1-11 | VGGT-DP: Generalizable Robot Control via Vision Foundation Models | 2509.18778 | 索引 06 § 2[11]；研究报告 02；02 备选方案 §3.2 | ✅ | ⚠️ | 项目索引中标题"VGGT-based Diffusion Policy"——实际论文标题是 "Generalizable Robot Control via Vision Foundation Models"（接近但不准确）；作者非匿名（Shijia Ge et al.）；AAAI 2026 投稿（项目未注）；MetaWorld 一致 | arxiv.org/abs/2509.18778 |
| L1-12 | 3D-Mix for VLA: Plug-and-Play Module for Integrating VGGT-based 3D Information | 2603.24393 | 索引 06 § 2[12]；研究报告 02 + 06 | ✅ | ⚠️ | 作者非匿名（Bin Yu et al. 11 人）；标题项目接近；"VGGT 在 policy 阶段集成"一致 | arxiv.org/abs/2603.24393 |
| L1-13 | Large Pre-Trained Models for Bimanual Manipulation in 3D（项目称 AttentionVoxel）| 2509.20579 | 索引 06 § 2[13]；研究报告 04 | ✅ | ⚠️ | **论文标题中无 "AttentionVoxel"**——这是项目自起的代号；作者 Yurchyk/Chang/Dudek/Meger 与项目索引一致；Humanoids 2025 一致；21.9% 相对增益一致；DINOv2 CLS attention 用法一致 | arxiv.org/abs/2509.20579 |
| L1-14 | AugVLA-3D: Depth-Driven Feature Augmentation for VLA | 2602.10698 | 索引 06 § 2[14] | ✅ | ⚠️ | 作者非匿名（Zhifeng Rao et al.）；ICRA 2026；提交日 2026-02-11（合法 arXiv ID）；用 VGGT 提 depth-aware feature 一致 | arxiv.org/abs/2602.10698 |
| L1-15 | DepthVLA: Enhancing VLA with Depth-Aware Spatial Reasoning | 2510.13375 | 索引 06 § 2[15] | ✅ | ⚠️ | 作者非匿名（Tianyuan Yuan et al.）；用 depth transformer + MoE 设计一致 | arxiv.org/abs/2510.13375 |
| L1-16 | SAFE: Multitask Failure Detection for VLA | 2506.09937 | 索引 06 § 4[24] | ✅ | 🔴 | **作者全错**——项目索引标 "Kim, M. J., et al."（OpenVLA-OFT 第一作者），实际作者 Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, Florian Shkurti；NeurIPS 2025 | arxiv.org/abs/2506.09937 |
| L1-17 | Averaging Trap（Shifting Uncertainty to Critical Moments）| 2603.18342 | 索引 06 § 4[25]；主方案 01 §6.3 H3 理论辩护 | ✅ | ⚠️ | 标题项目所写"Averaging Trap"是项目自起代号；论文实际标题完整版（已比对）；作者非匿名（Yanchuan Tang et al.）；entropy averaging 问题与 max-pooling 解药描述一致 | arxiv.org/abs/2603.18342 |
| L1-18 | I-FailSense: Towards General Robotic Failure Detection with VLMs | 2509.16072 | 索引 06 § 4[27] | ✅ | ⚠️ | 作者非匿名（Clemence Grislain et al.）；VLM-based failure detection 一致；多环境 generalize 一致 | arxiv.org/abs/2509.16072 |

### 2.3 L2 抽检 6 篇

| # | 文献 | arXiv | 项目引用位置 | 真实性 | 转述准确性 | 关键问题 | 检索源 |
|---|---|---|---|---|---|---|---|
| L2-1 | SpatialVLA: Exploring Spatial Representations for VLA | 2501.15830 | 索引 06 § 6.1 | ✅ | ✅ | 作者 Qu et al.；RSS（项目无标 venue 但 abstract page 显示已发表 v5）；Ego3D Position Encoding + Adaptive Action Grids 一致 | arxiv.org/abs/2501.15830 |
| L2-2 | GP3: A 3D Geometry-Aware Policy with Multi-View Images | 2509.15733 | 索引 06 § 6.1 | ✅ | 🟡 | 作者 Qian et al. 一致；**项目说"DUSt3R-based" abstract 中未明确提**——需 PDF 全文核查 | arxiv.org/abs/2509.15733 |
| L2-3 | Evo-0: VLA Model with Implicit Spatial Understanding | 2507.00416 | 索引 06 § 6.1 | ✅ | 🔴 | **项目说"动态层选蒸馏，98.3%"**——abstract 实际方法是 "plug-and-play geometric module，implicit 3D spatial understanding"，**无"动态层选蒸馏"** | arxiv.org/abs/2507.00416 |
| L2-4 | On Robustness of VLA against Multi-Modal Perturbations（项目称 RobustVLA）| 2510.00037 | 索引 06 § 5[31] | ✅ | ⚠️ | 作者 16 人，项目索引"Guo, J., et al."与第一作者 Jianing Guo 一致；多模态扰动鲁棒性主题一致 | arxiv.org/abs/2510.00037 |
| L2-5 | Shortcut Learning in Generalist Robot Policies | 2508.06426 | 索引 06 § 5[29]（项目标题为"Robotic Manipulation Generalizes Better via Bidirectional Cross-Modal Counterfactuals"，**完全不同的论文标题**）| ✅（论文真实）| 🔴 | **论文张冠李戴**——项目引用的标题对应另一篇论文，arXiv:2508.06426 实际是 Xing et al. "Shortcut Learning in Generalist Robot Policies"，CoRL 2025；项目错把 Bidirectional Cross-Modal Counterfactuals 论文的标题套在这个 arXiv ID 上 | arxiv.org/abs/2508.06426 |
| L2-6 | HD-VGGT: High-Resolution Visual Geometry Transformer | 2603.27222 | 索引 06 § 6.2 + 研究报告 02 | ✅ | ⚠️ | 作者 14 人非匿名；dual-branch 一致；"896×896 支持"abstract 未明示 | arxiv.org/abs/2603.27222 |

### 2.4 GitHub 仓库可用性

| 仓库 | URL | Stars | License | 状态 |
|---|---|---|---|---|
| openvla/openvla | github.com/openvla/openvla | 6300+ | MIT | ✅ OpenVLA-OFT 已 release（2025-03-03 release note），含 LIBERO 训练脚本，HF 权重已发 |
| facebookresearch/vggt | github.com/facebookresearch/vggt | 未抓 | （未抓）| ✅ 真实（项目报告 02 引用 + abstract 引用作 facebookresearch/vggt） |
| mystorm16/FastVGGT | github.com/mystorm16/FastVGGT | 782 | （LICENSE.txt 存在，类型未抓到）| ✅ 真实，提供 eval_scannet.py、eval_custom.py + `--merging` 参数 |
| OpenHelix-Team/Spatial-Forcing | github.com/OpenHelix-Team/Spatial-Forcing | 238 | MIT | ✅ Official implementation of arXiv:2510.12276，含 openvla-SF 文件夹 + LIBERO 训练脚本 |
| huggingface/lerobot | github.com/huggingface/lerobot | 24.4k | Apache 2.0 | ✅ 主线支持 SO-100 / Koch / HopeJR / Reachy2 / Unitree G1 等；**不直接支持** Franka / xArm / AgileX / UR——后者需 adapter |

**判定**：5 个核心仓库**全部真实开源可用**，工程可行性论证站得住。但 LeRobot 真机平台依赖需在阶段 2 落实具体型号。

---

## 3. 方法学与可行性分模块评审

### 3.1 核心论证链稳健性

项目立论建立在两根支柱：

**支柱 A**："InSpire（显式 grounding）+ Spatial Forcing（隐式 3D 对齐）的叠加是文献空白点"。
- 评估：✅ 这条**站得住**。本审查全核 L1 重点文献 18 篇 + 抽检 L2 6 篇，**未发现** InSpire 显式方向词提示 + SF 中间层 3D 对齐 + focus mask 调制三层叠加的现有工作。Evo-0、3D-Mix、AugVLA-3D 都涉及"隐式 3D"但**无 InSpire 风格显式提示**；FocusVLA 涉及 focus 但**作用于 attention 计算**而非 align loss 监督端。这是真正的空白点。

**支柱 B**："v2plus 优于 FocusVLA，因为 VGGT 用法不同（SF 中间层 Teacher vs FocusVLA policy 阶段并联）"。
- 评估：⚠️ **数据层面支柱站得住，但叙事需要重写**。具体：
  - 数据反转 1：FocusVLA Table 2 中 "VGGT only" 配置 96.8% **比 OpenVLA-OFT baseline 97.1% 还低 0.3pp**——加 VGGT 反而**降**。
  - 数据反转 2：FocusVLA 最佳配置（98.7%）**主动舍弃 VGGT**（用 VLM+DS+VLM）。
  - 数据反转 3：FocusVLA 最佳（98.7%）反而**略高于** Spatial Forcing 98.5%（同表对比）——单看 LIBERO 总成绩 FocusVLA ≥ SF。
  - 但：项目核心论点"VGGT 在 policy 阶段并联无效"在 FocusVLA 论文自身的数据中**得到强化**——FocusVLA 作者自己舍弃 VGGT，且引文 4（§4.3 Ablation）作者自述"梯度弱不稳定限制性能上限"是**真实的**。
  - 修订方向：阶段 2 应把 differentiation 论证从"v2plus 优于 FocusVLA（成功率对比）"改为"v2plus 的 VGGT 用法（继承 SF Teacher 范式）比 FocusVLA 的 VGGT 用法（policy 并联）更有效——FocusVLA 自家数据证实加 VGGT 反而降 0.3pp，作者也明确承认架构限制"。

**结论**：核心论证链**不需要推翻**，但需要：
1. 删除全部带引号的 FocusVLA 引文 1–3（虚构），保留并修正引文 4 的章节标注（§4.3 而非 §4.5）。
2. 修正 LIBERO 数字（97.4 → 96.8 with VGGT only / 98.7 best；OpenVLA-OFT 97.1 不变；SF 98.5 不变）。
3. 重写 differentiation 叙事，强调 FocusVLA 自家数据反证 VGGT policy 并联无效。

### 3.2 5 个 H 假设的可证伪性

| H | 命题 | 阈值 / 显著性 | 可证伪性评估 |
|---|---|---|---|
| H1 仿真鲁棒性增益 | V3 vs V1 在 ≥ 2 维 SR drop 差 ≥ 5pp | p < 0.05, Bonferroni / 7 维 → 单维 α ≈ 0.007 | ✅ 操作化清晰、Bonferroni 修正存在、effect size 与文献先验匹配（SF 原论文报 long-horizon +3.8pp，叠加 InSpire 预期 +3-5pp 共 6-10pp） |
| H2 叠加效应 | V3 平均 SR > max(V2, +FSF only) + 1.5pp | 项目自承"退化为半实证"（用 M-None 代理 +FSF only）| 🟡 **可证伪性偏弱**：M-None 代理本质上是"SF 原版 + 不使用 InSpire 显式提示"，但 V2 是"使用 InSpire 但不用 FSF"——M-None 与 V2 都不直接对应"+FSF only"。建议阶段 2 改成 "V3 > V2 + 1.5pp"（去掉 +FSF only 比较，因为没有真正的对照） |
| H3 Failure Predictor | AUC ≥ 0.65, 5-fold CV | LightGBM + InSpire 离散方向词时序 | ✅ 操作化清晰，AUC 阈值合理，理论辩护（Averaging Trap，2603.18342）真实存在 |
| H4 真机鲁棒性增益 | V3 在 9 cell 平均 drop 比 V1 小 ≥ 5pp | p < 0.10（"真机 sample 少→放宽"）| 🟡 **统计 power 偏弱**：9 cell × 30 episode/cell = 270 episode/变体，且 9 cell 是项目内部不一致的数字（实际应为 12 cell）；估算 power ≈ 0.6 即 40% 概率漏检真效应。建议阶段 2 评估是否能增加到 40 episode/cell |
| H5 双源 focus 优于单源/均匀 | M-AC > max(M-A, M-C) + 1pp & M-AC > M-None + 1.5pp | p < 0.05 | ✅ 操作化清晰；但 effect size 1pp 偏乐观（attention guidance 类工作的常见 effect 0.5–2pp）；建议阶段 2 评估是否能用"非劣 + 趋势分析"代替单一阈值 |

**多重比较修正**：项目在 H1 使用 Bonferroni（7 维），单维 α ≈ 0.007——这是**正确做法**。H5 用的 1pp 阈值在 5 ablation 配置上没说明 multiplicity 处理——阶段 2 应明确。

### 3.3 工程 scope 对本科生合理性

**核心数字**（基于 04 §5 + 可行性 §C.2 + 主方案 01 §7）：

| 项目 | 紧约束版 GPU-h | 全量版 GPU-h | 备注 |
|---|---|---|---|
| P0 sanity | 8 | 8 | 必做 |
| P1 VGGT 离线缓存（一次性）| 11 | 11 | 必做 |
| P1 V1/V2/V3 LoRA × 3 seed | 95 | 95 | 必做 |
| P2 LIBERO-Plus 主评测（紧约束 300 inst/cell）| 158 | 158 | 必做（21 cell × 300 × 3 seed = 18900 episode）|
| P2 核心 ablation A1/A2/A3 | 70 | 70 | A3 是 H5 核心；其余可弹性 |
| P3 视觉 ablation + β/bg_w 粗扫 | 10 | 35 | A4/A5/A6 |
| P3 真机 fine-tune | 20 | 20 | 必做 |
| Stretch A7/A8 / 续作 teacher | 0 | 35 | 可砍 |
| P5 reviewer 重跑 / 修订 | 8 | 8 | 必做 |
| **合计** | **~230** | **~327–370** | 用户 prompt 370 = 全量 + buffer |

**预算匹配**：
- ¥6000 GPU 预算 / ¥26 平均价 ≈ **231 GPU-h**——**精准对应紧约束版 230**
- 全量 327 GPU-h × ¥26 ≈ ¥8500——超出 ¥6000 GPU 预算 42%
- 用户 prompt 370 GPU-h × ¥26 ≈ ¥9620——超出 GPU 预算 60%，但仍在 ¥10000 总预算内

**判定**：**紧约束版可执行，全量版超出 GPU 预算**。阶段 2 应明确"申请书写 230 GPU-h"（与 ¥6000 GPU 预算自洽），全量版作为内部 stretch 目标。

**缓存磁盘空间**：650 GB（bf16）/ 325 GB（fp8 量化）——可行。学校共享存储 + 备份云冷存储（OSS 约 ¥30/月 / 100 GB，1 年 ¥360 在 ¥400 实验耗材预算内）覆盖。

**真机时间预算**：
- 数据采集 4-5 天（3 任务 × 50 demo）
- 评测 5 天（12 cell × 3 变体 × 30 episode ≈ 1080 inference 共享 360 物理 reset）
- 数据处理 + 标注 2-3 天
- 累计 **12-13 天净人工**——与项目 05 §7.1 估算一致，对 2 人本科生在 P1-P3 分散承担**可行**

**对本科生合理性总评**：紧约束版 scope **可行**，前提：(1) 实验室真机型号在 P0 D1 前确认；(2) 用户 prompt 中的 370 GPU-h 表述应统一为 230（紧约束基线）+ 注明全量为 stretch；(3) 砍 A7/A8 + 砍 teacher 备选项（锁 FastVGGT 单一）以释放 ablation GPU-h。

**给阶段 2 的砍 scope 候选**（按"必做度从低到高排序"，方便阶段 2 取舍）：
- 候选 a：砍 A7（融合策略 F2/F3/F4 扫描）+ A8（方向词粒度扫描）——释放 35 GPU-h
- 候选 b：砍 teacher 备选项（StreamVGGT / HD-VGGT）——锁 FastVGGT 单一选项，释放 ~10 GPU-h（备选 teacher 不缓存）
- 候选 c：A4/A5 从全扫降到 2 个值粗扫——释放 ~20 GPU-h（项目已采用）
- 候选 d：A6 视觉利用 ablation 砍一半（保留 token dropout 1 个强度）——释放 ~12 GPU-h
- 候选 e：仿真主评测 instance 数从 300 进一步降到 200——释放 ~50 GPU-h，但 statistical power 显著下降，不推荐

### 3.4 关键外部依赖代码可用性

见 §2.4：5 个核心仓库全部真实可用。

**强项**：
- OpenVLA-OFT 已 release（2025-03-03）
- Spatial-Forcing 官方实现已 release，含 LIBERO 训练脚本
- FastVGGT 提供 Python 接口 + 加速参数
- LeRobot 提供完整真机 SDK + 多平台支持

**潜在风险**：
- LeRobot 主线**不直接支持** Franka / xArm / AgileX / UR——如果实验室真机是这几类，需自实现 adapter（增加 1-2 周工程量）
- FocusVLA 论文代码**截至 2026-05-27 未公开**（项目报告 03 附录 B.1 已注明）——但 FocusVLA 是对比对象不是依赖，影响有限

### 3.5 实验室真机单点风险

见 §1.7：项目反复声明"实验室已有"但全文档无型号。

**严重度评估**：
- 这是 **P0 阶段最大单点风险**。如果实验室真机是 LeRobot 主线支持型号（SO-100 / Koch / Reachy2 等），项目 P0 D8 流程能跑通；如果是 Franka / xArm / AgileX 等需要 adapter，整个真机部分时间预算（项目估 12-13 天）将至少 +1-2 周。
- 单点风险因为**真机数据采集 + 评测 = H4 假设的全部基础**——P0 阶段如发现真机不可用或型号不匹配，会导致 H4 整个假设无法验证，项目 novelty 损失一半（"真机扰动评测"是 v2plus vs v2 的关键差异点）。

**修订建议**：
- 阶段 2 应在 plan 中要求：**在阶段 3 备份原文档前 + 写新文档前，团队人工确认实验室真机具体型号 + 借用时段 + SDK 兼容性**。
- 阶段 3 新文档（详情/简要/申请书）应统一表述为"实验室具备 [具体型号或最小要求] 真机平台，与 LeRobot 兼容"（或注明 adapter 计划）。
- 申请书 §4.6 已有基础部分应明确写"指导教师所在实验室具备 [具体型号待补] 真机机器人平台"，**避免向曦源评审递交时暴露空白**。

---

## 4. 项目内部不一致问题（按多文档间矛盾汇总）

| # | 问题 | 矛盾位置 | 建议统一为 |
|---|---|---|---|
| 4.1 | 总 GPU-h（230 / 327 / 370 三个值）| 04 §3 / 04 §5 / 可行性 §C.2 / 用户 prompt | 紧约束版 230（与 ¥6000 GPU 预算自洽）；全量版 327 作内部 stretch；370 = 全量 + buffer 仅用户 prompt 内部 |
| 4.2 | 真机评测 cell 数（9 / 12）| 00 §6 H4 / 01 §7.2 / 主方案 § H4 命题 vs 05 §6.1 / 可行性 §C.5 | 12 cell（3 任务 × (3 扰动 + 1 clean)）—— 项目内多数文档已用 12，仅 H4 命题写 9，需修订 |
| 4.3 | Focus mask 来源（双源 vs 三源舍 B）| 00 §1（"双源"）vs 01 §3（先列 A/B/C 再舍 B）| 阶段 3 新文档直接定义 A+C 双源；SAM2 排除决策不再保留（属决策档案语言）|
| 4.4 | 真机 demo 训练规模 vs 评测规模混淆 | 00 L77 "3 任务 × 50 demo × 3 扰动 = 360 episode" 表述混淆采集与评测 | 阶段 3 新文档统一："**150 demo 用于训练采集**（3 × 50）；**360 episode 用于评测**（12 cell × 30）；这是两个独立的数据集" |
| 4.5 | FocusVLA "局限数量"（3 vs 4）| 可行性 §B.3 "4 大局限" vs 论文 abstract 3 个 bottlenecks | 3 个 bottlenecks（与论文 abstract 一致）|
| 4.6 | OpenVLA vs OpenVLA-OFT 引用 | 申请书 §4.7 [4] OpenVLA（2406.09246）vs §4.2(1) 描述 OpenVLA-OFT 特性 | [4] 改为 OpenVLA-OFT（arXiv:2502.19645），与正文描述一致 |
| 4.7 | FastVGGT 标题（带/不带 "Grounded"）| 索引 06 § 1[2] "FastVGGT: Training-Free Acceleration of Visual Geometry **Grounded** Transformer" vs arXiv abstract "Training-Free Acceleration of Visual Geometry Transformer" | 删 "Grounded" 与原文一致 |
| 4.8 | StreamVGGT 标题（带/不带 "4D"）| 索引 06 § 2[9] "StreamVGGT: Streaming Visual Geometry Transformer" vs arXiv "Streaming 4D Visual Geometry Transformer" | 加 "4D" 与原文一致 |
| 4.9 | "AttentionVoxel" / "Averaging Trap" 代号 vs 论文真实标题 | 索引 06 多处把项目代号当作论文标题 | 阶段 3 文档分清"项目内代号"（仅内部备忘）与"论文真实标题"（参考文献 + 综述中使用）|

---

## 5. 附录 A：FocusVLA 引文逐字比对原始证据

**来源**：FocusVLA, arXiv:2603.28740, 抓取 `https://arxiv.org/html/2603.28740v1` 全文做关键词搜索后整理。

### A.1 引文 1（架构偏差）

**项目报告 03 §2.1**（带引号标"FocusVLA 的描述"）：

> "Existing VLA models inherit the parallel multimodal attention from VLMs, where text queries and visual tokens attend to each other symmetrically. This creates an architectural bias that prevents the model from focusing on action-relevant visual regions."

**论文原文（§1 Introduction，Figure 1 caption 附近）**：

> "VLA-Adapter introduces an efficient design to incorporate VLM information into the policy, yet its mixed attention mechanism inadvertently creates a 'structural shortcut'. This bias encourages the model to derive task-relevant signals primarily from learnable action queries while largely bypassing the critical spatial details embedded in visual features."

**差异分析**：
- 主体偷换：项目报告说"Existing VLA models"（泛指）；原文说"VLA-Adapter"（特指）
- 关键术语虚构："parallel multimodal attention"、"text queries and visual tokens attend to each other symmetrically"、"architectural bias"——这些短语在原文中**完全不存在**
- 原文实际术语：'structural shortcut'、'mixed attention mechanism'、'bypassing the critical spatial details'——项目报告 03 完全没引用这些

### A.2 引文 2（过多视觉 token）

**项目报告 03 §2.2**：

> "A typical VLA forward pass processes 1000+ visual tokens, of which fewer than 20% are semantically relevant to the current action step."

**论文原文（§3.2 附近）**：

> "The excessive quantity of visual tokens dilutes the model's attention, making it difficult for the policy to concentrate on critical manipulation regions."

**差异分析**：
- 量化数字编造：项目报告中的"1000+ tokens"与"<20% relevant"在原文中**不存在**
- 概念部分对应：原文也讨论"excessive visual tokens"，但**没有给具体百分比**

### A.3 引文 3（任务无关噪声）

**项目报告 03 §2.3**：

> "Training signals are diluted by task-irrelevant background tokens, leading to spurious correlations and reduced generalization."

**论文原文（§1）**：

> "Abundant background information results in a low signal-to-noise ratio (low quality) during cross-modal interactions, where meaningful task-relevant signals are buried under environmental noise, hindering action accuracy."

**差异分析**：
- 关键术语虚构："training signals are diluted"、"spurious correlations"、"reduced generalization"——这些短语在原文中**不存在**
- 原文实际术语：'low signal-to-noise ratio'、'buried under environmental noise'、'hindering action accuracy'

### A.4 引文 4（VGGT 梯度弱）

**项目报告 03 §4.5**：

> "Due to architectural constraints, VGGT features can only be injected at the policy stage to avoid disrupting the pretrained VLM, and their gradients are relatively weaker and less stable, which limits their performance upper bound."

**论文原文（§4.3 Ablation Study, "4) Visual Representation" subsection）**：

> "VGGT exhibits the most powerful spatial modeling ability. However, due to architectural constraints, to avoid disrupting the pretrained VLM, VGGT features can only be injected at the policy stage. As a result, their gradients are relatively weaker and less stable, which limits their performance upper bound."

**差异分析**：
- **核心内容逐字匹配** ✅（项目报告引文与原文除标点与从句顺序微调外完全一致）
- **章节位置标错** ❌：项目报告标 §4.5 Discussion，实际在 §4.3 Ablation Study
- 引文前的转折句"VGGT exhibits the most powerful spatial modeling ability."项目报告未引用，可能影响语境理解

### A.5 LIBERO 数字逐项核查

**论文 Table 1（Multi-weights setting + Single-weight setting）**：

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | **97.1** |
| Spatial Forcing | 99.4 | 99.6 | 98.8 | 96.0 | **98.5** |
| FocusVLA (multi-weights) | — | — | — | — | **98.7** |
| FocusVLA (single-weight) | — | — | — | — | **97.0** |

**论文 Table 2（FocusVLA 内部 ablation）**：

| Configuration | Avg |
|---|---|
| FocusVLA with VGGT (only) | 96.8 |
| FocusVLA with VLM+DS (DINOv2+SigLIP) | 98.4 |
| FocusVLA with VLM+DS+VLM（best）| 98.7 |

**项目报告 03 §A.1 + §7.2.1 的数字**：

| Method | Spatial | Object | Goal | Long | Avg | 项目标注 |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | ✅ 一致 |
| FocusVLA (without VGGT) | 97.7 | 98.4 | 97.8 | 94.7 | **97.2** | 🔴 实际 VLM+DS = 98.4，VLM+DS+VLM = 98.7 |
| FocusVLA (with VGGT) | 97.9 | 98.6 | 98.0 | 95.2 | **97.4** | 🔴 实际"with VGGT only"= 96.8 |
| Spatial Forcing | 98.7 | 99.0 | 98.5 | 97.6 | **98.5** | 🟡 Avg 一致但分项可能略偏（论文 SF 行 99.4/99.6/98.8/96.0）|

**判定**：项目报告 03 §A.1 主表中 FocusVLA 行的数字与论文 Table 1/Table 2 **均无法对应**——既不是 multi-weights 98.7，也不是 with VGGT 96.8，也不是 VLM+DS 98.4。最可能的来源是项目从其他二手综述（或更早版本草稿）抄写，未与原文核对。

### A.6 Cascaded Attention 结构

**论文公式 (6)-(7)**：三个 attention 模块 H_A、H_AQ、H_V **并行计算**然后**拼接**（concatenation）后传入下一层。

**项目报告 03 §3.1 公式**：

$$\mathrm{H}_A \to \mathrm{H}_{AQ} \to \mathrm{H}_V$$

标"串行三段管线"——**结构性反读**。

### A.7 真机实验

**论文 §5.1**：
- 平台：**Realman**（中国 Realman 机器人公司）
- 任务数：**3**
  - "Grasping fruit"（背景变化）
  - "Stacking cups"（空间变化）
  - "Placing left block"（目标变化）
- 每 variation 25 trials；每任务总共 100 trials

**项目报告 03 附录 A.5**：
> "FocusVLA 在 ALOHA 双臂机器人上 6 个任务的成功率提升约 8%（相对于 OpenVLA-OFT）"

**判定**：平台错（Realman 不是 ALOHA 双臂）、任务数错（3 不是 6）、trials 错（25/var 不是项目报告的笼统"8% 提升"）。

---

## 6. 附录 B：检索关键词、URL、查证过程

### B.1 已抓取的 arXiv 摘要页（L1 + L2）

| arXiv ID | URL | 用途 |
|---|---|---|
| 2603.28740 | https://arxiv.org/abs/2603.28740 | FocusVLA 元数据 |
| 2603.28740 | https://arxiv.org/html/2603.28740v1 | FocusVLA 全文（核心 4 引文比对 + Table 1/2 数字 + 真机实验）|
| 2510.12276 | https://arxiv.org/abs/2510.12276 | Spatial Forcing 元数据 |
| 2503.11651 | https://arxiv.org/abs/2503.11651 | VGGT 元数据 |
| 2509.02560 | https://arxiv.org/abs/2509.02560 | FastVGGT 元数据 |
| 2505.13888 | https://arxiv.org/abs/2505.13888 | InSpire 元数据 |
| 2502.19645 | https://arxiv.org/abs/2502.19645 | OpenVLA-OFT 元数据 |
| 2510.13626 | https://arxiv.org/abs/2510.13626 | LIBERO-Plus 元数据 |
| 2306.03310 | https://arxiv.org/abs/2306.03310 | LIBERO 元数据 |
| 2507.11539 | https://arxiv.org/abs/2507.11539 | StreamVGGT 元数据 |
| 2507.13347 | https://arxiv.org/abs/2507.13347 | π³ 元数据 |
| 2603.24393 | https://arxiv.org/abs/2603.24393 | 3D-Mix 元数据 |
| 2509.20579 | https://arxiv.org/abs/2509.20579 | AttentionVoxel/Bimanual 元数据 |
| 2506.09937 | https://arxiv.org/abs/2506.09937 | SAFE 元数据 |
| 2603.18342 | https://arxiv.org/abs/2603.18342 | Averaging Trap / Shifting Uncertainty 元数据 |
| 2509.18778 | https://arxiv.org/abs/2509.18778 | VGGT-DP 元数据 |
| 2510.13375 | https://arxiv.org/abs/2510.13375 | DepthVLA 元数据 |
| 2509.16072 | https://arxiv.org/abs/2509.16072 | I-FailSense 元数据 |
| 2602.10698 | https://arxiv.org/abs/2602.10698 | AugVLA-3D 元数据 |
| 2501.15830 | https://arxiv.org/abs/2501.15830 | SpatialVLA 元数据 |
| 2603.27222 | https://arxiv.org/abs/2603.27222 | HD-VGGT 元数据 |
| 2509.15733 | https://arxiv.org/abs/2509.15733 | GP3 元数据 |
| 2507.00416 | https://arxiv.org/abs/2507.00416 | Evo-0 元数据 |
| 2510.00037 | https://arxiv.org/abs/2510.00037 | RobustVLA 元数据 |
| 2508.06426 | https://arxiv.org/abs/2508.06426 | ShortcutLearning 元数据（**发现张冠李戴**）|

### B.2 GitHub 仓库

| URL | 用途 |
|---|---|
| https://github.com/openvla/openvla | OpenVLA-OFT 主 backbone 可用性 |
| https://github.com/mystorm16/FastVGGT | FastVGGT 加速实现 |
| https://github.com/OpenHelix-Team/Spatial-Forcing | SF 官方代码 |
| https://github.com/huggingface/lerobot | 真机 SDK 兼容性 |

### B.3 未完成核查（阶段 2 决策项）

下列文献因 arXiv ID 在项目索引中标"xxxx"占位符或时间预算紧张未做核查，建议阶段 2 或团队人工补查：

- Gaze-Regularized VLA（项目索引 06 § 3[16]，arXiv:2603.xxxx 占位符）
- AutoFocus-IL（项目索引 06 § 3[17]，arXiv:2511.xxxx 占位符）
- FPC-VLA（项目索引 06 § 4[26]，无 arXiv ID）
- VLA-Risk（项目索引 06 § 4[28]，无 arXiv ID，标 ICLR 2026）
- CF-VLA（项目索引 06 § 5[30]，arXiv:2512.xxxx 占位符）
- L2 剩余约 50 篇相关工作（未抽检）

### B.4 检索过程方法论

- **arXiv 主渠道**：用 `WebFetch` 抓 abstract 页（拿元数据）与 HTML 全文页（做关键词搜索 + 表格抽取）。abstract 页提供作者 / 提交日 / 摘要；HTML 全文提供章节级搜索。
- **PDF 回退**：未启用，因 abstract + HTML 已覆盖所有 L1 文献。
- **GitHub 回退**：用 `WebFetch` 抓仓库 README，确认 description / stars / license / 是否含 LIBERO 训练脚本等关键工程信号。
- **未触发的回退**：Semantic Scholar / Google Scholar / 出版社页面——本审查全部 L1 文献都能通过 arXiv 直接访问，未触发多渠道回退。带占位符 arXiv ID 的文献因时间预算未启用 Semantic Scholar 搜索，建议阶段 2 或团队人工补查。

---

## 7. 总结：给阶段 2 的关键输入

1. **学术规范一票否决项必须修订**：FocusVLA 3 段虚构引文必须删除；4 项数字必须修正；至少 10 处"Anonymous"作者必须补正确。
2. **核心论证不需要推翻**：v2plus 的差异化论证（vs FocusVLA）在数据层面**反而被强化**——FocusVLA 自家最佳设置主动舍弃 VGGT，且加 VGGT 反而降 0.3pp。叙事重写方向明确。
3. **scope 砍方向**：紧约束版 230 GPU-h 与 ¥6000 GPU 预算自洽，全量 327-370 超预算；阶段 2 应砍 A7/A8 + 砍 teacher 备选项（锁 FastVGGT）+ A4/A5 粗扫 → 释放约 50-60 GPU-h，给主评测 + 真机 fine-tune 留 buffer。
4. **5 个 H 的可证伪性**：H1 / H3 / H5 操作化清晰；H2 / H4 偏弱（H2 退化为半实证，H4 power 约 0.6）——阶段 2 决策是收敛到 3 个 H（删 H2 或 H4 之一）还是保留 5 个 H 但调整阈值。
5. **真机型号是 P0 单点风险**：阶段 3 备份原文档前必须由团队人工确认。
6. **文档间一致性**：GPU-h（统一 230 紧约束）/ 真机 cell（统一 12）/ Focus mask 来源（直接定义 A+C，不论述 SAM2 排除决策）等需在阶段 3 新文档中落实统一表述。
7. **申请书参考文献**：必须从 6 篇压到 5 篇——推荐方向：删 [4] OpenVLA 改为 [4'] OpenVLA-OFT（与正文一致）+ 阶段 2 决策是否保留 [1] FocusVLA（取决于差异化叙事是否还以 FocusVLA 为反例对象）。

%% Review completed 2026-05-27 %%
