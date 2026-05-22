---
create time: 2026-05-22
tags:
  - 曦源项目
  - 立项调研
  - 二次调研
  - INDEX
---

# 二次调研 · 索引

> [!info] 总览
> 本目录是曦源项目 Efficient VLA **二次调研**产出，聚焦一次调研推荐主线 **「SmolVLA (0.45B) + AAC (推理时) + RobustVLA-LoRA-r32 (训练时) 双方向贯通」** 的深度可行性、扩展相关工作、完整科研技术路线。一次调研产出见 `../调研/` 目录。

## 文件清单（本目录）

| 文件 | 主题 | 字数 | 作者 |
|------|------|------|------|
| [[00_主线深度可行性分析]] | 训练显存账本 / 推理延迟叠加 / 一致性 / 风险 | ~5500 字 | 主线程 |
| [[01_AAC相关工作扩展]] | action chunking / 推理调度 / 1-step diffusion 17+ 篇论文 | ~5800 字 | 二次调研 Agent 1 |
| [[02_RobustVLA相关工作扩展]] | adversarial training / robust BC / PEFT 12 篇论文 | ~10000 字 | 二次调研 Agent 2 |
| [[03_SmolVLA代码可注入点分析]] | lerobot 11 个源文件 + 16 个 file:function 级注入点 | ~36000 字符 | 二次调研 Agent 3 |
| [[04_交叉领域调研_efficient_robust]] | efficient × robust 11 篇 + 假设审判 | ~7000 字（含补强校验 callout） | 二次调研 Agent 4 + 主线程补强 |
| [[05_完整科研技术路线]] | Phase 0-5 细化（代码模块/实验/消融/go-no-go） | ~5500 字 | 主线程 |
| [[06_创新点定位与论文卖点]] | 3 层 contribution / 2 个标题候选 / 评审 Q&A | ~3500 字 | 主线程 |

## 核心结论速览

### 主线可行性：**A-** （Go decision）

- ✅ 训练显存账本：24 GB 4090 即可跑全套（LoRA r=32 路径），80 GB A100 极宽裕
- ✅ 推理延迟可控：从 SmolVLA 20 Hz 目标提升到 30+ Hz
- ✅ 主线在 421 篇候选中**无实质抢做**（Agent 2 核对）
- ✅ 16 个 AAC + RobustVLA 注入点已定位到 file:function 级（Agent 3）
- ⚠️ RobustVLA 是 JAX/Flax 实现，需 1 周翻译到 PyTorch（已纳入 Phase 1 时间表）
- ⚠️ LoRA r=32 而非 r=16（采纳 Agent 2 警告）
- ⚠️ 训练-推理一致性需 Phase 3 消融验证（属 contribution 一部分，非 blocker）

### 三层 contribution

| 层 | 内容 | 优先级 |
|----|------|------|
| **Systems** | AAC entropy × async queue 耦合调度 | 必做 |
| **Empirical** | < 1B VLA 上首次 worst-case 多模态训练 + LoRA + capacity/data scan | 必做 |
| **Theoretical** | worst-case δ 与 entropy uncertainty 的 dual relation | 弱版必做、强版 stretch |

### 核心 hypothesis

> 「**小模型在 worst-case 数据增广范式下能达到与大模型相当甚至更优的相对鲁棒增益**」 — Agent 4 + 补强校验判定为**条件成立**（极小数据 + 看相对增益 + worst-case 增广 + robust generalization gap 四个条件均满足）。基于此修订 framing。

### 论文目标

- **按学校算分体系调整投稿目标**：主投 **NeurIPS 2027**（CCF A，5 分），备投 **ICLR 2028**（CCF A，5 分），保底 **ICRA 2028**（CCF B，3 分），期刊路径 **TRO**（中科院 1 区，5 分）。CoRL/IROS 因不入 CCF 评分体系不作为目标。详见 [[05_完整科研技术路线]] §5.3
- 标题候选 A（稳）：*Capacity-Efficient Multi-Modal Robust VLA: Coupling Adaptive Inference with Worst-Case Regularization in 0.45B Parameters*
- 标题候选 B（赌）：*The Capacity-Robustness Duality: Why a 0.45B VLA Can Match a 3.3B VLA Under Multi-Modal Perturbation*

## 阅读路径

### 我是首次进入二次调研的人
1. [[00_主线深度可行性分析]] §6 可行性最终结论（2 分钟）
2. [[06_创新点定位与论文卖点]] §1 三层 contribution（3 分钟）
3. [[05_完整科研技术路线]] 时间表汇总段（2 分钟）

### 我是导师 / 评审委员会
1. [[00_主线深度可行性分析]] §1 训练显存账本（核账可行性）
2. [[06_创新点定位与论文卖点]] §3 与近邻工作差异化 + §4 评审 Q&A
3. [[05_完整科研技术路线]] Phase 1/2/3 go-no-go criteria

### 我准备启动 Phase 0
1. [[05_完整科研技术路线]] Phase 0 具体步骤（含 git clone 命令）
2. [[03_SmolVLA代码可注入点分析]] §1-2 仓库结构 + 关键 class 定位
3. [[00_主线深度可行性分析]] §1.4 三档 GPU 可行性对照

### 我准备做 Phase 1（AAC 集成）
1. [[03_SmolVLA代码可注入点分析]] §3 AAC 8 个注入点
2. [[01_AAC相关工作扩展]] §5 借鉴清单（特别是 AQC / RTC / FASTER）
3. [[05_完整科研技术路线]] Phase 1 详细设计

### 我准备做 Phase 2（RobustVLA 集成）
1. [[03_SmolVLA代码可注入点分析]] §4 RobustVLA 8 个注入点 + §5 LoRA attach
2. [[02_RobustVLA相关工作扩展]] §5 借鉴清单（Free-AT / LEAF / Lipschitz）
3. [[05_完整科研技术路线]] Phase 2 详细设计（含 JAX→PyTorch 翻译策略）

### 我准备写论文
1. [[06_创新点定位与论文卖点]] §2 标题/摘要候选 + §3 卡位差异化
2. [[05_完整科研技术路线]] Phase 5 论文章节框架 + figure list
3. [[02_RobustVLA相关工作扩展]] §3 论文卡片 + [[01_AAC相关工作扩展]] §3 论文卡片 → 直接做 related work section

## 关键新发现（vs 一次调研）

1. **AQC** (arxiv 2605.05544) 的 advantage-baseline 可替代 AAC 手调 ξ 阈值 ⭐
2. **RTC + TT-RTC** (arxiv 2506.07339, 2512.05964) 补齐 SmolVLA async queue 缺失，与 AAC **正交**可叠加 ⭐⭐
3. **Free-AT** (arxiv 1904.12843) 双循环改单循环，把 RobustVLA 显存从 +5～8 GB 压到 +0 GB ⭐⭐⭐
4. **LoRA r=16 在少样本下不够（Few-Shot Adv LoRA 2505.15130）→ 升级到 r=32** ⭐⭐⭐
5. **RobustVLA 是 JAX/Flax 实现**（Agent 3 代码侦察）→ 需 1 周翻译到 PyTorch ⭐⭐⭐
6. **SmolVLA 主要 class**：SmolVLAPolicy:226、VLAFlowMatching:541、SmolVLMWithExpertModel:72（Agent 3 file:line 级定位）
7. **421 篇候选中无实质抢做**（Agent 2 核对）—— 主线 framing 安全
8. **"小模型鲁棒训练增益更大"假设**：从一次调研的"强 claim"修正为"条件成立"（Agent 4 + 补强）

## 关键调整（vs 一次调研）

| 维度 | 一次调研 | 二次调研修订 | 原因 |
|------|---------|------------|------|
| LoRA rank | r=16 | **r=32** | Few-Shot Adv LoRA 警告 |
| 显存预算 | "全参 + 不到 24 GB" 估计 | 精确分项账本（详见 [[00_主线深度可行性分析]] §1） | 训练-推理峰值澄清 |
| RobustVLA 翻译 | 未提 | **JAX→PyTorch 1 周** 预留 | Agent 3 代码事实 |
| 论文 framing | "首次小模型鲁棒增益更大" | **"capacity-efficient × multi-modal robust"** | 假设审判修订 |
| 必做消融 | 模糊提到 | 明确 capacity scan + data scan + r-sweep | Agent 4 建议 |
| RTC / TT-RTC | 未覆盖 | **必装** baseline + AAC 正交 | Agent 1 发现 |
| Free-AT fallback | 未覆盖 | **显存安全网** | Agent 2 发现 |

## 调研元信息

- 调研日期：2026-05-22
- 数据来源：4 个 general-purpose agent（并行）+ 主线程综合
- 工具状态：
  - WebFetch 后端故障（mimo-v2.5-pro 模型不可用）全程
  - **改用 gh CLI + curl 直接抓 arxiv abs / arxiv html / GitHub raw**，验证 100% 可用
  - 4 个补强 agent 被启动但在 Windows 后台进程中卡死（输出文件 0 字节、mtime 不更新），由主线程接管最终综合
- 检验：Agent 4 报告中**2 篇 arxiv ID 标错**（2402.00667, 2005.10247），主线程已用 curl 核实并加补强 callout，主结论不受影响
- 总字数：综合报告 ~37000 字（含表格、公式、wikilink）

## 上下文链接

- 一次调研 INDEX：[[../调研/INDEX]]
- 一次调研总览：[[00_总览与立项建议]]（位于 ../调研/）
- 立项原始方向：[[立项方向]]（位于 ..）
- References 论文笔记：见一次调研 INDEX
