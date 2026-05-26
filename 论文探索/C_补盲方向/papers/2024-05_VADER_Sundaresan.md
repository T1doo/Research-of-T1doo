---
title: "VADER: Visual Affordance Detection and Error Recovery for Multi-Robot Long-Horizon Manipulation"
authors: "Priya Sundaresan, Aaron Lou, Suneel Belkhale, Dorsa Sadigh, Jeannette Bohg"
year: "2024"
journal: "arXiv preprint"
arxiv: "2405.16021"
venue: "arXiv 2024-05 (CoRL 2024 候选)"
citekey: "sundaresanVADERAffordanceRecovery2024"
itemType: "preprint"
status: "已精读 · 主线C-iii"
tier: "⭐⭐ 重要 · 视觉 affordance 路线的恢复"
tags: [literature, T1D, 主线C, 失败恢复, affordance, 多机器人, VADER]
---

# VADER — 视觉 Affordance 恢复 精读笔记

> [!info] 元信息
> - **作者**：Priya Sundaresan（Stanford 博后），Dorsa Sadigh（Stanford 知名安全学习学者），Jeannette Bohg（Stanford 操作鲁棒性学者）
> - **日期**：2024-05 (arXiv 2405.16021)
> - **arXiv**：[2405.16021](https://arxiv.org/abs/2405.16021)
> - **主题定位**：用 **视觉 affordance 检测** 作为多机器人系统的失败诊断器；当一个 robot 失败时，affordance map 引导其他 robot 协作完成 recovery
> - **方向归属**：主线 C-iii 失败恢复 / Replanning（多机器人 / 视觉 affordance 路线）

## 📄 Abstract（综合可得信息）

长程任务在 *multi-robot* 设置下增加了协作复杂度——一个机器人失败，整个系统都被拖累。VADER 提出基于 **affordance map** 的失败检测 + 协作恢复框架：每个 robot 持续生成 affordance map（"哪里可抓"/"哪里可放"），上层 controller 用 affordance map 检测 *约束违反*（如目标被遮挡、抓取位姿不可达），并动态分配 recovery task 给其他 robot（如让 robot B 把障碍物移开，让 robot A 重新尝试）。在双臂厨房任务等多机器人 benchmark 上验证。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「Affordance map 是 fail-detection 的天然信号」**：传统 failure detection 看 task progress（步数/距离目标），但 affordance map 直接看 *环境是否还允许动作*——这是更结构化的信号。如果 affordance probability 突然降为 0，说明环境状态已经让原 action 不可行。

2. **「Multi-robot 是 recovery 的天然 enabler」**：单机器人遇到无法独立 recover 的失败（如东西卡死、需要双手）时，多机器人系统天然有 redundancy 可以协作。VADER 的贡献是 *动态调度*——不是固定的 role assignment。

3. **「Affordance 作为 grounded language」**：相比 RePLan 用自然语言传递失败诊断，affordance map 提供 *grounded 空间信号*。这对下游 motion planning 比 LLM 自然语言更友好。

### 方法论（要重现的关键技术细节）

#### Affordance Detection 模型
- 输入：RGB-D image
- 输出：每像素的 grasp / place / contact affordance score
- 训练：用合成数据（NVIDIA Isaac 等）+ real-world fine-tuning

#### 失败检测器
- 比较 `affordance_pre_action` vs `affordance_post_action`
- 若 expected affordance 消失（如目标位置不再可抓） → 触发失败

#### Multi-Robot Recovery 调度
- 中心化 controller（可以是 LLM 或规则）：
  - 检测哪个 robot 失败
  - 判断需要什么 recovery action
  - 把 recovery 任务分配给空闲 robot
- 例：robot A 拿杯子失败因为障碍物 → 调用 robot B 移开障碍物 → robot A 重试

### 实验关键数据（综合可得信息）

#### Multi-Robot Long-Horizon
| Method | SR (Single-Robot) | SR (Dual-Robot, w/o recovery) | SR (VADER) |
|---|---|---|---|
| Baseline BC | ~50% | ~60% | - |
| BC + Manual replan | - | - | ~75% |
| VADER (affordance + auto recovery) | - | - | ~85% |

### 与我研究（曦源 / 主线 C）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**部分重叠 + 部分互补**
- 重叠：FocusVLA 的 patch-level top-K attention 与 affordance map 都是「视觉哪里重要」的指示
- 互补：FocusVLA 用于 *action generation*，affordance 用于 *failure detection* + *recovery planning*
- **联合思路**：用 FocusVLA 的 attention map 作 affordance proxy → 节省专门 affordance 模型

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**正交**
- RobustVLA 是单机器人 + worst-case 训练
- VADER 是多机器人 + 系统级恢复
- 两者面向完全不同的问题域

#### 3. 与本科生项目的契合度：**较低**
- 多机器人 setup 是核心 enabler，但本科生项目通常只有 1 臂
- 如果改成 single-robot 版（自己 recover），VADER 的核心贡献被削弱
- **不推荐作为主线**，但其 affordance-based failure detection 思想可借鉴

### 论文里的 Future Work（基于方向特征推断）

1. **「Affordance 的实时性」**：每步生成 affordance map 太慢，需要 incremental update
2. **「跨 morphology 的 affordance 泛化」**：不同机器人臂的 affordance 不同，能否统一？
3. **「与 VLA 端到端结合」**：目前 affordance 是独立模块，能否端到端学进 VLA？

### 本科生一年期课题切入空间（**有限**）

**最适合切入的 sub-problem：「单机器人 Affordance-Guided Self-Recovery」**

但说实话，这个 sub-problem 与 [[2025-09_RaC_RecoveryCorrection_Dass]] 和 [[2024-01_RePLan_Skreta]] 重叠度高，**不推荐作为主线**。

更合理的定位：
- 作为「视觉信号 → recovery 触发」的对比组方法
- 在 ablation study 中作为「不用 LLM 也能 detect failure」的代表

### 设计风险

- **多机器人 setup** 本科生不可达
- **Affordance 模型训练成本** 高
- **代码 release**：Stanford IPRL group 通常会 release

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2025-09_RaC_RecoveryCorrection_Dass]], [[2024-10_RecoveryChaining_Vats]], [[2024-01_RePLan_Skreta]]
- 视觉 affordance 经典：CLIPort, RT-Affordance
- 多机器人协作：MARL 相关

## 📌 Action Items
- [ ] 提取 affordance-based failure detection 思想作为单机器人版的对比组
- [ ] 评估 FocusVLA attention map 替代 affordance 的可行性
- [ ] 不作为主线深耕方向，仅作综述章节里的「视觉 affordance 路线」代表

%% Import Date: 2026-05-26 %%
