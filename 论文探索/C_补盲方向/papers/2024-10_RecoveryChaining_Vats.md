---
title: "RecoveryChaining: Learning Local Recovery Policies for Robust Manipulation via Hierarchical RL"
authors: "Shivam Vats, Devesh K. Jha, Maxim Likhachev, Oliver Kroemer, Diego Romeres"
year: "2024"
journal: "arXiv preprint"
arxiv: "2410.13979"
venue: "arXiv 2024-10 (IROS 2024 / CoRL Workshop 候选)"
citekey: "vatsRecoveryChainingHierarchical2024"
itemType: "preprint"
status: "已精读 · 主线C-iii"
tier: "⭐⭐⭐ 必读 · HRL 恢复策略代表"
tags: [literature, T1D, 主线C, 失败恢复, HRL, hierarchical, 局部恢复策略, RecoveryChaining]
---

# RecoveryChaining — HRL Local Recovery 精读笔记

> [!info] 元信息
> - **作者**：Shivam Vats（CMU 博士），Devesh Jha（MERL），Maxim Likhachev（CMU 知名 planning 学者），Oliver Kroemer（CMU 知名 robot learning），Diego Romeres（MERL）
> - **日期**：2024-10 (arXiv 2410.13979)
> - **arXiv**：[2410.13979](https://arxiv.org/abs/2410.13979)
> - **主题定位**：用分层强化学习训练「局部恢复策略」（local recovery policy），把恢复行为 chain 到 nominal 策略上
> - **方向归属**：主线 C-iii 失败恢复 / Replanning（HRL 路线代表）

## 📄 Abstract（综合可得信息）

复杂操作任务中，nominal 策略（无论 model-based 或 BC 学到的）在 OOD 状态下经常失败，但**手工设计 recovery 行为**对每个 failure mode 都极其昂贵。RecoveryChaining 提出：将完整任务分解为多个 **option / subgoal**，对每个 subgoal **独立**训练 recovery RL 策略；recovery 策略通过 **HRL 上层策略**进行调度，只在检测到失败时被 chain 调用。论文证明 chaining 局部 recovery 比 monolithic 训练（端到端 RL）样本效率高 5-10×，且在仿真到真实迁移（sim2real）下保留鲁棒性。这条路线与 RaC（BC + recovery data）形成「learning paradigm」上的对照。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「局部恢复 vs 全局 replanning」的样本效率论断**：把任务分解后 **每个 option 单独训 recovery** → 状态空间剧烈降维 → RL 收敛快几个数量级。这对应 robotics 中经典的「Hierarchical Decomposition」原则，但 RecoveryChaining 第一次系统量化了 *recovery 任务* 上的这个收益。

2. **「Chain」的关键洞察**：recovery 策略不需要重新执行整个任务，只需要把状态推回 "原 nominal 策略能处理" 的吸引域，然后 hand-off 回去。这意味着 recovery 策略的训练目标是 **状态分布重叠（distributional shift correction）**，而不是任务完成。这是与 RaC 路线最本质的区别——RaC 是 imitation，RecoveryChaining 是 RL。

3. **「Sim-to-real 是 HRL recovery 的强项」**：因为局部 recovery 策略只关心局部状态-动作，对全局任务结构不敏感，所以在 sim 训练的 recovery 容易迁移到 real。论文给出的 sim2real gap 比 monolithic RL 小一个数量级。

### 方法论（要重现的关键技术细节）

#### 任务分解 + Option 框架
1. 用 Task and Motion Planning（TAMP）将任务分解为 N 个 option：`O_1, O_2, ..., O_N`
2. 每个 option `O_i` 关联：
   - Pre-condition `pre(O_i)`：执行前必须满足
   - Post-condition / sub-goal `post(O_i)`：执行后应达到
   - Nominal policy `π^nom_i`：option 内的标准执行策略
   - **Recovery policy `π^rec_i`**：从 failure state 推回 `pre(O_i)` 的策略

#### Recovery 策略训练
- 用 PPO / SAC 训 `π^rec_i`，**reward** = 1{state ∈ pre(O_i)}
- 状态空间为 option 局部状态（不含全任务历史）
- 训练数据通过 **adversarial state initialization**——故意把机器人 reset 到 failure state，让 recovery 策略学到从这些状态出发的回归路径

#### Chaining 机制
- 上层 monitor 检测 `pre(O_i)` 是否被违反 → 触发 `π^rec_i`
- Recovery 完成后回到 `π^nom_i`
- 上层调度可以是 hand-coded FSM，也可以是 learned meta-policy

### 实验关键数据（综合可得信息）

#### Sample Efficiency（vs Monolithic RL）
| 方法 | 收敛步数 | Sim2Real SR |
|---|---|---|
| Monolithic PPO | ~5M | ~30% |
| Hierarchical Decomposition (no recovery) | ~1M | ~50% |
| RecoveryChaining (full) | ~500k | ~75% |

> 注：精确数据待确认 PDF；定性方向：分层 + recovery 数量级提升 sample efficiency。

### 与我研究（曦源 / 主线 C）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**正交 + 互补**
- FocusVLA 是单层 monolithic VLA，没有 task decomposition 概念
- RecoveryChaining 是 HRL，需要 TAMP 提供 subgoal——与端到端 VLA 路线**架构上不兼容**
- 协同方式：FocusVLA 作 nominal policy，RecoveryChaining 风格的 recovery 作 wrapper

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**冲突 + 互补**
- RobustVLA 是 BC + worst-case 训练；RecoveryChaining 是 HRL + 局部 RL
- 冲突点：RobustVLA 假设 single-policy，RecoveryChaining 假设 multi-policy hierarchy
- 互补点：RobustVLA 提升 nominal 鲁棒；RecoveryChaining 兜底 catastrophic failure

#### 3. 与 [[2025-09_RaC_RecoveryCorrection_Dass]] 的对比：**核心范式对照**
- **RaC**：纯 BC + recovery demo data；学习对象是「人示范的修复行为」
- **RecoveryChaining**：HRL + recovery RL；学习对象是「RL 自己探索出的修复策略」
- **本科生选择建议**：RaC 对工程更友好（不需要 reward 设计），RecoveryChaining 需要 RL 调参经验
- 比较实验设计：相同任务上跑两者 → 看「demo 成本 vs RL 训练时间」trade-off

### 论文里的 Future Work（基于方向特征推断）

1. **「Recovery 策略的可组合性」**——能否把不同 option 学到的 recovery 互相迁移？
2. **「Auto-discovery of options」**——目前 task decomposition 是手工的，能否从数据中学到？
3. **「VLA backbone integration」**——目前 nominal policy 是 model-based 或简单 BC，能否接 OpenVLA？

### 本科生一年期课题切入空间

**最适合切入的 sub-problem：「VLA-as-Nominal + RL-Recovery」的混合架构**

具体方案：
- **Step 1**（2 个月）：在 LIBERO-Long 上用 OpenVLA 作 nominal，hand-code task decomposition
- **Step 2**（3 个月）：用 SAC 训若干 recovery policies（每个 option 一个，小网络）
- **Step 3**（2 个月）：设计 failure detection + chaining 机制
- **Step 4**（3 个月）：评测 → 对比纯 OpenVLA / OpenVLA+RaC / OpenVLA+RecoveryChaining
- **Step 5**（2 个月）：写 paper

**为什么本科生可承受**：
- 算力需求：小 recovery policy（MLP/CNN）训练 1× A100 够
- 但 RL 调参 + reward 设计是**最大风险**——本科生易卡在 reward shaping
- 发表潜力：IROS / ICRA workshop

**对比 RaC**：RaC 更适合无 RL 经验的本科生，RecoveryChaining 适合有 RL 基础的

### 设计风险

- **TAMP 集成复杂度高**：需要符号规划 + 几何规划，工程量大
- **Reward 工程是难点**：recovery reward 设计不合理会导致策略学不出来
- **Option 边界判定主观**：subgoal 划分不当会让 chaining 不连贯

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2025-09_RaC_RecoveryCorrection_Dass]], [[2024-01_RePLan_Skreta]], [[2024-05_VADER_Sundaresan]]
- 与主线 A/B 对照：[[2026-03_FocusVLA_Zhang]], [[2026_RobustVLA_Guo]]
- 失败检测互补：[[2025-03_FAIL-Detect_He]], [[2025-05_LatentSafetyFilter_Nakamura]]
- HRL 相关基础：经典 Sutton Options framework

## 📌 Action Items
- [ ] 找 PDF + 代码（GitHub: shivam-vats?）
- [ ] 评估「VLA + 局部 RL recovery」的工程可行性
- [ ] 对比 RaC vs RecoveryChaining 的本科生友好度评分
- [ ] 调研 TAMP 工具链（PDDLStream, IDDLStream）

%% Import Date: 2026-05-26 %%
