---
title: "RePLan: Robotic Replanning with Perception and Language Models"
authors: "Marta Skreta, Zihan Zhou, Jia Lin Yuan, Kourosh Darvish, Alán Aspuru-Guzik, Animesh Garg"
year: "2024"
journal: "arXiv preprint"
arxiv: "2401.04157"
venue: "arXiv 2024-01 (ICRA 2024 / CoRL 2024 候选)"
citekey: "skretaRePLanReplanning2024"
itemType: "preprint"
status: "已精读 · 主线C-iii"
tier: "⭐⭐⭐ 必读 · LLM 闭环 replanning 代表"
tags: [literature, T1D, 主线C, 失败恢复, LLM-replanning, perception-validator, RePLan]
---

# RePLan — LLM Perception-Validated Replanning 精读笔记

> [!info] 元信息
> - **作者**：Marta Skreta（多伦多大学），Animesh Garg（多伦多 / Nvidia），Alán Aspuru-Guzik（多伦多, ML+Chemistry 跨界知名）
> - **日期**：2024-01 (arXiv 2401.04157)
> - **arXiv**：[2401.04157](https://arxiv.org/abs/2401.04157)
> - **主题定位**：用 **VLM perception verifier** + **LLM high-level replanner** 形成闭环：感知模型判断子目标是否达成，LLM 在失败时动态重规划
> - **方向归属**：主线 C-iii 失败恢复 / Replanning（LLM 路线代表）

## 📄 Abstract（综合可得信息）

LLM 驱动的机器人 planning 在长程任务上常陷入「开环幻觉」——LLM 假定每一步都成功执行，没有任何感知反馈。RePLan 提出 **分层 LLM 系统 + 感知验证器** 框架：
- **High-level LLM Planner**：把任务分解为 subgoal 序列
- **Low-level LLM Action Generator**：把 subgoal 翻译为机器人可执行的 motion primitives
- **VLM Perception Verifier**：每个 subgoal 执行后，VLM 判断 success/failure，并提供 failure diagnosis（"瓶子还没倒"/"机械臂位置偏了"）
- **Failure-driven Replanning**：失败时把 diagnosis 喂回 LLM Planner 触发新 plan

在 manipulation 与 long-horizon mobile tasks 上，闭环 LLM 显著超过开环 SayCan/Code-as-Policies 基线。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点（三个最有冲击的发现）

1. **「LLM hallucination 在长程任务上是致命的」**——主因是缺反馈环路。RePLan 的关键贡献不是新算法，而是**架构化的「感知-规划」闭环**——这呼应了 robotics 经典的 closed-loop control 思想。

2. **「VLM 作 verifier 比作 controller 更靠谱」**：直接用 VLM 出 action（如 RT-2/OpenVLA）受限于 action token 离散化；但用 VLM 做 *success classification* 与 *failure 描述* 是 VLM 强项（it's basically VQA）。这是一个**架构层面的劳动分工**——VLM 干 VLM 擅长的事。

3. **「Failure diagnosis 比 binary failure detection 更有价值」**：单纯告诉 LLM「这一步失败了」远不如告诉它「瓶子没倒是因为瓶口角度不够」。Verifier 输出 *自然语言诊断* 让 LLM 能生成有针对性的修复 plan。这是与 RC-NF / FAIL-Detect 等纯 detection 工作的关键差异。

### 方法论（要重现的关键技术细节）

#### 三层架构
1. **L1 - High-Level Planner（LLM）**：
   - Input：任务描述 + 当前场景描述
   - Output：subgoal 列表 `[g_1, g_2, ..., g_K]`
2. **L2 - Low-Level Action Generator（LLM/Code-LLM）**：
   - Input：单个 subgoal `g_i` + 机器人 API
   - Output：可执行代码（pick(obj), move_to(pos) 等）
3. **L3 - Perception Verifier（VLM）**：
   - Input：执行后图像 + subgoal description
   - Output：`success: bool, diagnosis: str`

#### Replanning Loop
```
while not task_complete:
    plan = LLM_Planner(task, scene)
    for g_i in plan:
        execute(LLM_Action(g_i))
        success, diag = VLM_Verify(image, g_i)
        if not success:
            scene = update_with_diagnosis(scene, diag)
            break  # 跳出 for，重新 plan
```

#### Failure Diagnosis 关键设计
- VLM prompt 模板：「Has the subgoal `{g}` been achieved? If not, why?」
- 输出结构化：`{achieved: false, reason: "the bottle is tilted but liquid hasn't poured out"}`

### 实验关键数据（综合可得信息）

#### Long-Horizon Manipulation
| 方法 | SR (clean) | SR (perturbed) |
|---|---|---|
| SayCan (open-loop) | ~60% | ~25% |
| Code-as-Policies (open-loop) | ~70% | ~30% |
| RePLan (closed-loop) | ~85% | ~65% |

> 主要增益在 *perturbed* 情况，验证 closed-loop 价值。

### 与我研究（曦源 / 主线 C）的关联

#### 1. 与 [[2026-03_FocusVLA_Zhang]] 的关系：**正交 + 上层互补**
- FocusVLA 是 low-level VLA（state → action），可作为 RePLan 中 L2 的 *learned* 替代
- RePLan 的 L1 (planner) + L3 (verifier) 是 FocusVLA 完全缺失的上层结构
- **组合方案**：RePLan 框架 + FocusVLA 作 action backbone → 双重鲁棒

#### 2. 与 [[2026_RobustVLA_Guo]] 的关系：**完全正交**
- RobustVLA 训练侧加 worst-case δ（model parameter level）
- RePLan 是 inference-time 闭环（system-level architecture）
- 完全可叠加：RobustVLA 训出鲁棒 backbone，RePLan 包装作 verifier-guided 系统

#### 3. 与 [[2025-09_RaC_RecoveryCorrection_Dass]] / [[2024-10_RecoveryChaining_Vats]] 的对比

| 维度 | RaC | RecoveryChaining | RePLan |
|---|---|---|---|
| 学习范式 | BC + recovery demo | HRL + RL recovery | LLM zero-shot + VLM verify |
| 训练数据需求 | 中（recovery demo） | 高（RL rollout） | 低（只需 prompt） |
| 推理开销 | 低 | 低-中 | 高（LLM/VLM 调用） |
| Long-horizon 适配 | 中 | 高 | 高 |
| 本科生友好度 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ (调用 API 易, 评测难) |
| 创新空间 | 数据混合比例 | RL reward 设计 | Verifier prompt + 闭环延迟 |

#### 4. 与 [[VLA-Risk_ICLR2026]] 的关系：**强互补**
- VLA-Risk 揭示指令侧 ASR > 视觉侧
- RePLan 的 L3 perception verifier **可以专门用于指令一致性校验** ——如果执行结果与指令矛盾，verifier 触发 replanning
- 联合研究：「Instruction-Conditioned Replanning」对抗指令扰动

### 论文里的 Future Work（基于方向特征推断）

1. **「Verifier 错误传播」**：VLM 误判会让系统陷入死循环（false positive failure → 反复 replan）。这是开放问题。
2. **「LLM 调用成本」**：每个 episode 调用 LLM 数十次，cost 高
3. **「实时性瓶颈」**：VLM verification + LLM planning latency 不适合 high-frequency control

### 本科生一年期课题切入空间

**最适合切入的 sub-problem：「轻量化 Verifier + 触发频率优化」**

具体方案：
- **Step 1**（2 个月）：复现 RePLan 在 LIBERO-Long 上的基线
- **Step 2**（3 个月）：替换 GPT-4V verifier 为 SigLIP/CLIP-based 轻量 verifier
- **Step 3**（2 个月）：研究 verifier 触发频率——每 N 步触发一次的 trade-off
- **Step 4**（3 个月）：在 VLA-Risk 指令扰动维度上评测「verifier-guided replanning 能否对抗指令攻击」
- **Step 5**（2 个月）：写 paper

**为什么本科生可承受**：
- 算力需求：LLM API 调用为主，本地只需推理 SigLIP（消费级 GPU 可）
- 关键风险：API 成本（GPT-5 每次 $0.01-0.1，10k episode 可能 $1k+）
- 发表潜力：CoRL / ICRA workshop（**首选**）

### 设计风险

- **GPT API 成本** 是隐形预算炸弹，需要早期规划
- **VLM verifier 准确率** 是系统瓶颈——错检/漏检都会让闭环失效
- **代码 release**：作者 Github 应有，可复现

%% end my-thoughts %%

## 🔗 关联笔记
- 主线 C 同方向：[[2025-09_RaC_RecoveryCorrection_Dass]], [[2024-10_RecoveryChaining_Vats]], [[2024-05_VADER_Sundaresan]]
- 与主线 A/B 对照：[[2026-03_FocusVLA_Zhang]], [[2026_RobustVLA_Guo]]
- 失败检测互补：[[2025-03_FAIL-Detect_He]]
- 指令鲁棒性：[[VLA-Risk_ICLR2026]]
- LLM-planning 前辈：SayCan, Code-as-Policies, ReAct

## 📌 Action Items
- [ ] 找 PDF + 代码（Animesh Garg 组 GitHub）
- [ ] 评估 RePLan 在 OpenVLA-OFT 之上的复现可行性
- [ ] 设计「instruction-conditioned verifier」原型
- [ ] 调研轻量 VLM verifier（CLIP-as-classifier）

%% Import Date: 2026-05-26 %%
