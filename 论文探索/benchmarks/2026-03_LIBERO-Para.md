---
title: "LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models"
authors: "Chanyoung Kim, Minwoo Kim, Minseok Kang, Hyunwoo Kim, Dahuin Jung"
year: "2026"
venue: "arXiv preprint (2026-03)"
arxiv: "2603.28301"
status: "已精读"
tier: "⭐⭐⭐ 语言鲁棒性最佳专测"
tags: [literature, T1D, benchmark, robustness-eval, LIBERO-family, paraphrase, language]
---

# LIBERO-Para — Paraphrase 鲁棒性专测（22-52pp drop）

> [!info] 元信息
> - **作者**：Chanyoung Kim, Minwoo Kim, Minseok Kang, Hyunwoo Kim, Dahuin Jung（CAU HAI Lab）
> - **arXiv**：[2603.28301](https://arxiv.org/abs/2603.28301)
> - **GitHub**：[cau-hai-lab/LIBERO-Para](https://github.com/cau-hai-lab/LIBERO-Para)（CC BY-NC-ND 4.0）
> - **核心定位**：首个**指令语言侧**的 fine-grain 鲁棒性专测

## 📄 这个 benchmark 是做什么的

LIBERO-Para 独立扰动**动作表达**和**物体指代**两个语言成分，定量评估 VLA 对**语义等价、表面不同**的指令是否鲁棒。提出 **PRIDE** 指标（Paraphrase difficulty using semantic and syntactic factors）量化 paraphrase 难度。

## 🧠 我的评估

### 评测维度（关键！）
- **纯语言侧 paraphrase**：
  - Action verb 变换（"pick up" ↔ "grasp" ↔ "lift"）
  - Object reference 变换（同义词 / 描述语 / 类别词）
- **与其他 benchmark 关系**：
  - LIBERO-Plus 的 language dim 太弱（论文称 VLA 几乎忽略语言）→ Para 强制测细颗粒度
  - VLA-Risk 的 I_obj/I_act/I_spa 测「不可行指令」（语义错），Para 测「同义改写」（语义对）——**完全互补**
  - LIBERO-PRO 的 Semantic 维度 < Para 的细颗粒度

### 任务规模
- 沿用 LIBERO base + paraphrase variants
- 难度分级用 PRIDE 自动量化（不是人为分级）

### 评估 baseline 模型
- **7 个 VLA 配置，参数 0.6B → 7.5B**
- 覆盖小模型到大模型对比

### 关键 finding
- **22-52 pp SR 退化**——主流 VLA 都中招，无一幸免
- **80-96% 失败来自 planning-level trajectory divergence**（不是 execution error）——意味着模型从一开始就**理解错任务**
- **Object lexical variation**（同义词替换）破坏力最大
- 结论：「VLA 是表面字符串匹配，不是语义 grounding」

### 复现成本
- **GPU**：与 LIBERO 同级（24GB 推理）
- **开源**：✅ GitHub 已 release（CC BY-NC-ND 4.0——**禁商用 + 禁改**，学术 OK）
- **评测时间**：单 VLA 全套约 1 天
- **优势**：构造方式简单（只改 instruction 字符串），完全不需要重新渲染仿真——**最快**

### 是否适合曦源课题
- ✅✅ **强烈推荐 TOP-2/3 候选（语言侧故事）**
- 理由：
  1. **复现门槛最低**（不动仿真，只改指令）
  2. **故事独特**：「VLA 是字符串匹配 vs 语义 grounding」——很容易写出对比实验
  3. 7 个 baseline（0.6B-7.5B 跨度大）已公开，可直接对比
- ⚠️ 风险：
  1. **故事单一**——只测语言，如果曦源课题想全面 robust，需配合 Plus/PRO
  2. CC BY-NC-ND 协议禁止改动数据——只能跑评测，不能 fine-tune
- **决策建议**：若曦源走「VLA 语言侧鲁棒性」主线（与 VLA-Risk infeasible instruction 呼应），Para 是必选

## 🔗 关联
- [[VLA-Risk_ICLR2026]]：互补——VLA-Risk 测语义错，Para 测语义对
- [[2025-10_LIBERO-Plus]]：Plus 的 language dim 被认为太弱，Para 是其升级
- [[2025-10_LIBERO-PRO]]：PRO 的 Semantic 维度的细颗粒度版
- [[C-x_instruction_perturbation]]：指令侧鲁棒性子方向
