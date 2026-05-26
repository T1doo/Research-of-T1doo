---
title: "FAST: Efficient Action Tokenization for Vision-Language-Action Models"
authors: "Karl Pertsch, Kyle Stachowicz, Brian Ichter, et al. (Physical Intelligence + UC Berkeley + Stanford)"
year: "2025"
venue: "RSS 2025"
arxiv: "2501.09747"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, tokenization, DCT, BPE]
---

# FAST tokenizer

> [!info] 元信息
> Physical Intelligence + UC Berkeley + Stanford 2025.01。**针对 VLA 动作 tokenization 的根本性改进**：用 DCT (Discrete Cosine Transform) + BPE 替代 RT-2/OpenVLA 的 naive bin-by-bin 离散化。压缩率提升 10×（4 step 动作从 28 token → 3 token），训练速度提升 5×，关键是**精细操作任务（Mobile ALOHA Towel-Folding、Bimanual Insertion）成功率从 30% 跃升到 70%+**。Karl Pertsch 同时是 Octo / OpenVLA 主作之一，本论文相当于「OpenVLA 的官方升级」。

## 📄 TL;DR（100-150 字）

FAST 指出现有 VLA 动作 tokenization 的根本缺陷：（1）bin-by-bin 离散化对高度相关的动作维度独立处理，浪费表征；（2）chunk 长度增加时 token 数线性爆炸；（3）连续平滑信号被强行离散化损失精度。解决方案：先对 action chunk 做 DCT 频域变换（保留低频系数），再用 BPE 算法在频域 token 上学一个 word vocabulary。结果：典型 4-step chunk 从 28 token 压缩到 3 token，OpenVLA-FAST 训练快 5×，在精细操作任务上从 30% 提升到 70%+。FAST 已成为 2025+ VLA 的事实新标准。

## 🧠 我的思考

### 在 VLA 谱系中的位置

FAST 在 VLA 谱系里是**「2025 年最重要的方法学改进」之一**——它解决了 RT-2 / OpenVLA 沿用至今的根本性瓶颈（动作离散化粒度差），且修复 cost 很低（只改 tokenizer，不动 VLM 主干）。

**动作表征演化时间线**：
- Gato（22.05）、RT-1（22.12）：μ-law / 256-bin 独立离散化；
- RT-2（23.07）、OpenVLA（24.06）：复用 RT-1 离散化 + 映射到 LLM vocab；
- Octo（24.05）、RDT-1B（24.10）：diffusion head 替代离散动作（continuous）；
- π0（24.10）：flow matching action expert（continuous）；
- **FAST（25.01）：在「保持离散动作」前提下做最大优化**；
- π0-FAST（同期）：FAST 应用于 π0；
- π0.5（25.04）：进一步集成 FAST。

引用关系：FAST 引用 RT-2、OpenVLA、Octo、π0 作为离散动作 baseline；被 NORA、π0.5、Gemini Robotics 等 2025 论文引用作为「new standard tokenizer」。FAST 团队就是 OpenVLA / Octo / π0 的核心人员（Karl Pertsch、Kyle Stachowicz），这是「同一团队迭代改进」的典型案例。

### 核心方法

**Naive 离散化的缺陷**（FAST 的 motivation）：
- OpenVLA: 4-step chunk × 7-DoF = 28 token；
- 28 token 自回归生成 → 慢；
- 256-bin 粒度对精细操作不够；
- 不同维度独立处理，浪费跨维度相关性；
- 不同 frequency 分量混杂，模型难学。

**FAST 的两阶段方案**：

**Stage 1: DCT 频域变换**：
- 对每个动作 chunk（如 4 step × 7-DoF）做 DCT；
- 保留 top-k 低频系数（典型 k=10-20），高频系数置零（人类操作本身低频为主）；
- 每个 DCT 系数离散化到 1024 bin；
- 得到稀疏的频域 token 序列（大部分维度被 zero 掉）。

**Stage 2: BPE compression**：
- 把频域 token 序列当 "text"，跑 BPE 算法学 vocabulary；
- 高频出现的 token pair 合并为新词；
- 学完 vocab 大小约 2048-4096；
- 同样的 4-step chunk 现在只需 3-6 token（10× 压缩）。

**反 tokenize**：BPE 反 merge → 频域系数 → IDCT → 时域 action chunk。

**集成到 VLA**：
- 不改 VLM 主干（Llama-2 / PaliGemma）；
- 把 FAST tokens 加到词表（unused token mapping）；
- 训练时只换 tokenizer，loss 形式不变；
- 推理时生成 FAST token 然后 detokenize。

**关键实验**：
- OpenVLA-FAST 训练快 5×（短序列效应）；
- Mobile ALOHA Towel-Folding：30% → 70%；
- Bimanual Insertion：成功率显著提升；
- 与 π0 (diffusion baseline) 性能相当但推理更快；
- 跨多个 VLA backbone 一致提升。

### 鲁棒性视角下的解读

FAST 对鲁棒性的贡献需要细致分析——它**不直接提升语义 / 视觉鲁棒性**，但**根本性改善精细操作的「物理执行鲁棒性」**：

**FAST 提升的鲁棒性维度**：
1. **执行精度鲁棒**：精细操作（毫米级对齐、薄物体抓取）从「prone-to-fail」变得「reliable」，这是 OpenVLA 范式最大的痛点之一被解决；
2. **chunk 一致性**：BPE merging 让常见动作 pattern（如「直线移动」「圆弧抓取」）成为单 token，模型生成更一致，减少抖动；
3. **DCT 低频偏置**：高频噪声被自然过滤，对训练数据的微小标注噪声鲁棒；
4. **推理速度提升**：5× 训练加速也意味着推理 token 数少 → 闭环频率提升 → 对动态扰动响应更快。

**FAST 不解决的脆弱性**：
1. **语义鲁棒性**：VLM 主干不变，对指令改写、新概念的鲁棒性不变；
2. **视觉 OOD**：视觉编码器不变；
3. **CoT / 推理能力**：不涉及；
4. **长时序记忆**：tokenization 改进不解决 context length 问题。

**与其他改进的可组合性**：
- FAST + ECoT：可能性高，CoT 输出 + FAST action token；
- FAST + π0：已实现（π0-FAST），证明可组合；
- FAST + LoRA fine-tune：直接可用。

**意义**：FAST 证明了「离散动作路线」（OpenVLA-style）不是死路——只要 tokenization 设计合理，可以达到 diffusion 路线（Octo / π0）的精度，且保持 LLM 自回归生成的所有优势（CoT 兼容、推理快）。这对曦源选择路线判断重要。

### 与曦源（鲁棒性主线）的关联

FAST 对曦源**极其重要**，建议作为**主 baseline 的改进对照**：

**作为 baseline**：
1. OpenVLA-FAST 权重应该在 HuggingFace 上（Physical Intelligence 维护），可直接使用；
2. 46GB 显存与 OpenVLA 等价（同 7B 模型）；
3. 鲁棒性实验对照非常清楚：OpenVLA vs OpenVLA-FAST 是「is tokenization the bottleneck?」的直接对照；
4. 强烈建议曦源**同时报告 OpenVLA 和 OpenVLA-FAST 两个 baseline**，说明现代 SOTA 的标准。

**作为研究方向启示**：
1. 如果曦源研究方向涉及「精细操作鲁棒性」，FAST 思路非常值得借鉴；
2. 可以研究 FAST + 鲁棒性扰动：DCT 频域过滤是否对鲁棒性有额外好处？
3. 可以研究 BPE vocab 的「鲁棒性结构」：哪些 token 是「鲁棒 token」？

**综述里的引用方式**：必须在「VLA 演化」章节叙述 FAST 是 2025 年的新标准，论证「OpenVLA 的离散化限制不是路线本身的限制」。在「精细操作鲁棒性」章节里 FAST 是关键案例。

### 待解问题

1. FAST 的 DCT 低频偏置对「需要快速反应的任务」（如 catch 移动物体）是否有副作用？（高频被 truncate）
2. FAST + ECoT + diffusion 三者的最佳组合是什么？是否有论文已经做了这个 ablation？

## 🔗 关联笔记

- **直接 base**：OpenVLA（24.06）——FAST 的主要 host model。
- **方法学前身**：Gato（22.05）、RT-1/2 离散化——FAST 改进的对象。
- **平行方案**：Octo（24.05）的 diffusion、π0（24.10）的 flow matching——continuous action 路线对比。
- **集成应用**：π0-FAST、π0.5（25.04）——后续工作直接使用 FAST。
- **可结合改进**：ECoT（24.07）——FAST + CoT 是潜在曦源研究方向。
- **后续轻量化**：NORA（25.04）——3B 模型也用了类似 token 优化思路。
