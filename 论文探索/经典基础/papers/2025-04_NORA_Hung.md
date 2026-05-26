---
title: "NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks"
authors: "Chia-Yu Hung, Qi Sun, Pengfei Hong, et al. (SUTD + DAMO)"
year: "2025"
venue: "preprint 2025"
arxiv: "2504.19854"
status: "已精读"
tier: "⭐⭐⭐"
tags: [literature, T1D, 经典基础, VLA, lightweight, edge, open-source]
---

# NORA

> [!info] 元信息
> SUTD + 阿里 DAMO 2025.04。**3B 开源轻量 VLA**，基于 Qwen2.5-VL-3B + FAST tokenizer，目标是「OpenVLA 性能的 3B 替代品」。在 OXE 970k episodes 上预训练。LIBERO 平均成功率 0.798（接近 OpenVLA-7B 的 0.797），但模型小一半以上，推理快 ~2×，**边缘部署友好**。是 VLA 轻量化趋势的代表作。

## 📄 TL;DR（100-150 字）

NORA 针对 OpenVLA-7B 「太重，难以边缘部署」的痛点，用 Qwen2.5-VL-3B（更新的强 VLM backbone）+ FAST tokenizer（更高效的动作编码）做 OXE 大规模训练。模型只有 3B 参数（OpenVLA 的 43%），但 LIBERO benchmark 平均 0.798，几乎与 OpenVLA-7B 的 0.797 持平。SimplerEnv（Google Robot / WidowX）也接近 OpenVLA。证明 **2025 年的 VLM backbone + 现代 tokenization 让 3B 模型达到 2024 年 7B 水平**，是 VLA scaling efficiency 的重要 datapoint，也是显存受限场景的实用选择。

## 🧠 我的思考

### 在 VLA 谱系中的位置

NORA 在 VLA 谱系里代表**「2025 轻量化趋势」**——VLA 不再追求一味做大（RT-2 55B → OpenVLA 7B），而是探索 effective scaling 和边缘可部署性。

**VLA 规模演化时间线**：
- RT-1（22.12）：35M；
- Gato（22.05）：1.2B（多任务）；
- PaLM-E（23.03）：12B / 562B；
- RT-2（23.07）：5B / 12B / 55B；
- Octo（24.05）：27M / 93M；
- OpenVLA（24.06）：**7B（成为标准）**；
- π0（24.10）：~3B（PaliGemma-3B + action expert）；
- RDT-1B（24.10）：1.2B；
- TinyVLA、SmolVLA（24-25）：<1B 边缘探索；
- **NORA（25.04）：3B 但性能持平 OpenVLA-7B**；
- π0.5（25.04）：~3B 改进版；
- Gemini Robotics（25.03）：闭源 frontier，参数未知。

引用关系：NORA 引用 OpenVLA、Octo、π0、FAST、Qwen2.5-VL；被 2025+ 同期/后续 lightweight VLA 论文引用对比。它**不是范式创新**，而是工程精细化的代表，但对实际部署和边缘 robot 研究意义重大。

### 核心方法

**架构**：
1. **VLM backbone**：Qwen2.5-VL-3B-Instruct（2025.01 阿里发布的强多模态模型，比 Llama-2-7B 更小但性能更强）；
   - 视觉：自研 ViT（动态分辨率）；
   - 语言：Qwen2.5-3B（数学 / 代码 / 推理强）；
2. **动作 tokenization**：FAST tokenizer（DCT + BPE），token 数比 OpenVLA 少 5-10×；
3. **训练**：OXE 970k episodes（与 OpenVLA 同数据），但用 FAST tokenizer 使训练效率高很多。

**训练细节**：
- ~3M training steps；
- Bf16 + Zero-2，8-16 A100；
- LoRA-free full fine-tune（3B 可控）。

**关键实验**：
- **LIBERO benchmark**：NORA 0.798 vs OpenVLA 0.797（持平）；
- **SimplerEnv (Google Robot)**：略低 OpenVLA（NORA 0.62 vs OVLA 0.70 visual matching）；
- **SimplerEnv (WidowX)**：相当；
- **推理速度**：~2× 快于 OpenVLA（7B → 3B + 短 token）；
- **显存**：bf16 推理 ~6GB（OpenVLA 14GB），可部署 RTX 4080 / 边缘 GPU。

### 鲁棒性视角下的解读

NORA 的鲁棒性 profile 是「OpenVLA 范式的精简版」，重点关注 trade-off：

**NORA 相对 OpenVLA 的鲁棒性**：

**优势**：
1. **更新的 VLM backbone**：Qwen2.5-VL 比 Llama-2 在视觉推理、多语言、数学常识上更强，可能带来更好的语义 / 数值理解鲁棒性；
2. **FAST tokenizer 集成**：精细操作鲁棒性继承 FAST 的提升；
3. **推理快**：闭环响应更快，对动态扰动反应更及时；
4. **部署可访问**：低显存 → 更多研究者可独立做鲁棒性测试，可能加速社区改进。

**劣势 / 不确定**：
1. **小模型容量**：3B vs 7B，长尾任务、复杂语义指令可能性能下降；
2. **Qwen2.5-VL 训练数据偏中文**：对英文指令、英文物体名称可能略弱（实验有待验证）；
3. **OXE 训练时长 / hyperparameter 不一定最优**：作为新论文，训练配方可能未完全优化；
4. **真机部署验证少**：论文主要是 sim benchmark，真机鲁棒性数据少；
5. **没有 reasoning 增强**：相对 ECoT 没有特殊处理，纯依赖 VLM 能力。

**与 π0 的对比**：
- 同规模（~3B）；
- NORA 是离散 + FAST，π0 是 flow matching；
- 推理速度 NORA 更快（单 token vs flow steps），但 π0 在连续控制更平滑；
- 选哪个取决于任务特性。

### 与曦源（鲁棒性主线）的关联

NORA 对曦源是**强候选 baseline 之一**，特别适合显存受限场景：

**作为 baseline**：
1. **3B 在 46GB 显存上极其轻松**：
   - bf16 推理：~6GB；
   - 全参数 fine-tune：~30GB（with Zero-2），有充足余量；
   - LoRA fine-tune：<10GB；
   - 可以同时跑多个实验；
2. **开源完整**：模型权重、训练代码、评测脚本在 HuggingFace + GitHub；
3. **与 OpenVLA 公平对比**：同数据、同 benchmark，鲁棒性结果可直接对照；
4. **2025 最新 VLM backbone**：用 Qwen2.5-VL 而非 2023 的 Llama-2，体现 SOTA 水平。

**作为方向启示**：
1. 论证「轻量 VLA 不一定鲁棒性差」是有价值的研究问题；
2. 可以研究 NORA vs OpenVLA 的「鲁棒性退化曲线」——3B 在什么扰动下开始显著差于 7B？
3. 可以研究 Qwen2.5-VL 的中文 / 多语言指令鲁棒性。

**推荐用法**：
- **主 baseline 仍用 OpenVLA**（社区共识）；
- **NORA 作为 "modern lightweight" 对照**，特别是显存压力大或想做大量 ablation 时；
- 综述里把 NORA 作为「VLA 轻量化趋势」的标志性案例。

### 待解问题

1. NORA 的中文指令鲁棒性是否显著高于英文 VLM？（Qwen2.5-VL 的中文偏置）
2. NORA 在真机部署的实际鲁棒性 vs sim benchmark 是否有显著 gap？（很多 sim-strong 模型真机弱）
3. NORA + ECoT-style reasoning 是否能进一步提升？（小模型 + 推理增强是否仍 work？）

## 🔗 关联笔记

- **直接 base**：OpenVLA（24.06）——性能对照；FAST tokenizer（25.01）——动作编码方案。
- **VLM backbone**：Qwen2.5-VL（25.01）——最新 VLM 基础。
- **同规模平行**：π0（24.10）——~3B flow matching 路线；RDT-1B（24.10）——1.2B diffusion。
- **更轻量化方向**：TinyVLA、SmolVLA——<1B 探索。
- **数据基础**：Open X-Embodiment（23.10）。
- **下一代**：π0.5（25.04）——同期 3B 改进版。
