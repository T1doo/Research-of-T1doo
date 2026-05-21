---
title: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"
authors: "Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, Remi Cadene"
year: "2025"
journal: ""
doi: "10.48550/arXiv.2506.01844"
citekey: "shukorSmolVLAVisionLanguageActionModel2025"
zotero-link: "zotero://select/library/items/3Z5IDWKM"
itemType: "preprint"
tags: [literature, T1D]
---

# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

> [!info] 元信息
> - **作者**：Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, Remi Cadene
> - **日期**：2025-06-02
> - **来源**：
> - **DOI**：[10.48550/arXiv.2506.01844](https://doi.org/10.48550/arXiv.2506.01844)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/3Z5IDWKM)
> - **Citekey**：`@shukorSmolVLAVisionLanguageActionModel2025`

## 📄 Abstract

Vision-language models (VLMs) pretrained on large-scale multimodal datasets encode rich visual and linguistic knowledge, making them a strong foundation for robotics. Rather than training robotic policies from scratch, recent approaches adapt VLMs into vision-language-action (VLA) models that enable natural language-driven perception and control. However, existing VLAs are typically massive--often with billions of parameters--leading to high training costs and limited real-world deployability. Moreover, they rely on academic and industrial datasets, overlooking the growing availability of community-collected data from affordable robotic platforms. In this work, we present SmolVLA, a small, efficient, and community-driven VLA that drastically reduces both training and inference costs, while retaining competitive performance. SmolVLA is designed to be trained on a single GPU and deployed on consumer-grade GPUs or even CPUs. To further improve responsiveness, we introduce an asynchronous inference stack decoupling perception and action prediction from action execution, allowing higher control rates with chunked action generation. Despite its compact size, SmolVLA achieves performance comparable to VLAs that are 10x larger. We evaluate SmolVLA on a range of both simulated as well as real-world robotic benchmarks and release all code, pretrained models, and training data.

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点
- **0.45 B 参数模型可以打平 3.3 B 的 π0**：LIBERO 平均 87.3% vs π0 86.0%、Meta-World 57.3% vs π0 47.9%、真实 SO100 三任务 78.3% vs π0 61.7%，证明 VLA 不必走大模型路线。
- **异步推理是真正的工程突破**：解耦「感知-预测」与「执行」，Pick-Place 任务从同步 13.75 s 降至 9.70 s（-30%），60 s 内完成次数 9→19，控制频率从 ~7 Hz 提升到 ~14+ Hz。
- 「**用一半 VLM 层 + 一半视觉 token + 一半 Action Expert 宽度**」的极简组合可以保持性能；VLM 后 16 层贡献边际很小。
- **CA+SA 交错**胜过单 CA（85.5 vs 79.0）和单 SA（85.5 vs 74.5），是 action expert 设计的关键决策。
- **社区数据可以替代学术数据**：481 个 LeRobot 数据集、10.6 M 帧，比 OpenVLA 100 万轨迹少一个数量级，但成果反而更好。

### 方法论
- **架构**：SmolVLM-2（SigLIP + SmolLM2）只用前 16/32 层；视觉 token 64/帧（全局图像 + 像素随机化）；Action Expert 隐层 0.75×d，输出 n=50 动作块；CA 与 SA 层交错，因果掩码。
- **训练目标**：条件流匹配（Flow Matching，胜回归 5 pp）。
- **训练设置**：4×GPU ~30k GPU-hr，bfloat16 + torch.compile，HuggingFace accelerate 多卡。
- **异步推理核心算法**：维护动作队列 $A_t$，当 $|A_t|/n < g$（队列阈值）触发新一轮预测；观测相似度过滤避免冗余 forward。
- **数据规模**：22.9K 轨迹、10.6 M 帧，Qwen2.5-VL-3B 生成任务标注，OBS_IMAGE_1/2/3 相机统一。
- **关键消融**：CA+SA 交错 (85.5%)、因果掩码 (74.5% vs 双向 67.5%)、N=16 层 VLM (78.5%)、Flow Matching 训练 (80.25% vs 回归 75.25%)、n=10 动作块 (84.0%)。

### 与我研究的关联（T1D）
- **方向一最重要的 baseline / 基座**：0.45 B 完全在 46 GB 显存约束内可全参微调（甚至单卡 24 GB 也可 LoRA），是曦源项目的天然起点。
- **异步推理框架可作为「容器」**：AAC 的 chunk 截断信号可直接接入 async 队列作为第二触发条件；A2A 的 1-step 起点可替换 action expert 的 flow matching 噪声起点。
- **可复用模块**：
    - LeRobot 框架（HuggingFace `huggingface/lerobot`）：包含训练 / 评测 / 部署完整链路
    - 异步推理调度器：算法 1 的 Python 实现可直接复用
    - 预训练权重：lerobot/smolvla_base、svla_so100_*、metaworld_mt50 全部开源
- **数据成本最低**：复用 LeRobot 社区数据，不需要采集新轨迹即可起步。

### 待解问题 / 后续追读
- 异步推理的「观测相似度阈值」如何设置论文未详尽消融，可能存在场景敏感性。
- 在视觉扰动（LIBERO-Plus）下，0.45 B 模型是否仍能保持 87% 级性能？小模型可能比大模型更脆弱。
- 与 A2A 结合后，Flow Matching 起点从随机噪声改为历史动作 CNN 嵌入，是否需要重新调 CA/SA 比例？
- 追读：SmolVLM-2 原文、LeRobot 框架 README、π0 (RSS 2025) 以对比同尺度细节。
- 待验证：在 RoboTwin 双臂任务上的迁移效果；async 推理在长程任务（CALVIN）上的增益是否显著。

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:20:11.994+08:00 %%
