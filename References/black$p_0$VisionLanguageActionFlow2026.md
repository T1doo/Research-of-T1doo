---
title: "$π_0$: A Vision-Language-Action Flow Model for General Robot Control"
authors: "Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky"
year: "2026"
journal: ""
doi: "10.48550/arXiv.2410.24164"
citekey: "black$p_0$VisionLanguageActionFlow2026"
zotero-link: "zotero://select/library/items/NBHWU2QB"
itemType: "preprint"
tags: [literature, T1D]
---

# $π_0$: A Vision-Language-Action Flow Model for General Robot Control

> [!info] 元信息
> - **作者**：Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky
> - **日期**：2026-01-08
> - **来源**：
> - **DOI**：[10.48550/arXiv.2410.24164](https://doi.org/10.48550/arXiv.2410.24164)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/NBHWU2QB)
> - **Citekey**：`@black$p_0$VisionLanguageActionFlow2026`

## 📄 Abstract

Robot learning holds tremendous promise to unlock the full potential of flexible, general, and dexterous robot systems, as well as to address some of the deepest questions in artificial intelligence. However, bringing robot learning to the level of generality required for effective real-world systems faces major obstacles in terms of data, generalization, and robustness. In this paper, we discuss how generalist robot policies (i.e., robot foundation models) can address these challenges, and how we can design effective generalist robot policies for complex and highly dexterous tasks. We propose a novel flow matching architecture built on top of a pre-trained vision-language model (VLM) to inherit Internet-scale semantic knowledge. We then discuss how this model can be trained on a large and diverse dataset from multiple dexterous robot platforms, including single-arm robots, dual-arm robots, and mobile manipulators. We evaluate our model in terms of its ability to perform tasks in zero shot after pre-training, follow language instructions from people and from a high-level VLM policy, and its ability to acquire new skills via fine-tuning. Our results cover a wide variety of tasks, such as laundry folding, table cleaning, and assembling boxes.

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点
- 把 flow matching 这种连续生成范式嫁接到 VLM 上，得到 3.3B 参数的 generalist 机器人基础模型，是当前 VLA "大模型路线"的标杆。
- 数据是核心瓶颈，作者构建了 ~10,000 小时、903M timesteps、68 任务、7 机器人配置的多源数据池，强调 pre-train + post-train 两阶段范式（前者教恢复，后者教流畅）。
- Action chunk H=50 + 10 步 Euler 推理 → 73 ms 总延迟，是 VLA 实时控制可行性的一份重要数据点。
- π0-small（470M，无 VLM 初始化）大幅落后 → VLM 预训练知识对操作能力是有效迁移的。
- 在 shirt folding、bussing 等长程灵巧任务上显著超过 OpenVLA、Octo 等同代方法。

### 方法论
- 架构：PaliGemma 3B VLM + 300M action expert（MoE 分支），blockwise causal mask 划分 images/language / state / actions 三段。
- Flow matching：噪声化 `A_t^τ = τA_t + (1-τ)ε`，损失 `L = E‖v_θ(A_t^τ, o_t) − u(A_t^τ|A_t)‖²`，Beta 分布偏低 τ 采样、cutoff s=0.999。
- 推理 forward Euler 10 步，δ = 0.1，输出 H=50 动作 chunk，20–50 Hz 实时控制。
- 数据：内部 903M timesteps（单臂 106M / 双臂 797M）+ OXE Magic Soup + Bridge v2 + DROID（9.1%），按 n^0.43 缩放权重。
- 推理 ~73 ms（image encoder 14 ms + observation 32 ms + 10 步 action 27 ms，RTX 4090）。
- 性能（vs OpenVLA / Octo / π0-small）：shirt folding ~0.95 / 0.30 / 0.40 / 0.70；bussing easy ~0.80 / 0.20 / 0.30 / 0.50。

### 与我研究的关联（T1D）
- **首要 baseline**：曦源项目 Efficient VLA 立项书明确把 π0 列为经典 VLA 代表，3.3B 参数 + 46 GB 显存（bs=2 全参微调）是项目的硬件参考点。
- **方向二的载体**：RC-NF 在原论文真实实验中直接挂在 π0 上做异常检测验证，本项目复现方向二必须先把 π0（或 openpi 实现）跑通。
- **方向一切入点**：73 ms 推理由 14+32+27 ms 三段组成，存在动态算力分配空间（简单任务可减步数 / 跳过 image encoder）。
- **方向三参考**：pre-train + post-train 两阶段提供鲁棒性微调的数据策略蓝本。
- LIBERO 上的具体数字论文未给（评测主打内部任务），需要从 openpi 社区复现拿。

### 待解问题 / 后续追读
- LIBERO 各子集（spatial / object / goal / long）的精确成功率，论文没给，须看 openpi GitHub 或后续社区复现。
- π0 原版未官方开源 checkpoint，需追 openpi 仓库（社区移植）的对应版本号与许可证。
- 推理 10 步 Euler 是否可降到 5 步（直接砍半延迟）？是方向一可挖的低悬果。
- action expert 300M 是否可替换为更轻的 head（影响 27 ms 这一段）？
- π0 vs OpenVLA / RT-2 / RDT / Octo 完整的 LIBERO 对比矩阵需自行整理。
- 跨实体（不同机器人）零样本能力的边界在哪里。

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:19:05.167+08:00 %%
