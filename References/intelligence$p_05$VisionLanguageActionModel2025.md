---
title: "$π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization"
authors: "Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, Ury Zhilinsky"
year: "2025"
journal: ""
doi: "10.48550/arXiv.2504.16054"
citekey: "intelligence$p_05$VisionLanguageActionModel2025"
zotero-link: "zotero://select/library/items/LK47EX27"
itemType: "preprint"
tags: [literature, T1D]
---

# $π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization

> [!info] 元信息
> - **作者**：Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, Ury Zhilinsky
> - **日期**：2025-04-22
> - **来源**：
> - **DOI**：[10.48550/arXiv.2504.16054](https://doi.org/10.48550/arXiv.2504.16054)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/LK47EX27)
> - **Citekey**：`@intelligence$p_05$VisionLanguageActionModel2025`

## 📄 Abstract

In order for robots to be useful, they must perform practically relevant tasks in the real world, outside of the lab. While vision-language-action (VLA) models have demonstrated impressive results for end-to-end robot control, it remains an open question how far such models can generalize in the wild. We describe $π_{0.5}$, a new model based on $π_{0}$ that uses co-training on heterogeneous tasks to enable broad generalization. $π_{0.5}$\ uses data from multiple robots, high-level semantic prediction, web data, and other sources to enable broadly generalizable real-world robotic manipulation. Our system uses a combination of co-training and hybrid multi-modal examples that combine image observations, language commands, object detections, semantic subtask prediction, and low-level actions. Our experiments show that this kind of knowledge transfer is essential for effective generalization, and we demonstrate for the first time that an end-to-end learning-enabled robotic system can perform long-horizon and dexterous manipulation skills, such as cleaning a kitchen or bedroom, in entirely new homes.

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点
- π0.5 把开放世界泛化做成了首个端到端学习能在 3 个**完全未见过的真实家庭**完成 10–15 分钟级清理任务的工作。
- 关键认识：97.6% 的训练样本可以不来自目标平台（移动操作），异构 co-training 是泛化的核心引擎。
- Hierarchical 推理（先吐 semantic subtask 文本再吐 action）显式注入语义中间表示，是 long-horizon 任务的稳定器。
- 两阶段训练（FAST 离散 token 预训练 + flow matching action expert post-training）兼顾效率与流畅度。
- 训练地点数 scaling（3/12/22/53/82/104）显示泛化与场景多样性近似单调正相关。

### 方法论
- 架构：在 π0 基础上加 hierarchical 输出，先预测 object detection box → semantic subtask 文本 → 低层 action。
- Co-training 数据 6 类：MM（移动操作 ~400h / ~100 家）、ME（多环境固定臂）、CE（跨实体含 OXE）、HL（人工子任务标注）、WD（图像描述/VQA/定位）、VI（语言指令 post-training）。
- 预训练 280k 步 + post-training 80k 步。
- 推理：autoregressive 文本解码 → flow matching 10 步去噪，50 Hz 控制 target pose + base velocity。
- 评估：3 个真实家庭，long-horizon 任务（清厨房 / 卧室），并做 in-distribution vs OOD 物体的语言遵从对比。
- 实验显示 π0.5 在 mock home test 上显著超过 π0（即使 π0 训练拉长到 300k 步）。

### 与我研究的关联（T1D）
- **次要 baseline**：与 π0 并列在立项书的 baseline 区，是 VLA 最前沿代表。
- **方向二的进阶载体**：π0.5 的 hierarchical subtask 输出可直接复用作 RC-NF 的任务嵌入 τ，省掉额外 task encoder，减少 RC-NF 的工程复杂度。
- **方向一启发**：subtask 文本先于 action 输出，意味着可以根据 subtask 难度动态分配 action expert 算力。
- **硬件契合度**：与 π0 同量级（3B+），同样适配 46 GB 显存约束；LoRA 微调可压更低。
- 长程任务能力让 RC-NF 触发的 task-level replanning 有了真实落地价值（subtask 边界 = 高异常风险点）。

### 待解问题 / 后续追读
- 论文未明确披露总参数量，需追 openpi 仓库实际 checkpoint 体积。
- 清理厨房整体成功率（10–15 min 任务的端到端完成率）没给绝对数值，只展示了对比相对性。
- 各类数据源（MM/ME/CE/HL/WD/VI）的精确比例只给了 MM 的 2.4%，其它需追正文表。
- web data 的引入对动作生成的具体贡献是多少（消融未细化）。
- π0.5 在 LIBERO 等公开 benchmark 上的数据缺失，需要等社区评测。
- FAST 离散 token 与 flow matching 两段如何衔接的工程细节，需读 FAST 原文。
- 是否会公开 openpi 版本 checkpoint 与 web data 训练 recipe。

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:19:05.184+08:00 %%
