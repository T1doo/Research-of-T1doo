---
title: "On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations"
authors: "Jianing Guo, Zhenhong Wu, Chang Tu, Yiyao Ma, Xiangqi Kong, Zhiqian Liu, Jiaming Ji, Shuning Zhang, Yuanpei Chen, Kai Chen, Qi Dou, Yaodong Yang, Xianglong Liu, Huijie Zhao, Weifeng Lv, Simin Li"
year: "2026"
journal: ""
doi: "10.48550/arXiv.2510.00037"
citekey: "guoRobustnessVisionLanguageActionModel2026"
zotero-link: "zotero://select/library/items/R9QU5697"
itemType: "preprint"
tags: [literature, T1D]
---

# On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations

> [!info] 元信息
> - **作者**：Jianing Guo, Zhenhong Wu, Chang Tu, Yiyao Ma, Xiangqi Kong, Zhiqian Liu, Jiaming Ji, Shuning Zhang, Yuanpei Chen, Kai Chen, Qi Dou, Yaodong Yang, Xianglong Liu, Huijie Zhao, Weifeng Lv, Simin Li
> - **日期**：2026-02-24
> - **来源**：
> - **DOI**：[10.48550/arXiv.2510.00037](https://doi.org/10.48550/arXiv.2510.00037)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/R9QU5697)
> - **Citekey**：`@guoRobustnessVisionLanguageActionModel2026`

## 📄 Abstract

In Vision-Language-Actionf(VLA) models, robustness to real-world perturbations is critical for deployment. Existing methods target simple visual disturbances, overlooking the broader multi-modal perturbations that arise in actions, instructions, environments, and observations. Here, we first evaluate the robustness of mainstream VLAs under 17 perturbations across four modalities. We find (1) actions as the most fragile modality, (2) Existing visual-robust VLA do not gain robustness in other modality, and (3) pi0 demonstrates superior robustness. To build multi-modal robust VLAs, we propose RobustVLA against perturbations in VLA inputs and outputs. For output robustness, we perform offline robust optimization against worst-case action noise that maximizes mismatch in flow matching objective. This can be seen as adversarial training, label smoothing, and outlier penalization. For input robustness, we enforce consistent actions across input variations that preserve task semantics. To account for multiple perturbations, we formulate robustness as a multi-armed bandit problem and apply an upper confidence bound algorithm to automatically identify the most harmful noise. Experiments on LIBERO demonstrate our RobustVLA delivers absolute gains over baselines of 12.6% on the pi0 backbone and 10.4% on the OpenVLA backbone across all 17 perturbations, achieving 50.6x faster inference than existing visual-robust BYOVLA that requires external LLMs, and a 10.4% gain under mixed perturbations. On the real-world FR5 robot, under four types of multimodal perturbations, RobustVLA shows strong low-data performance, outperforming pi0 by 65.6% success rate with 25 demonstrations. Even with abundant demos, our method still outperform pi0 by 30% success rate. Code and demo videos available at https://github.com/gakakulicc/RobustVLA.

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点

1. **VLA 鲁棒性是多模态问题，而非单纯视觉问题**：现有工作（BYOVLA、AdaWorld）只看图像扰动，论文首次系统评测 17 种跨 4 模态扰动（action / observation / environment / instruction）。
2. **Action 是最脆弱模态**——这是反直觉发现：π0 在 0.05 幅度 action 噪声下成功率从 96% 跌到 52.4%、0.1 幅度下完全归零（0%）。意味着方向二的"加 image augmentation"路线远远不够。
3. **视觉鲁棒不迁移**：BYOVLA 在高斯噪声 +7.3%、死像素 +22.3%，但在 action / instruction / environment 模态改进 **0.0%**——这彻底否定了视觉数据增广为银弹的假设。
4. **π0 优于 OpenVLA / π0-FAST**：扩散式 action head（flow matching）相对自回归和 FAST tokenizer 分别高 27.9% 和 5.1%；这是选骨干时的重要数据。
5. **鲁棒训练几乎不损失 clean accuracy**：95.5% vs 96.0%，0.5% 代价换 14% 鲁棒性提升，性价比极高。

### 方法论

- **双正则框架**：
  - Output robust：$\min_\theta \mathcal{L}_{\pi_0} + \lambda_{out} \max_\delta \|v_\theta - u - \delta\|^2$，找最坏 action 噪声 δ 最大化 flow-matching mismatch；等价于"对抗训练 + label smoothing + outlier penalization"三件套。
  - Input robust：$\min_\theta \mathcal{L}_{\pi_0} + \lambda_{in} \max_{\omega^*} \|v_\theta(\omega^*(o)) - u\|^2$，对语义保留的输入扰动 ω* 强制 action 一致。
- **多臂老虎机 + UCB 选最坏扰动**：17 扰动 = 17 arm，reward $r_n(\omega_i) = \mathcal{L}_{perturbed} - \mathcal{L}_{clean}$，UCB 系数 α、EMA 衰减 0.9、z-score 归一化。训练动态聚焦最脆弱模态。
- **关键数值**：LIBERO 上 π0 +12.6% / OpenVLA +10.4%、混合扰动 +10.4%、FR5 真机 25 demo +65.6%；推理 50.6× 快于 BYOVLA（无需外部 LLM）。
- **代码**：`https://github.com/gakakulicc/RobustVLA`（开源，含 demo 视频）。

### 与我研究的关联（T1D）

- **Efficient VLA 立项的最佳一角**：鲁棒训练**零推理开销**、**零参数增加**——所有改动在训练损失里，与"轻量化推理"目标完美对齐。
- **数据高效是天然附带的**：FR5 真机 25 demo +65.6%，比"加数据"路线（方向二）的需求量低一个数量级；本质上 worst-case 噪声是合成的数据增广。
- **硬件契合度**：worst-case 优化的外内双循环会让训练显存比标准 π0 高 30%-50%，46 GB GPU 上能否 batch=2 微调 π0 + RobustVLA，**是立项前必须实测的第一个数字**。
- **可复用模块**：
  - 17 扰动定义可以直接搬到 SmolVLA / ACT / DP 等任何模仿学习模型。
  - 多臂老虎机扰动调度器是一个独立模块，可拆出来做消融。
  - LIBERO + LIBERO-Plus 评测协议（每扰动 10 试、4 任务聚合、clean-robust gap 报告）是行业标准。
- **与方向一/二的协同**：
  - 方向一（LoRA / 量化）+ 鲁棒训练 = "小模型 + 强鲁棒"双赢，是 Efficient VLA 的旗舰组合。
  - 方向二（数据增广）的图像扰动模块可以与 RobustVLA 的 input-robust 共享。

### 待解问题 / 后续追读

- **跨模态鲁棒性是否会牺牲单模态极限**？比如同时训 17 扰动 vs 只训 visual 子集，单一视觉扰动下哪个更鲁棒？
- **多臂老虎机的 UCB 在 17 arm 下的统计效率**：n=10000 step 内 UCB 是否能稳定区分所有 arm？需对比 ε-greedy / Thompson sampling。
- **真机实验只做了 4 个 pick-place 任务**：长程操作（如 CALVIN 4-step chain）下鲁棒训练是否仍 work？
- **追读**：
  1. BYOVLA（被 50.6× 击败的对手，理解视觉鲁棒 VLA 的天花板）
  2. AdaWorld（另一个视觉鲁棒 VLA 工作）
  3. π0 原始论文（理解 flow-matching action head 为什么对扰动鲁棒）
  4. Madry et al. 2018 对抗训练（理解 worst-case δ 优化的收敛性理论）
  5. LIBERO-Plus 论文（核心评测平台）

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:07:44.817+08:00 %%
