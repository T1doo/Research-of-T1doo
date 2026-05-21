---
title: "RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation"
authors: "Shijie Zhou, Bin Zhu, Jiarui Yang, Xiangyu Zhao, Jingjing Chen, Yu-Gang Jiang"
year: "2026"
journal: ""
doi: "10.48550/arXiv.2603.11106"
citekey: "zhouRCNFRobotConditionedNormalizing2026"
zotero-link: "zotero://select/library/items/67548VZ5"
itemType: "preprint"
tags: [literature, T1D]
---

# RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation

> [!info] 元信息
> - **作者**：Shijie Zhou, Bin Zhu, Jiarui Yang, Xiangyu Zhao, Jingjing Chen, Yu-Gang Jiang
> - **日期**：2026-03-11
> - **来源**：
> - **DOI**：[10.48550/arXiv.2603.11106](https://doi.org/10.48550/arXiv.2603.11106)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/67548VZ5)
> - **Citekey**：`@zhouRCNFRobotConditionedNormalizing2026`

## 📄 Abstract

Recent advances in Vision-Language-Action (VLA) models have enabled robots to execute increasingly complex tasks. However, VLA models trained through imitation learning struggle to operate reliably in dynamic environments and often fail under Out-of-Distribution (OOD) conditions. To address this issue, we propose Robot-Conditioned Normalizing Flow (RC-NF), a real-time monitoring model for robotic anomaly detection and intervention that ensures the robot's state and the object's motion trajectory align with the task. RC-NF decouples the processing of task-aware robot and object states within the normalizing flow. It requires only positive samples for unsupervised training and calculates accurate robotic anomaly scores during inference through the probability density function. We further present LIBERO-Anomaly-10, a benchmark comprising three categories of robotic anomalies for simulation evaluation. RC-NF achieves state-of-the-art performance across all anomaly types compared to previous methods in monitoring robotic tasks. Real-world experiments demonstrate that RC-NF operates as a plug-and-play module for VLA models (e.g., pi0), providing a real-time OOD signal that enables state-level rollback or task-level replanning when necessary, with a response latency under 100 ms. These results demonstrate that RC-NF noticeably enhances the robustness and adaptability of VLA-based robotic systems in dynamic environments.

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点
- VLA 模仿学习的 OOD 失败是结构性问题，与其改主干不如外挂一个独立的概率密度模型来兜底，模块解耦让 VLA 主干可以保持冻结。
- 把 robot state（条件 s）与 object state（被建模量 x）拆开处理，是性能关键 —— 消融显示去掉机器人状态 AUC 掉 21.6 个百分点，去掉动态形状分支掉 24.7 个百分点。
- 仅用正样本无监督训练，规避了"如何采集罕见异常样本"这一在真实机器人上几乎不可解的难题。
- 响应延迟 < 100 ms（SAM2 50 ms + 流推理 30 ms）使得它在 20–50 Hz 控制频率下实时可用，是 plug-and-play 的硬性前提。
- LIBERO-Anomaly-10 同时是方法验证和 benchmark 贡献，为后续异常检测工作提供标准测试床。

### 方法论
- 关键映射 `z = f_c(x)`，`c = (s, τ)`，对数似然 `log p_{X|C}(x|c) = log p_{Z|C}(z|c) + Σ log|det ∂f_{i,c}|`，先验 `log p_{Z|C}(z|c) = Const − ½‖z − μ_task‖²`。
- 训练：只用正样本最大似然，BalancedHardSampler 平衡，100 epochs。
- 推理：算异常分数 → 阈值 `Upper_T = μ_T + Q_{1−α}(D_T)`，α = 0.05。
- 触发：state-level rollback（homing 等异常分回落）/ task-level replanning（通知 LLM 或人）。
- 关键数值：平均 AUC 0.9309（vs 基线最高 0.8500）、AP 0.9494（vs 0.8507）、端到端延迟 < 100 ms。
- 基线覆盖 GPT-5 / Gemini 2.5 Pro / Claude 4.5（多模态 LLM 直接判定）+ FailDetect（flow matching 失败检测器）。

### 与我研究的关联（T1D）
- **方向二核心论文**：直接对应曦源项目「基于轻量级模块的异常检测」方向，是要复现并扩展的对象。
- **硬件契合度极高**：训练显存远低于 46 GB 上限，可在单卡 24 GB 跑通；推理只新增 30 ms 流计算 + 50 ms SAM2，对 80 GB A100 部署 π0 完全无压力。
- **可复用模块**：(1) 条件归一化流框架；(2) LIBERO-Anomaly-10 benchmark 构造方法学（可迁到 RoboTwin / CALVIN）；(3) state/task 两级触发策略；(4) 与 π0 的 side-car 集成接口。
- **与方向一（动态推理）和方向三（鲁棒微调）有联动空间**：高异常分数轨迹可作为主动学习的难样本输入到方向三的鲁棒微调中。

### 待解问题 / 后续追读
- LIBERO-Anomaly-10 三类异常之外，真实世界长尾异常（机械臂卡顿、感知遮挡、电机过热）能否泛化检测？论文未答。
- SAM2 是 50 ms 的"重"分割大模型，把它换成 MobileSAM / EfficientSAM 之后 AUC 掉多少？需要自己测。
- α = 0.05 阈值是 per-task 标定还是全局共享？跨任务迁移成本如何？
- GitHub 仓库尚未在论文中公开，需追：项目主页 `https://heikaishuizz.github.io/RC-NF/` 的代码与 LIBERO-Anomaly-10 数据释放。
- 追读 FailDetect、π0 真实部署细节、SAM2 在机器人场景的最佳替代。
- 双臂（RoboTwin）/ 移动操作（π0.5）场景下 robot state 维度更高，flow 是否需要更深，是开放工程问题。

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:07:44.829+08:00 %%
