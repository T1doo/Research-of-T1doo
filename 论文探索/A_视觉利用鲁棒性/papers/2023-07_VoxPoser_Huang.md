---
title: "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models"
authors: "Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei"
year: "2023"
journal: "CoRL 2023"
doi: "10.48550/arXiv.2307.05973"
arxiv: "2307.05973"
venue: "CoRL 2023 (Oral)"
citekey: "huangVoxPoser2023"
itemType: "conferencePaper"
status: "已精读"
tier: "⭐⭐⭐ 必读"
tags: [literature, T1D, 主线A, 视觉grounding, value map, LLM, A4, 经典]
---

# VoxPoser — Composable 3D Value Maps 精读笔记

> [!info] 元信息
> - **作者**：Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei
> - **机构**：Stanford
> - **日期**：2023-07-12 (arXiv)
> - **arXiv**：[2307.05973](https://arxiv.org/abs/2307.05973)
> - **会议**：CoRL 2023 Oral
> - **定位**：A4 grounding 范式开山 — 把 LLM 当 "value-map composer"，绕开端到端 policy 学习

## 📄 TL;DR

VoxPoser 把 6-DoF 机器人操作任务拆为 **3D 体素 value map 的合成问题**。LLM 把自然语言指令分解为子任务，调用一组 *视觉-语言原语*（"在桌子上"、"远离杯子" 等）生成 **affordance map** 和 **constraint map**；这些体素图被 model-based 规划器组合成轨迹，实时执行。整套系统 **不需要任务专属训练数据**。在仿真和真机上跑通了几十种日常操作任务，并能在线学习接触动力学模型。本质是「**LLM 当作 spatial program composer，VLM 负责 grounding，规划器执行**」的零样本范式。

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「Value map 是更连续的 grounding 表示」**。比 ReKep 的关键点更稠密，比 mask 更含 task semantic（map 值表示"应该去哪 / 应该避开哪"）。这种表示对 LLM 友好（LLM 容易组合 affordance），对规划器友好（直接当代价函数）。
2. **「LLM 编程能力 + 3D 视觉 grounding」的组合是 *Foundation Model for Robotics* 第一波的核心方法**。VoxPoser 证明：**不需要 demonstration**，纯靠 prompt + 视觉特征也能完成多样任务；这与 VLA 端到端学习形成鲜明对照。
3. **在线动力学学习**：作者特别强调"contact-rich interaction"靠在线学一个局部动力学模型来调整 — 这是 VoxPoser 对纯 LLM 方案的一大补充，说明 *单纯靠 LLM + 视觉无法解决物理交互*。

### 方法论

- **任务分解**：LLM (GPT-4) 接收 "open the drawer" → 拆为 ["定位 handle", "施力方向", "约束保持手平直"] 等子约束
- **Value map 合成**：每个子约束由 VLM (OWL-ViT) 在 RGB-D 上 grounding → 投影到 3D 体素 → 累加得到总 value map
- **轨迹生成**：在 value map 上做 model-based MPC，输出 6-DoF 轨迹
- **闭环**：每步重新评估 value map → 对扰动 / 新物体出现敏感、可自适应

### 与曦源关联（FocusVLA / RobustVLA）

- **与 FocusVLA 方法学差异**：FocusVLA 把"视觉利用"放在 attention 内化；VoxPoser 把"视觉利用"放在 3D value map 外显化。两条路本质都是 *让 task-relevant 信号被显式编码*。
- **与 RobustVLA 组合**：VoxPoser 的 value map 可作为 RobustVLA 训练时的 *扰动不变性约束* — 同一物理场景下不同扰动应得到相同 value map，可设计 consistency loss。
- **暴露的子问题**：VoxPoser 假设 *相机标定 + 深度可得*；现实中深度噪声 / 标定漂移会破坏 value map 几何精度。这是 FocusVLA / RobustVLA 都没碰的硬件鲁棒性维度。
- **本科生子模块可行性**：⭐⭐⭐ — 完整复现需要 GPT-4 API + RGB-D 设置；但 *用 VoxPoser 的 value map 作为 attention supervision* 是可行的子模块（拿开源 VoxPoser code 提取 value map，监督 FocusVLA 的 attention pattern）。

### 待解问题

1. **VLM grounding 误差累积**：每个原语都做一次 grounding，误差累加进 value map。论文没分析这个误差链。
2. **多模态歧义**：value map 是单峰还是多峰？多峰时规划器如何选 mode？
3. **GPT-4 推理时延**：每步秒级，难做高频闭环。需要蒸馏到小模型。

%% end my-thoughts %%

## 🔗 关联笔记

- **同作者后续**：[[2024-09_ReKep_Huang]]（keypoint 路线，VoxPoser 的更紧凑版）
- **同代 grounding 工具**：[[2024-01_GroundedSAM_Ren]]
- **互补的端到端路径**：[[2026-03_FocusVLA_Zhang]]、[[2024_OpenVLA]]
- **诊断相关**：[[2025-08_ShortcutLearning_Xing]]（shortcut 学习的对照参考）
- **主线 A 总结**：[[A4_visual_grounding_总结]]

## 📌 Action Items

- [ ] 复现：用 VoxPoser 开源 code 在 LIBERO 上跑 5 个任务，记录 value map
- [ ] 实验：把 value map 作为 FocusVLA attention prior（KL 散度 loss）
- [ ] 分析：比较 VoxPoser zero-shot 和 FocusVLA full-shot 在 OOD 物体上的成功率

%% Import Date: 2026-05-26 %%
