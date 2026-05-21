---
title: "Action-to-Action Flow Matching"
authors: "Jindou Jia, Gen Li, Xiangyu Chen, Tuo An, Yuxuan Hu, Jingliang Li, Xinying Guo, Jianfei Yang"
year: "2026"
journal: ""
doi: "10.48550/arXiv.2602.07322"
citekey: "jiaActiontoActionFlowMatching2026"
zotero-link: "zotero://select/library/items/4RJVSXNK"
itemType: "preprint"
tags: [literature, T1D]
---

# Action-to-Action Flow Matching

> [!info] 元信息
> - **作者**：Jindou Jia, Gen Li, Xiangyu Chen, Tuo An, Yuxuan Hu, Jingliang Li, Xinying Guo, Jianfei Yang
> - **日期**：2026-05-07
> - **来源**：
> - **DOI**：[10.48550/arXiv.2602.07322](https://doi.org/10.48550/arXiv.2602.07322)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/4RJVSXNK)
> - **Citekey**：`@jiaActiontoActionFlowMatching2026`

## 📄 Abstract

Diffusion-based policies have recently achieved remarkable success in robotics by formulating action prediction as a conditional denoising process. However, the standard practice of sampling from random Gaussian noise often requires multiple iterative steps to produce clean actions, leading to high inference latency that incurs a major bottleneck for real-time control. In this paper, we challenge the necessity of uninformed noise sampling and propose Action-to-Action flow matching (A2A), a novel policy paradigm that shifts from random sampling to initialization informed by the previous proprioceptive action. Unlike existing methods that treat proprioceptive action feedback as static conditions, A2A leverages historical proprioceptive sequences, embedding them into a high-dimensional latent space as the starting point for action generation. This design bypasses costly iterative denoising while effectively capturing the robot's physical dynamics and temporal continuity. Extensive experiments demonstrate that A2A exhibits high training efficiency, fast inference speed, and improved generalization. Notably, A2A enables high-quality action generation in as few as a single inference step, and exhibits superior robustness to visual perturbations and enhanced generalization to unseen configurations. Lastly, we also extend A2A to video generation, demonstrating its broader versatility in temporal modeling. Project site: https://lorenzo-0-0.github.io/A2A_Flow_Matching.

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点
- **挑战「无信息噪声采样」的必要性**：标准 diffusion / flow matching 都假设从 $\mathcal{N}(0, I)$ 出发，A2A 用历史本体感觉动作 CNN 嵌入做起点，把多步去噪压成 1 步。
- **1 步推理 0.56 ms**：相对 vanilla diffusion 快 20×、相对标准 flow matching 快 5×，真正达到毫秒级实时控制。
- **生成范式比回归范式更鲁棒**：在 Level 1 视觉扰动下，flow-latent (A2A) 训练 95% / 泛化 38%，回归 latent 在扰动下「显著失败」——说明保留生成过程对 OOD 有帮助。
- 「**历史 → 未来」初始化范式不限于动作**：论文把同一框架扩展到视频生成 (F2F)，PSNR/SSIM/LPIPS 显著超过回归基线，证明这是一类通用的时间建模思路。
- 不是单纯加速，**是「轻量 + 加速 + 鲁棒」三赢**。

### 方法论
- **核心机制**：$z_0 = E_a(a_{\le t})$，$E_a$ 是 kernel size=5 的 1D CNN，n 帧历史动作 → 512 维潜空间。
- **训练损失**：$\mathcal{L}_{FM}=\mathbb{E}_{\tau,z_0,z_1}\|f_\theta(z_\tau,\tau,c)-v_\tau(z_\tau,\tau,c)\|^2$，最优传输 $z_\tau=(1-\tau)z_0+\tau z_1$。
- **总目标**：$\mathcal{L}_{total}=\lambda_1\mathcal{L}_{FM}+\lambda_2\mathcal{L}_{AE}+\lambda_3\mathcal{L}_{IC}$，$\lambda=(1, 0.5, 1)$。
- **解码器**：残差 MLP 把 $z_1$ 还原成未来 n 帧动作序列。
- **推理**：1 步前向计算 $f_\theta$ 即可拿到可执行动作，无需迭代。
- **实验结果**：LIBERO 5 任务 30 epoch 100 演示，A2A 平均 **90.4%**，超过 VITA 88%、FM-UNet 60.4%、DDPM-UNet 51.6%；单步推理 0.56 ms。
- **消融**：flow-latent 比 reg-latent 在扰动下鲁棒得多；UNet 架构次于 latent 架构。

### 与我研究的关联（T1D）
- **方向一关键 baseline**：曦源项目「单步推理」诉求的代表工作，与 AAC 正交、与 SmolVLA 互补。
- **直接可移植到 SmolVLA**：SmolVLA action expert 当前 flow matching 用随机噪声起点，**直接换成 A2A 范式即可继承 SmolVLA 的视觉编码 + 获得 1 步推理**，这是曦源项目的「切入点 B」。
- **硬件契合**：A2A 主要新增 $E_a$ CNN encoder（参数量极小，<5 M），完全在 46 GB 显存裕度内；甚至单卡 24 GB 也可训练。
- **可复用模块**：
    - 1D CNN encoder $E_a$（kernel=5，<5 M 参数）
    - 流匹配损失（已在 SmolVLA 中实现，可直接复用）
    - 共享 512 维潜空间设计
- **鲁棒性优势**：在 LIBERO-Plus / RoboTwin 等扰动场景下，A2A 的生成范式相对回归范式有天然优势，恰好响应曦源项目对鲁棒性的隐式要求。

### 待解问题 / 后续追读
- 论文**仅在 Roboverse 5 任务**上评测，未覆盖 CALVIN、RoboTwin、LIBERO-Plus，更不曾对比 π0 / OpenVLA / CogACT / RDT / DexVLA 等主流大模型 baseline。
- 当任务模式跳变（如夹爪从静止到突然张开）时，历史动作起点反而可能成为「锚定偏置」，论文未讨论这种 mode-jumping 风险。
- 与 AAC 结合时，AAC 的 entropy 计算需要多次采样，而 A2A 的设计是确定性 1 步——如何在确定性框架下做「分布采样」是开放问题（可能需要给 $E_a$ 加噪声扰动或用 dropout）。
- 追读：VITA 论文（A2A 最强 baseline）、Rectified Flow / Consistency Models（同类加速思路）、项目站源码 https://lorenzo-0-0.github.io/A2A_Flow_Matching/。
- 待验证：A2A 在 SmolVLA 上 warm-start 后是否需要全量重训 action expert，还是只需 fine-tune $E_a$。

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:27:41.457+08:00 %%
