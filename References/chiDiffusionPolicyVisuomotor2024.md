---
title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
authors: "Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song"
year: "2024"
journal: ""
doi: "10.48550/arXiv.2303.04137"
citekey: "chiDiffusionPolicyVisuomotor2024"
zotero-link: "zotero://select/library/items/LU48I6QZ"
itemType: "preprint"
tags: [literature, T1D]
---

# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

> [!info] 元信息
> - **作者**：Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song
> - **日期**：2024-03-14
> - **来源**：
> - **DOI**：[10.48550/arXiv.2303.04137](https://doi.org/10.48550/arXiv.2303.04137)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/LU48I6QZ)
> - **Citekey**：`@chiDiffusionPolicyVisuomotor2024`

## 📄 Abstract

This paper introduces Diffusion Policy, a new way of generating robot behavior by representing a robot's visuomotor policy as a conditional denoising diffusion process. We benchmark Diffusion Policy across 12 different tasks from 4 different robot manipulation benchmarks and find that it consistently outperforms existing state-of-the-art robot learning methods with an average improvement of 46.9%. Diffusion Policy learns the gradient of the action-distribution score function and iteratively optimizes with respect to this gradient field during inference via a series of stochastic Langevin dynamics steps. We find that the diffusion formulation yields powerful advantages when used for robot policies, including gracefully handling multimodal action distributions, being suitable for high-dimensional action spaces, and exhibiting impressive training stability. To fully unlock the potential of diffusion models for visuomotor policy learning on physical robots, this paper presents a set of key technical contributions including the incorporation of receding horizon control, visual conditioning, and the time-series diffusion transformer. We hope this work will help motivate a new generation of policy learning techniques that are able to leverage the powerful generative modeling capabilities of diffusion models. Code, data, and training details is publicly available diffusion-policy.cs.columbia.edu

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点

1. **Diffusion 在 visuomotor policy 上是 SOTA 思路**：12 task / 4 benchmark 平均 **+46.9%** 相对 LSTM-GMM / IBC / BET 等 baseline；尤其在 multimodal action 任务（Block Pushing）上把离散化 baseline 摁在地上。
2. **多模态 action 是关键优势**：DDPM 的去噪过程天然不会模式坍塌，对人类示教中的多解性（同一观测下多种合法 action）建模能力远超 MSE-based BC。
3. **CNN vs Transformer 二选一**：CNN-based（1D temporal CNN + FiLM）开箱即用、几乎不调参；Transformer-based（minGPT-like）更适合高频动作但超参敏感。论文推荐 CNN 作为默认。
4. **受控的 Receding horizon**：观测 $T_o$、预测 $T_p$、执行 $T_a$ 三参数解耦；**$T_a = 8$ 对多数任务最优**——比 ACT 的 k=100 短得多，靠频繁 re-predict 维持 reactivity。
5. **DDIM 把推理从 100 步压到 10 步**：3080 GPU **0.1 s 推理延迟**，10 Hz policy 预测 + 125 Hz 线性插值执行，完全满足实时控制。

### 方法论

- **损失**：标准 DDPM noise prediction loss $\mathbb{E}_{k,\epsilon}\|\epsilon - \epsilon_\theta(a^k_t, k, o)\|^2$，其中 $k$ 是 diffusion timestep、$a^k_t$ 是被加噪的 action chunk、$o$ 是观测条件。
- **训练**：state-based 4500 epoch、image-based 3000 epoch、100 denoising step。
- **推理**：DDIM 加速到 **10 step**；3080 上 0.1 s 单次预测。
- **视觉编码**：ResNet-18 backbone + FiLM conditioning（CNN 分支）。
- **4 benchmark 12 task**：
  - **Robomimic（5）**：Lift / Can / Square / Transport / ToolHang
  - **Push-T（1）**：源自 IBC
  - **Multimodal Block Pushing（1）**：源自 BET
  - **Franka Kitchen（1）**：源自 Relay Policy Learning
  - **真机/双臂（5）**：Push-T 真机 / Mug Flip / 6DoF Pour / Sauce Spread / Egg Beater / Mat Unrolling / Shirt Folding
- **3 个 baseline**：LSTM-GMM (BC-RNN)、IBC、BET——分别代表 RNN、隐式 BC、Transformer-based 离散 BC。

### 与我研究的关联（T1D）

- **第二个低算力对照**：ResNet-18 + 1D CNN/Transformer 的轻量轨迹，4090 24 GB 应能轻松全参微调（论文未明说显存上限，但 3080 能推理意味着训练显存也不大）。
- **DDPM noise prediction loss 可直接套 RobustVLA 的 output-robust**：改造为 $\|\epsilon - \epsilon_\theta\|^2 + \lambda_{out} \max_\delta \|\epsilon - \epsilon_\theta - \delta\|^2$，内层 max 用 PGD 几步即可解。这是方向三立项中**最低风险、最快可验证**的实验。
- **Multimodal action 与鲁棒性的潜在交互**：DDPM 不模式坍塌的特性可能让鲁棒训练增益更明显（worst-case δ 可以推动模型探索更多 action mode）。
- **Receding horizon 与扰动累积**：$T_a = 8$ 频繁 re-predict 意味着扰动不会被长时间累积，可能比 ACT 的 k=100 在扰动下更鲁棒——这是个可测的假设。
- **可复用模块**：
  - CNN-based 1D temporal diffusion 模块是 plug-and-play 的 action head，可以替换任何 VLA 的输出头。
  - DDIM 推理加速（100 → 10 step）是 Efficient VLA 推理优化的标准技巧。
  - Robomimic 5 task 是 imitation learning 的金标准 benchmark，鲁棒性实验可以在此评测。

### 待解问题 / 后续追读

- **DP 在 LIBERO / RoboTwin 上的表现**？论文用的是 Robomimic / Push-T / Kitchen，与 VLA 主流的 LIBERO 评测协议不重合，需要自己 port。
- **DDPM 100 步训练对鲁棒训练外层 max 优化的影响**：worst-case δ 是否需要对每个 diffusion timestep 单独找？还是只在小 k（高噪声）时找？
- **Transformer vs CNN 在鲁棒训练下的相对优势**：原论文说 Transformer 超参敏感，鲁棒训练的额外正则是否能稳定 Transformer？
- **追读**：
  1. π0 原论文（flow matching vs DDPM 的对比）
  2. Consistency Policy / Streaming Diffusion Policy（DP 的推理加速后续工作）
  3. 3D Diffusion Policy（DP3，从 2D 扩展到 3D 点云观测）
  4. ACT（同期模仿学习对照）
  5. Robomimic benchmark 论文（理解 baseline）

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:20:35.018+08:00 %%
