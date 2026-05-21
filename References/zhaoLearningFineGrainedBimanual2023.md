---
title: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
authors: "Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn"
year: "2023"
journal: ""
doi: "10.48550/arXiv.2304.13705"
citekey: "zhaoLearningFineGrainedBimanual2023"
zotero-link: "zotero://select/library/items/J72RKYHL"
itemType: "preprint"
tags: [literature, T1D]
---

# Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

> [!info] 元信息
> - **作者**：Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn
> - **日期**：2023-04-23
> - **来源**：
> - **DOI**：[10.48550/arXiv.2304.13705](https://doi.org/10.48550/arXiv.2304.13705)
> - **Zotero**：[在 Zotero 中打开](zotero://select/library/items/J72RKYHL)
> - **Citekey**：`@zhaoLearningFineGrainedBimanual2023`

## 📄 Abstract

Fine manipulation tasks, such as threading cable ties or slotting a battery, are notoriously difficult for robots because they require precision, careful coordination of contact forces, and closed-loop visual feedback. Performing these tasks typically requires high-end robots, accurate sensors, or careful calibration, which can be expensive and difficult to set up. Can learning enable low-cost and imprecise hardware to perform these fine manipulation tasks? We present a low-cost system that performs end-to-end imitation learning directly from real demonstrations, collected with a custom teleoperation interface. Imitation learning, however, presents its own challenges, particularly in high-precision domains: errors in the policy can compound over time, and human demonstrations can be non-stationary. To address these challenges, we develop a simple yet novel algorithm, Action Chunking with Transformers (ACT), which learns a generative model over action sequences. ACT allows the robot to learn 6 difficult tasks in the real world, such as opening a translucent condiment cup and slotting a battery with 80-90% success, with only 10 minutes worth of demonstrations. Project website: https://tonyzhaozh.github.io/aloha/

## 🧠 我的思考
%% begin my-thoughts %%

### 核心观点

1. **极致低算力天花板**：80 M 参数（比 π0 小 40×）、单卡 11 G RTX 2080 Ti 训练 5 小时、推理 0.01 s/step——证明精细操作不必非要 VLA 大模型。
2. **Action chunking 是核心创新**：把单步策略改成 k 步序列预测（k=100、对应 50 Hz 下 2 秒），把 compounding error 缩小 k 倍；ablation 显示 k=1 时 4 任务平均成功率 1%、k=100 时 44%（**43 个百分点提升**）。
3. **CVAE 对 noisy 人类示教是命门**：消融中人类数据下去掉 CVAE，成功率从 35.3% 跌到 2%；脚本数据上则几乎无差异——说明 CVAE 的 style variable z 是吸收人类 stochasticity 的关键。
4. **Temporal ensemble 是廉价 inference-time trick**：训练免费、推理多几次 forward 即可，TE 给 ACT 带 +3.3%、给 BC-ConvMLP 带 +4%。
5. **6 个真机任务、10 分钟演示、80-90% 成功率**：这是 2023 年模仿学习的真机天花板，直到 RT-2 / OpenVLA 出现才被语言条件突破。

### 方法论

- **硬件 ALOHA**：双 ViperX 6-DoF 臂（$5600/支）+ 双 WidowX leader 臂（$3300/支）+ 4 个 Logitech C922x 480×640 摄像头 + 3D 打印夹爪，总成本 **< $20k**，2 小时即可组装；50 Hz 关节空间 teleoperation。
- **算法**（Algorithm 1 训练 / Algorithm 2 推理）：
  - 输入：4 摄像头图像（480×640×3）+ 双臂关节位置（14 维）
  - 视觉 backbone：4 个独立 ResNet-18，flatten 后得 1200×512 序列
  - Transformer encoder（输入 1202×512，加 CLS token + style variable z）
  - Transformer decoder：交叉注意力，输出 k×14 动作（绝对关节位置，**L1 loss**——L1 比 L2 更适合精细动作建模）
  - CVAE encoder：BERT-like，把 (action chunk + joint obs) 编码成 z 的均值方差（推理时 z=0）
- **6 真机任务**：Slide Ziploc / Slot Battery（这两个表格里）、Open Cup / Thread Velcro / Prep Tape / Put On Shoe（这四个表 II）。每任务 50 demo（Thread Velcro 100）。
- **成功率**：Slide Ziploc 92/96/88、Slot Battery 100/100/96、Open Cup 100/96/84、Thread Velcro 92/40/20（最弱）、Prep Tape 96/92/72/64、Put On Shoe 100/92/92/92。
- **2 仿真任务**：Cube Transfer、Bimanual Insertion，平均比次优 baseline 高 59% / 49% / 29% / 20%（不同 subtask）。

### 与我研究的关联（T1D）

- **Efficient VLA 的极致下界对照**：80 M / 单卡 11 G GPU / 5 h 训练——如果鲁棒训练在 ACT 上 work，那 Efficient VLA 完全可以走"轻量骨干 + 鲁棒训练"路线，绕开 46 GB 显存瓶颈。
- **CVAE 损失的可改造性**：ACT 的 L1 reconstruction + β·KL 在数学形式上与 RobustVLA 的 flow-matching loss 同构，可以平行地加 worst-case δ 项：$\|a - \hat{a}\|_1 + \lambda_{out} \max_\delta \|a - \hat{a} - \delta\|_1$。L1 下 PGD 退化为 FGSM-like 单步，**计算开销极低**。
- **Action chunking 与鲁棒性的交互**：chunking horizon k 越大，理论上对 action 噪声越鲁棒（噪声被时间平均稀释）；这是一个值得测的假设——RobustVLA 没做 chunking-noise 的交叉消融。
- **硬件契合度**：4090 / 5090 单卡能轻松训练，且推理 0.01 s 远超实时控制需求。
- **可复用模块**：
  - Temporal ensemble 是 inference-time 加固技巧，可直接套到任何序列预测策略上。
  - 4 摄像头 + ResNet-18 视觉编码器是双臂任务的标准配置，可复用到 RoboTwin。
- **局限**：ACT 没有语言输入，instruction 模态扰动不适用（只能做 12/17 扰动）。

### 待解问题 / 后续追读

- **ACT 在 LIBERO 上的成功率究竟多少**？论文只在 ALOHA / MuJoCo cube 上测了，没在 LIBERO/CALVIN 等标准 VLA benchmark 上跑过。
- **action chunking 是否对所有任务都最优**？长程 / 高频反应型任务（juggling）下 k=100 是否过大？
- **多任务版本的 ACT**：原版每任务独立训，多任务下 80 M 是否够？是否需要加语言条件成为 mini-VLA？
- **追读**：
  1. ALOHA 2 / Mobile ALOHA（同一作者后续工作）
  2. Diffusion Policy（同期、同 imitation learning 范式，DDPM vs CVAE 的对比）
  3. RT-1（语言条件模仿学习的代表）
  4. BeT (Behavior Transformer)、IBC、VINN（ACT 的 4 个 baseline，理解失败模式）

%% end my-thoughts %%

## ✏️ PDF 高亮与注释
%% begin annotations %%

%% end annotations %%


%% Import Date: 2026-05-21T17:20:12.001+08:00 %%
