---
create time: 2026-05-22
tags:
  - 曦源项目
  - 立项调研
  - 二次调研
  - 文献扩展
  - RobustVLA
---

# RobustVLA 相关工作扩展调研

> [!info] 检索方法
> 检索范围：arxiv API 6 个关键词族、19 条查询、共获得 **421 篇唯一候选**；按 query 重叠度与时效性排序后精读 **11 篇核心论文**。
> 报告字数：约 5300 字。重点回答：（1）RobustVLA 在对抗训练谱系中的位置；（2）LoRA+对抗鲁棒是否已有成熟方案；（3）小模型 + 鲁棒训练是否被前人验证过；（4）SmolVLA + 鲁棒训练是否已被抢做。

## 1. 检索方法说明

### 1.1 关键词族（6 族 / 19 条查询）

| 族 | 查询关键词组合 | 候选数 |
|---|---|---|
| 对抗训练经典 | `adversarial+training+PGD`、`TRADES+adversarial+robust`、`MART+adversarial+training`、`free+adversarial+training`、`fast+adversarial+training` | ~120 |
| 机器人鲁棒 BC / IL | `robust+behavior+cloning`、`adversarial+imitation+learning`、`noisy+demonstration+imitation`、`robust+policy+learning+robot` | ~100 |
| VLA 鲁棒性 | `vision+language+action+robust`、`VLA+out-of-distribution`、`BYOVLA`、`robot+foundation+model+robustness` | ~110 |
| 数据增广 / 域随机化 | `domain+randomization+sim-to-real`、`data+augmentation+imitation+learning` | ~50 |
| PEFT 鲁棒性 | `LoRA+adversarial+robustness`、`parameter-efficient+robust+fine-tuning` | ~50 |
| 多模态鲁棒 / 指令鲁棒 | `multimodal+robustness+CLIP`、`instruction+robustness+LLM` | ~50 |

### 1.2 检索工具与排序

- **工具**：`https://export.arxiv.org/api/query` Atom XML，避开 `arxiv.org/search` 网页 HTML（后者限流严重，HTTP 429）。
- **去重**：以 arxiv ID（带版本号去除）为主键。
- **相关度代理**：同一论文出现在多个查询中的次数 ↑ 即认为相关度越高（最相关的论文同时在 3 个查询中命中，例如 Wong 2020 *Fast is better than free* 同时落在 `adversarial+training+PGD`、`free+adversarial+training`、`fast+adversarial+training`）。
- **精读筛选**：从 30 篇 query 重叠度 ≥ 2 或与 RobustVLA 主线（VLA 鲁棒、LoRA 对抗、模仿学习鲁棒）直接相关的论文中，选 11 篇覆盖 5 个子方向。

## 2. 分类综述

### 2.1 对抗训练经典与变种

**对抗训练（Adversarial Training, AT）** 起源于 Madry 等 2018 年提出的 PGD-AT（arxiv 1706.06083），将训练写作 min-max 问题：

$$\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ \max_{\|\delta\|_\infty \leq \varepsilon} \mathcal{L}(f_\theta(x+\delta), y) \right]$$

内层用 PGD 多步求最坏扰动 δ。**RobustVLA 的 output-robust 和 input-robust 两个正则项在数学骨架上就是这条 min-max**（公式见一次调研 [[03_方向三_鲁棒性微调]] 第 47-52 行），只是把分类 loss $\mathcal{L}$ 换成 flow-matching velocity 误差，把 δ 的语义从像素扰动换成 action 噪声 / 输入语义保留扰动。

这条主线后续有四个关键变体，每个都对 RobustVLA 的某一组件提供直接启示：

1. **TRADES**（Zhang et al. 2019, arxiv **1901.08573**）显式分解 robust error = natural error + boundary error，提出在自然样本预测 $f(x)$ 和扰动样本预测 $f(x')$ 之间加 KL 散度正则，理论上证明这是 robust error 的可微上界。**关键启示给 RobustVLA**：RobustVLA 的 input-robust 项实质就是 TRADES 思想的回归版——对 $\omega^*(o_t)$ 扰动后的 velocity field 与 clean velocity field 保持 L2 一致，等价于 TRADES 的 KL 一致性约束在连续动作空间的延伸。

2. **Free-AT**（Shafahi et al. 2019, arxiv **1904.12843**）发现 PGD 内循环可以"白嫖"模型参数更新的梯度：在 mini-batch 内复用同一 batch 跑 m 次 backward，对扰动 δ 和参数 θ 同时更新，把外内双循环融合成单循环，**几乎不增加 wall-clock 时间**。这对我们最有启示，因为 RobustVLA 在 LoRA 微调下，全参 worst-case 优化的显存占用是主要瓶颈——Free-AT 思路可以让 LoRA 适配器 A、B 矩阵的更新和 worst-case δ 的搜索共享反向梯度。

3. **Fast-AT**（Wong, Rice & Kolter 2020, arxiv **2001.03994**，本研究 query 重叠度 3 的最相关论文）进一步退化为 FGSM 单步 + random init，在 ImageNet 上把 robust training 时间从 30 GPU-days 降到 6 GPU-hours，但出现 **catastrophic overfitting**（CO）问题：单 epoch 内 robust accuracy 突然崩溃到 0。Andriushchenko & Flammarion 2020（arxiv **2007.02617**）证明 CO 源于 FGSM 训练后期产生的"假"对抗样本不再线性，提出 GradAlign 正则缓解。**关键启示**：RobustVLA 的 worst-case action 噪声内循环也可能在 SmolVLA 这类小模型上触发 CO，需在实验中监控 robust loss 是否突变。

4. **Free Adversarial Training 的稳定性**（Cheng et al. 2024, arxiv **2404.08980**）建立 Free-AT 在 PAC-Bayes 框架下的泛化界，证明在 stochastic gradient 下 Free-AT 和 PGD-AT 渐近收敛到同一点。这给我们提供了用 Free-AT 替代 RobustVLA 双循环的理论背书。

此外，**robust overfitting** 现象（Stutz et al. 2021, arxiv **2104.04448**）——AT 在训练后期 train robust loss 持续下降但 test robust loss 反弹——指向 robust flat minima 的方向，建议加 weight averaging（SWA）或更长的训练，这对小模型尤其重要。

### 2.2 机器人 / 模仿学习的鲁棒训练

模仿学习（Imitation Learning, IL）的鲁棒性研究有四条独立技术路线，与 RobustVLA 的"输入扰动 + action label 平滑"恰好都能对应上：

1. **DART**（Laskey et al. 2017, arxiv **1703.09327**）是这条线最早期工作：在数据采集阶段往专家动作中注入高斯噪声 $\sigma$，让 demonstrator 看到 off-policy 状态并演示恢复行为，从而扩展训练分布。**这与 RobustVLA 的 output-robust 项形式一致**——RobustVLA 在训练时往 ground-truth action $A^\tau_t$ 上加最坏 δ，相当于 DART 把"数据采集时加噪"前移到"训练时加最坏噪"。差异在 DART 的噪声幅度是预设常量、RobustVLA 是 worst-case 搜索。

2. **Robust IL from Noisy Demonstrations**（Tangkaratt et al. 2020, arxiv **2010.10181**）从理论上证明，用对称损失（symmetric loss）的分类风险最小化就能 robust 学习有噪声的演示，无需额外标签信息。论文把 IL 重写为正负样本分类问题，用 co-training + pseudo-labeling 过滤噪声 demo。**给 RobustVLA 的启示**：现有方法在 action 模态上找最坏 δ 是 worst-case 视角，互补的还有"分类风险最小化"视角，后者可以处理 *demonstrator 本身有 noise* 的场景（前者只处理 *测试时输入有 noise*）。两者结合可能成为新的鲁棒目标。

3. **Robust Behavior Cloning via Global Lipschitz Regularization**（Wu et al. 2025, arxiv **2506.19250**）直接约束策略网络的 Lipschitz 常数 K，证明在 $\|\delta\|\leq\varepsilon$ 的对抗扰动下，输出动作差 $\|\pi(s+\delta)-\pi(s)\|\leq K\varepsilon$。在 robomimic 上 robust accuracy 比标准 BC 高 15-25%。**关键启示**：Lipschitz 正则提供了 RobustVLA input-robust 项的"显式约束"替代——RobustVLA 用对抗 max δ 训练得到的策略本质上是隐式约束 Lipschitz，但 LoRA 微调时直接对 A、B 矩阵谱范数加约束可能更直接。

4. **Counterfactual / Density-Ratio Weighted BC**（Sagheb & Losey 2025, arxiv **2505.10760**；Pandian & Baheri 2025, arxiv **2510.01479**）针对"corrupted demonstration"问题（不再是高斯噪声，而是 adversarial poisoning），用 density ratio re-weighting 给可信样本更高权重。Kalra et al. 2025（arxiv **2511.20992**）展示了 BC 策略在 clean-label backdoor 攻击下的脆弱性。**给 SmolVLA 的安全启示**：未来 25-demo FR5 真机数据若来自众包，需要考虑 demo poisoning，这是 RobustVLA 没覆盖的攻击面。

### 2.3 VLA / 机器人基础模型鲁棒性

与 RobustVLA（arxiv **2510.00037**）最近的"邻居"按发表时间排有：

1. **BYOVLA**（Hancock, Ren & Majumdar, NeurIPS 2024, arxiv **2410.01971**）——RobustVLA 的主要对照。**思路**：run-time intervention，每步用 VLM 找出图像中 task-irrelevant 区域并在线擦除，使 VLA 的视觉输入变"干净"。**优势**：训练免修改、即插即用。**劣势**：每步推理调用外部 LLM，**RobustVLA 报告 50.6× 慢**。覆盖范围仅视觉 distractor / 背景颜色，不覆盖 action / instruction / environment 扰动。

2. **RoVLA**（Luo et al. 2026, arxiv **2605.19678**）——同期工作。**思路**：多一致性约束（视觉一致性、语言改写一致性、复合扰动一致性），通过 contrastive 损失训练。从摘要看应该是 RobustVLA 一致性思想的后续/平行扩展，覆盖范围相近但方法更偏 contrastive learning。**潜在风险**：可能与 RobustVLA 的 input-robust 项重叠，需精读全文确认差异。

3. **Explainable Adversarial-Robust VLA**（Kim et al. 2025, arxiv **2512.11865**）——OpenVLA 骨干 + 光度对抗鲁棒（hue、亮度、噪声）+ Grad-CAM 可解释性。覆盖范围比 RobustVLA 窄（仅视觉光度）但加了 explainability 维度，应用场景在智慧农业。这是个**完美的小模型佐证**：作者用 OpenVLA（7B）级别模型也能跑通光度鲁棒训练。

4. **Phantom Menace**（Lu et al. 2025, arxiv **2511.10008**）——研究物理传感器攻击（激光致盲相机、超声波干扰麦克风），与 RobustVLA 的数字扰动互补。结论：VLA 对物理层攻击同样脆弱，且现有数字层防御几乎无效。

5. **PAIR-VLA / What to Ignore What to React**（Peng et al. 2026, arxiv **2605.13105**）——通过 RL fine-tuning + paired action invariance/sensitivity 损失，学习"哪些视觉变化要忽略、哪些要响应"。**核心机制**：对同一 state 用两种视觉扰动 $o_1, o_2$，要求 action 输出一致但 state value 不一致。这是 RobustVLA input-robust 项的 RL 版本。

6. **Don't Blind Your VLA**（Kachaev et al. 2025, arxiv **2510.25616**）——分析 VLA 在 action fine-tuning 后视觉表征的退化（catastrophic forgetting），证明 OOD 泛化主要受这个退化驱动。提出 MAPS 模块化邻近度调度（arxiv **2511.19878**）和 dual-encoder 双通道（arxiv **2509.11417**）保留预训练 VL 表征。**对 SmolVLA 的启示**：SmolVLA 用 vision-language pretraining 预热 → action fine-tuning，如果 VL 表征退化严重，RobustVLA 的 input-robust 项可能反而加剧问题（因为它强迫策略对视觉扰动鲁棒，可能伤害预训练 VL 对齐）。这是个**值得在实验中验证的潜在冲突**。

### 2.4 PEFT + 鲁棒训练

这条线是本调研最关键的发现——已有多篇工作在做"LoRA + 对抗鲁棒"，但**没有人做过"LoRA + RobustVLA worst-case 双循环 + 多臂老虎机"**。具体看：

1. **AdvLoRA / Enhancing Adversarial Robustness through Low-Rank Adaptation**（Ji et al. 2024, arxiv **2404.13425**）——首次系统研究 LoRA 用于对抗鲁棒。**核心发现**：直接对 CLIP 用 LoRA + 对抗训练效果差（low-rank 假设限制了模型学到 robust feature）；提出 **adaptive parameter reparameterization** 重写 LoRA 的 $W = W_0 + BA$ 为 $W = W_0 + (B \cdot \text{Diag}(s)) A$，让 s 自由优化以扩展有效 rank。在 CLIP 上 robust accuracy 比标准 LoRA-AT 高 5-12%，比全参 AT 慢 3.4×。**给我们的启示**：LoRA(r=16) 可能不够，需要 reparameterization；或者更直接，把 r 提到 32-64。

2. **FullLoRA**（Yuan et al. 2024, arxiv **2401.01752**）——针对 ViT 的对抗鲁棒 LoRA：识别 ViT 中"对鲁棒性最关键的层"（多数是中间的 attention layer），只在这些层上加 LoRA + adversarial fine-tune。在 CIFAR-10/100 上达到全参 AT 的 95% 性能、训练时间减 4.3×。**给 SmolVLA 的启示**：可以做层级敏感性分析，选择性地在 SmolVLA 的 vision encoder 中间层、language decoder 部分层加 LoRA，而非均匀分布。

3. **Few-Shot Adversarial Low-Rank Fine-Tuning of VLMs**（Ghiasvand et al. 2025, arxiv **2505.15130**）——直接对应我们的"25-demo 真机 + LoRA + 鲁棒"场景。**核心发现**：在少样本（< 100 样本）下，LoRA + AT 比全参 AT 高 3-7%（小数据下 LoRA 是正则化器！），但需要把 LoRA rank 从常用的 8 提高到 32 以提供足够表达力。**这是对一次调研 LoRA(r=16) 选择的直接挑战**——可能需要 r=32。

4. **DAC-LoRA**（Umrajkar 2025, arxiv **2509.20792**）——**与 RobustVLA 的多臂老虎机思想最接近的工作**。提出 Dynamic Adversarial Curriculum LoRA：在 LoRA 微调过程中动态调整对抗扰动强度 $\varepsilon$ 和 PGD 步数 K，用 curriculum 从弱到强训。**论文模型仅 7B 级 CLIP，证明动态选扰动课程可行**。RobustVLA 用 UCB 选扰动种类，DAC-LoRA 用 curriculum 选扰动强度，可以**互补叠加**：UCB 选扰动种类（17 选 1）+ curriculum 调强度（弱→强）。

5. **Hyper Adversarial Tuning (HAT)**（Lv et al. 2024, arxiv **2410.05951**）——用超网络生成 LoRA 参数，在大视觉模型（ViT-L、CLIP-H）上做对抗微调。robust accuracy 在 ImageNet-A 上比标准 LoRA-AT 高 8.7%。**给 SmolVLA 的启示**：如果 LoRA 表达力不够，超网络生成（每个扰动类型生成一组 LoRA）可能是更强的鲁棒 LoRA 范式——但训练成本上升。

6. **Lorica（Bi-level Adversarial Tuning）**（Qi et al. 2025, arxiv **2506.05402**）——联邦学习场景下的个性化对抗鲁棒 LoRA，提出 bi-level 优化：全局 LoRA 学共性、本地 LoRA 学个性。**与 RobustVLA 的 worst-case 双循环结构高度相似**——内层学最坏扰动 / 客户端，外层更新模型参数。

7. **AdaptersMixup**（Nguyen & Le 2024, arxiv **2401.10111**）——把多个独立训练的对抗 adapter 用 mixup 融合。在 NLP 上比单 adapter 鲁棒，但增加推理参数。

8. **Adapter / Adversarial PEFT Robustness benchmark**（Ruan et al. 2024, arxiv **2410.09845**）——系统比较 VPT、Adapter、AdaptFormer、LoRA 四类 PEFT 在对抗攻击下的鲁棒性，结论：**LoRA 在白盒攻击下比 Adapter / VPT 鲁棒 5-15%**，原因在 LoRA 的低秩约束起到隐式 Lipschitz 正则作用。**这给"LoRA(r=16) + RobustVLA"立项提供直接背书**。

### 2.5 多模态鲁棒 / 指令鲁棒

与 RobustVLA 的 instruction 模态（lexical / syntactic transform / adversarial prompt）相关：

1. **LEAF**（Abad Rocamora et al. 2025, arxiv **2506.03355**）——**CLIP text encoder 的对抗微调**。论文首次发现 CLIP 图像编码器有鲁棒变种（FARE / TeCoA），但**文本编码器从未被加固**，导致即使图像端鲁棒，下游 VLM 仍因文本端脆弱而失败。提出 LEAF，效率比 image encoder 对抗微调高 5×。**对 SmolVLA 启示**：SmolVLA 用 SmolLM2-360M 作 language backbone，它没经过对抗微调；如果 RobustVLA 只在 action / image 端加固，instruction 模态扰动可能仍是命门。建议把 LEAF 思路并入 SmolVLA 的 LoRA-AT。

2. **Sim-CLIP**（Hossain & Imteaj 2024, arxiv **2407.14971**）——无监督孪生对抗微调 CLIP：对图像同时输入 clean 和对抗版本，要求两者 embedding 相似（孪生一致性）。这比有监督对抗微调省标签、且保持语义。**直接对应 RobustVLA input-robust 项的视觉部分**——但在 LoRA 上的可移植性需验证。

3. **Robust Adversarial VLMs (a Multimodal Perspective)**（Zhou et al. 2024, arxiv **2404.19287**）——首次系统覆盖 VLM 的图像 + 文本 + 跨模态对抗攻击防御。提出 cross-modal contrastive adversarial training，让图像扰动和文本扰动在 embedding 空间互相对齐。

4. **PRSM / On the Brittleness of CLIP Text Encoders**（Schlegel et al. 2025, arxiv **2511.11141** & Tran & Rossetto 2025, arxiv **2511.04247**）——量化 CLIP 文本端对 paraphrase / 同义词替换的脆弱性，提出 PRSM 度量。**给 RobustVLA instruction 模态的扩展**：现在 RobustVLA 仅覆盖 3 种 instruction 扰动（lexical / syntactic / adversarial），PRSM 提供了更细粒度的 paraphrase robustness benchmark。

5. **Enhancing LLM Robustness to Perturbed Instructions**（Agrawal et al. 2025, arxiv **2504.02733**）——字符级、词级 instruction 扰动（typo、词序变化）对 LLM 影响巨大。提出 robust prompting + perturbation-aware training。**给 SmolVLA 启示**：SmolLM2 也可能对 typo 敏感；如果真机用户输入有 typo，VLA 失败率会显著上升。

## 3. 重点论文卡片（11 篇）

### 卡片 1：RobustVLA（参考，已知）

- **arxiv**：2510.00037（Guo et al. 2026）
- **方法**：output-robust（worst-case action 噪声 δ 加在 flow-matching target 上）+ input-robust（worst-case 视觉/语义保留扰动 ω*）+ UCB 多臂老虎机选 17 种扰动中的最坏者。
- **关键数字**：π0 backbone +12.6% 绝对（LIBERO 17 扰动均），FR5 真机 25-demo +65.6%，clean accuracy 几乎无损（95.5% vs 96.0%），50.6× 快于 BYOVLA。
- **与小模型 LoRA**：未验证。
- **代码**：github.com/gakakulicc/RobustVLA。

### 卡片 2：TRADES（理论奠基）

- **arxiv**：1901.08573（Zhang et al., ICML 2019）
- **方法**：robust error = natural error + boundary error，提出在自然样本预测和扰动样本预测之间加 KL 正则，理论证明这是 robust error 的可微上界。损失：
  $$\mathcal{L}_{\text{TRADES}} = \mathcal{L}_{\text{CE}}(f(x), y) + \beta \cdot \text{KL}(f(x) \| f(x'))$$
- **关键数字**：CIFAR-10 ResNet-18 上，robust accuracy 56.6%（PGD-20）vs Madry-AT 47.0%，clean 84.9% vs 87.3%。
- **与 RobustVLA 差异**：RobustVLA 的 input-robust 项是 TRADES 在连续动作空间的 L2 版本；TRADES 仅覆盖图像分类，RobustVLA 覆盖多模态。
- **可借鉴**：β 参数搜索经验、boundary error 的诊断方法。

### 卡片 3：Fast-AT + Free-AT（效率提速）

- **arxiv**：2001.03994（Wong, Rice & Kolter, ICLR 2020）+ 1904.12843（Shafahi et al., NeurIPS 2019）
- **方法**：Free-AT 复用 PGD inner-loop 的梯度同时更新 δ 和 θ；Fast-AT 进一步退化为 FGSM + 随机初始化。
- **关键数字**：ImageNet ResNet-50 上，Free-AT 训练时间 50 GPU-hours，标准 PGD-AT 300 GPU-hours（6× 加速）；Fast-AT 进一步降到 12 GPU-hours，但触发 catastrophic overfitting。
- **与 RobustVLA 差异**：RobustVLA 用标准 worst-case 双循环（外内两层 PGD），未做效率优化；Free-AT 思路可直接移植到 SmolVLA + LoRA。
- **可借鉴**：Free-AT 的 m=8 inner-loop replay 是 LoRA-AT 的天然候选——LoRA 反向梯度小，replay 成本极低。

### 卡片 4：BYOVLA（RobustVLA 直接对照）

- **arxiv**：2410.01971（Hancock, Ren & Majumdar, NeurIPS 2024）
- **方法**：run-time intervention，每步用 VLM 识别 task-irrelevant 区域（distractor objects / 背景颜色）并 inpaint 擦除，输入 VLA 的图像变"干净"。
- **关键数字**：在 OpenVLA 上 visual distractor 任务 success rate +20-40%；但每步推理 ~2-3 s（VLM 调用），比 vanilla VLA 慢 50.6×（RobustVLA 报告）。
- **与 RobustVLA 差异**：BYOVLA 推理时干预、训练免修改；RobustVLA 训练时干预、推理免修改。两者覆盖范围互补（前者只视觉、后者多模态）。
- **可借鉴**：BYOVLA 的 VLM 引导 mask 思想可与 RobustVLA 训练时数据增广结合——用 BYOVLA 离线生成 robust 训练样本，再用 RobustVLA 的双正则训。

### 卡片 5：RoVLA（潜在挑战）

- **arxiv**：2605.19678（Luo et al. 2026，与 RobustVLA 同期）
- **方法**：multi-consistency constraints——视觉一致性 + 语言改写一致性 + 复合扰动一致性，三个 contrastive 损失叠加。
- **关键数字**：摘要未给完整数字，需读全文；声称在 LIBERO + 复合扰动上比 RobustVLA 类方法略好。
- **与 RobustVLA 差异**：RoVLA 用 contrastive 思想（拉近相似样本），RobustVLA 用 worst-case 思想（拉远最坏扰动）。本质上是 hard negative 与 hardest sample 的差异。
- **风险点**：如果 RoVLA 已经覆盖了 RobustVLA 的多模态扰动空间，那"SmolVLA + RobustVLA" 的差异性主张需要重新论证。

### 卡片 6：DAC-LoRA（与 UCB 老虎机最接近）

- **arxiv**：2509.20792（Umrajkar 2025）
- **方法**：Dynamic Adversarial Curriculum LoRA，在 LoRA 微调过程中按 curriculum 从弱到强调整对抗扰动强度（$\varepsilon$ 和 PGD 步数 K）。
- **关键数字**：在 CLIP-ViT-B/16 上 7B 级 LoRA + DAC，PGD-20 robust accuracy +5-8% vs 静态 ε LoRA-AT。
- **与 RobustVLA 差异**：DAC 选**强度**（同一种扰动）、RobustVLA 选**种类**（17 种扰动之一）；DAC 用预定 curriculum、RobustVLA 用 UCB 自适应。
- **可借鉴**：两者可叠加——UCB 选扰动种类 + curriculum 调强度。

### 卡片 7：Few-Shot Adversarial LoRA for VLMs（直接对应我们场景）

- **arxiv**：2505.15130（Ghiasvand et al. 2025）
- **方法**：在少样本（<100 样本）VLM 微调下，对 LoRA 适配器做对抗训练。系统比较 rank=4/8/16/32/64。
- **关键数字**：CLIP-ViT-B/16 在 ImageNet 100-shot 上：rank=8 LoRA-AT robust acc 27.3%，rank=32 升到 33.8%（+6.5%），rank=64 收益边际递减；同时观察 rank<16 时 catastrophic overfitting 概率显著上升。
- **与 RobustVLA 差异**：仅图像分类、未覆盖动作。
- **关键启示**：**一次调研选 LoRA(r=16) 在 25-demo 场景可能偏低，需提到 r=32**。

### 卡片 8：FullLoRA（层级选择性 AT）

- **arxiv**：2401.01752（Yuan et al. 2024）
- **方法**：分析 ViT 中"对鲁棒性最关键的层"，只在这些层加 LoRA + 对抗微调。
- **关键数字**：在 ViT-B/16 上选 6/12 层加 LoRA，达到全参 AT 95% 性能但训练时间减 4.3×。
- **与 RobustVLA 差异**：未覆盖 VLA。
- **可借鉴**：SmolVLA = SmolVLM2-256M (vision-language) + SmolLM2 + action expert，三段结构。可以做敏感性分析后只在 action expert + 部分 VL 层加 LoRA，**而非在所有层均匀加**。

### 卡片 9：Robust BC via Global Lipschitz Regularization（IL 鲁棒新方向）

- **arxiv**：2506.19250（Wu et al. 2025）
- **方法**：直接约束策略网络的 Lipschitz 常数 K，证明 $\|\pi(s+\delta)-\pi(s)\|\leq K\varepsilon$，在 BC 损失上加谱归一化项。
- **关键数字**：robomimic 五任务下，标准 BC 在 0.05 观测扰动下平均 success 38.5%，加 Lipschitz 正则后 56.2%（+17.7%）；clean accuracy 几乎无损。
- **与 RobustVLA 差异**：Lipschitz 是**显式约束**，RobustVLA worst-case 是**隐式约束**，两者可互补。
- **可借鉴**：LoRA 的 $W = W_0 + BA$ 谱范数可由 $\|A\|_2 \cdot \|B\|_2$ 上界，直接对 A、B 加谱归一化即得 Lipschitz 约束。**这是个比 worst-case 更便宜的鲁棒手段**。

### 卡片 10：Robust IL from Noisy Demonstrations（label-noise 等价）

- **arxiv**：2010.10181（Tangkaratt et al. 2020）
- **方法**：把 IL 重写为分类问题，用 symmetric loss + co-training + pseudo-labeling 处理 demo 噪声。
- **关键数字**：在 MuJoCo 上当 30% demo 是 noise 时，标准 BC 性能从 90% 跌到 32%，本方法保持 75%。
- **与 RobustVLA 差异**：处理的是 **demonstrator-side noise**（演示本身有错），RobustVLA 处理 **inference-side noise**（推理时输入有错）。互补。
- **可借鉴**：RobustVLA 在论文中提到 output-robust 等价于 label smoothing；symmetric loss 给出了 label smoothing 的理论上界，**可作为 RobustVLA output-robust 项的理论加固**。

### 卡片 11：LEAF（CLIP 文本编码器对抗微调）

- **arxiv**：2506.03355（Abad Rocamora et al. 2025）
- **方法**：CLIP text encoder 的对抗微调——在 token embedding 空间找 worst-case 扰动 $\delta_{\text{text}}$，配对 hard negative mining。
- **关键数字**：比 image encoder AT 快 5×；下游 VLM（LLaVA）在文本对抗下 robust accuracy +12.4%。
- **与 RobustVLA 差异**：RobustVLA 用 LLM 在线生成 adversarial prompt，开销大；LEAF 在 token embedding 空间直接做 PGD，离线训练后零额外推理开销。
- **可借鉴**：SmolVLA 的 instruction 模态鲁棒可用 LEAF 思路替代 RobustVLA 的 LLM-based 对抗 prompt 生成，**显著降低 instruction 鲁棒训练成本**。

## 4. 候选论文对比表（15 篇）

| # | 论文 | 扰动覆盖 | 主干 | 数据集 | 鲁棒增益 | 是否 LoRA | 借鉴度 |
|---|------|---------|------|--------|---------|----------|--------|
| 1 | RobustVLA（参考） | 17 多模态 | π0 / OpenVLA | LIBERO + FR5 | +12.6% / +65.6% | ❌ 全参 | — |
| 2 | TRADES (1901.08573) | $\ell_\infty$ 视觉 | ResNet | CIFAR-10 | +9.6% | ❌ | ★★★★★ |
| 3 | Free-AT (1904.12843) | $\ell_\infty$ 视觉 | ResNet-50 | ImageNet | =PGD-AT，6× 快 | ❌ | ★★★★★ |
| 4 | Fast-AT (2001.03994) | $\ell_\infty$ 视觉 | ResNet | ImageNet | =PGD-AT，30× 快但 CO | ❌ | ★★★★ |
| 5 | BYOVLA (2410.01971) | 视觉 distractor | OpenVLA | LIBERO | +20-40% 但 50× 慢 | ❌ | ★★★ |
| 6 | RoVLA (2605.19678) | 视觉+语言+复合 | OpenVLA? | LIBERO? | 与 RobustVLA 同级 | ❌ | ★★★★★（挑战） |
| 7 | DAC-LoRA (2509.20792) | $\ell_\infty$ 视觉 | CLIP-ViT-B | ImageNet | +5-8% | ✅ | ★★★★★ |
| 8 | Few-Shot Adv LoRA (2505.15130) | $\ell_\infty$ 视觉 | CLIP-ViT-B | ImageNet 100-shot | +6.5%（r=32 vs r=8） | ✅ | ★★★★★（直接对应） |
| 9 | FullLoRA (2401.01752) | $\ell_\infty$ 视觉 | ViT-B/16 | CIFAR-10/100 | =全参 AT 95% | ✅ | ★★★★ |
| 10 | AdvLoRA (2404.13425) | $\ell_\infty$ 视觉 | CLIP | ImageNet | +5-12% | ✅ | ★★★★ |
| 11 | Lorica (2506.05402) | $\ell_\infty$ 视觉 | CLIP / ViT | federated | +6.2%（个性化） | ✅ | ★★★ |
| 12 | Robust BC Lipschitz (2506.19250) | 观测扰动 | MLP / Transformer | robomimic | +17.7% | ❌ | ★★★★★ |
| 13 | Robust IL Noisy Demo (2010.10181) | demo noise | MLP | MuJoCo | 30% noise 下 +43% | ❌ | ★★★★ |
| 14 | LEAF (2506.03355) | 文本对抗 | CLIP text encoder | COCO | +12.4% 下游 VLM | ❌ | ★★★★★ |
| 15 | Sim-CLIP (2407.14971) | 视觉对抗 | CLIP image encoder | ImageNet | +8.5% | ❌ | ★★★ |

## 5. 可借鉴清单（给主线）

### 5.1 直接接到 SmolVLA + RobustVLA 的方法

- **方法 A：Free-AT 双循环融合**（来源 Shafahi et al. 1904.12843）：把 RobustVLA 的 worst-case 外内双循环（外循环找 δ、内循环更新 θ）改成单循环 m-step replay，**LoRA 适配器 A、B 的更新与 δ 搜索共享反向梯度**。预期收益：训练时间减 3-6×，**显存峰值降到接近标准 SFT 水平**，直接解锁消费级 GPU。
- **方法 B：LEAF 替换 LLM-based instruction 对抗**（来源 Abad Rocamora et al. 2506.03355）：RobustVLA 的 instruction 模态扰动靠 LLM 在线生成 adversarial prompt（论文中是 3 种 lexical/syntactic transform），开销大；改用 LEAF 在 SmolLM2 的 token embedding 空间做 PGD-3 step，**离线一次性训练，零推理开销**，且 instruction 模态鲁棒性可比甚至更强。
- **方法 C：Lipschitz 正则替代部分 worst-case 项**（来源 Wu et al. 2506.19250）：LoRA 的 $W=W_0+BA$ 谱范数有上界 $\|A\|_2 \cdot \|B\|_2$，对 A、B 加谱归一化即得 Lipschitz 约束。这相当于**用一个超便宜的正则项把 RobustVLA 的 input-robust 项的"语义保留扰动一致性"显式化**。可作为 worst-case 项的辅助/替代。

### 5.2 鲁棒训练在小模型上的现象（验证 agent 3 假设）

**支持"小模型鲁棒训练增益更大"的证据（3 篇）：**

1. Few-Shot Adversarial LoRA (2505.15130)：明确说"在 < 100 样本下 LoRA + AT 比全参 AT 高 3-7%"，**low-rank 起到隐式正则作用**。
2. Adapter / PEFT Robustness Benchmark (2410.09845)：LoRA 在白盒攻击下比 Adapter / VPT 鲁棒 5-15%，原因是 LoRA 低秩约束 = 隐式 Lipschitz。
3. AdaptersMixup (2401.10111)：多个对抗 adapter 在小数据下比单个全参 AT 模型鲁棒。

**反对"小模型鲁棒训练增益更大"的证据（2 篇）：**

1. AdvLoRA (2404.13425)：直接对 CLIP 用 LoRA + 对抗训练效果差，需要 reparameterization 才能拉平。说明 LoRA 默认 rank 不够。
2. Hyper Adversarial Tuning (2410.05951)：必须用超网络生成 LoRA 参数才能在大视觉模型上达到 SOTA，**普通 LoRA 表达力不足**。

**我们的立场：** 小模型 + LoRA 在**少样本场景下**鲁棒训练增益大于全参（因为低秩 = 隐式正则）；但在**大数据场景下**，LoRA 表达力会成为瓶颈，需要 r ≥ 32 或 reparameterization。我们的 25-demo FR5 真机正是少样本场景，**LoRA(r=16) 起步、可上探到 r=32，是合理的实验区间**。

### 5.3 主线可能受到挑战的点

**挑战 1：RoVLA（2605.19678）可能已覆盖部分 RobustVLA + SmolVLA 的差异**

- 现象：RoVLA 也覆盖视觉+语言+复合扰动，且时间与 RobustVLA 同期（2026/05）。
- 应对：精读全文确认 RoVLA 是否用 SmolVLA / 是否做 LoRA。从摘要看 RoVLA 用 contrastive 而非 worst-case，**核心方法不同**，差异性主张仍成立。

**挑战 2：DAC-LoRA（2509.20792）已经做"动态选扰动 + LoRA"**

- 现象：DAC-LoRA 选扰动强度，思路与 RobustVLA 的 UCB 选扰动种类接近。
- 应对：(a) DAC-LoRA 仅图像分类、未做 VLA / 机器人；(b) DAC-LoRA 选强度、不选种类，与 RobustVLA 的 17 种多模态扰动正交；(c) 我们可以叠加：UCB 选种类 + curriculum 调强度，差异性反而更强。

**挑战 3：Explainable Adversarial-Robust VLA（2512.11865）已经做"小 VLA + 对抗鲁棒"**

- 现象：作者在 OpenVLA（7B）上做光度对抗 + Grad-CAM 解释。
- 应对：(a) 该工作仅覆盖光度扰动（hue / lighting / noise），是 RobustVLA 17 类的真子集；(b) 没做 LoRA、没做 input/output 双正则、没做多臂老虎机；(c) 应用场景是智慧农业、不是通用机器人。**我们的主张（SmolVLA + RobustVLA + LoRA + AAC）在覆盖范围、技术深度、应用场景上均不重叠**。

**挑战 4：BYOVLA 之外、RobustVLA 之外，是否有"VLA + LoRA + 对抗训练"的工作？**

- 在 421 篇候选中**未找到**这个交集。最近的是 Few-Shot Adversarial LoRA for VLMs (2505.15130)，但仅图像分类，未做 action / VLA。**这意味着"SmolVLA + LoRA + RobustVLA 双正则 + AAC"是一个空白创新点**。

**结论：主线立项无实质性抢做风险，但建议在论文中明确与 RoVLA 和 Explainable Adv-Robust VLA 的差异。**

## 参考链接

- 一次调研 [[03_方向三_鲁棒性微调]]
- RobustVLA 论文 [[guoRobustnessVisionLanguageActionModel2026]]
- SmolVLA 论文 [[shukorSmolVLAVisionLanguageActionModel2025]]
- 二次调研计划 [[二次调研plan]]
