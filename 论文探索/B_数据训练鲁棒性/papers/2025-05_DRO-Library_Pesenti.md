---
title: "dro: A Python Library for Distributionally Robust Optimization"
authors: "Jiashuo Liu, Tianyu Wang, Henry Lam, Hongseok Namkoong, Jose Blanchet (Stanford & Columbia)"
year: "2025"
journal: "arXiv preprint"
doi: "10.48550/arXiv.2505.23565"
arxiv: "2505.23565"
venue: "arXiv 2025-05"
citekey: "liuDROPythonLibrary2025"
itemType: "preprint / software"
status: "精读完成"
tier: "⭐⭐⭐ 必读 · B6 工具基础设施"
tags: [literature, T1D, 主线B, B6, DRO, 工具库, Python, 工程化]
---

# dro Library — DRO Python 库精读笔记

> [!info] 元信息
> - **arXiv**：[2505.23565](https://arxiv.org/abs/2505.23565)
> - **GitHub**：https://github.com/namkoong-lab/dro
> - **核心定位**：把 DRO 各类 ambiguity set（KL / Wasserstein / chi^2 / Sinkhorn / MMD）统一进一个 sklearn-style API，**为工程团队提供低门槛 DRO 工具**
> - **B6 价值**：从「写论文用」到「直接调库」的范式转变，对**曦源工程实施 DRO 鲁棒训练**至关重要

## 📄 Abstract

DRO 在过去十年理论快速发展但工程落地慢，因为不同 ambiguity set 求解器各自实现、API 不一致、复现门槛高。本文发布 **dro library**：
- 统一 API：`fit(X, y, ambiguity='KL', eps=0.1)`
- 支持 10+ 种 ambiguity set
- 支持 supervised learning（regression / classification）+ unsupervised（risk）
- 与 scikit-learn 兼容
- 文档 + tutorial 完整

## 🧠 我的思考

%% begin my-thoughts %%

### 核心观点

1. **「DRO 工程化」是 2024-2026 关键 enabler**：
   - 在此之前，做 DRO 实验需手写 dual 优化、Lagrangian projection、constraint set
   - 现在：`from dro import DROClassifier; m = DROClassifier(ambiguity='Wasserstein', eps=0.05); m.fit(X, y)`
   - **门槛骤降 → DRO 大规模进入 robotics / VLA 训练 pipeline 的时机成熟**

2. **Ambiguity set 选择的实用指南**：
   - **KL divergence**：closed-form dual，最快，但对 OOD 不友好（KL 对 0 测度点发散）
   - **Wasserstein**：geometrically intuitive，对 perturbation 鲁棒，但优化贵
   - **chi^2**：折中，适合大多数场景
   - **Sinkhorn**：entropy-regularized Wasserstein，更快但近似
   - **MMD**：kernel-based，适合 distribution shift 但 kernel 选择难
   - 本文表格列出每种 set 的「样本复杂度」「优化复杂度」「robust radius 直觉」

3. **「DRO as a service」的趋势**：
   - 类似 TensorFlow / PyTorch 让深度学习平民化
   - dro library 让 DRO 平民化
   - 预测：未来 3 年 robotics 论文里 DRO 出现频率会显著上升

### 方法论

#### 库结构
```
dro/
├── ambiguity/          # KL, Wasserstein, chi2, Sinkhorn, MMD
├── solvers/            # primal / dual / SAA
├── models/             # DROClassifier, DRORegressor, DROCustom
└── utils/              # cross-val for eps
```

#### 典型使用
```python
from dro import DROClassifier
import numpy as np

# Fit DRO model with Wasserstein ball radius 0.05
model = DROClassifier(
    base_model='logistic',
    ambiguity='Wasserstein',
    eps=0.05,
    p=2  # Wasserstein-2
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Cross-validate epsilon
from dro.utils import cv_eps
best_eps = cv_eps(X_train, y_train, eps_grid=[0.01, 0.05, 0.1])
```

#### 与 sklearn / PyTorch 集成
- sklearn：直接作为 estimator 用 GridSearchCV
- PyTorch：提供 `DROLoss` module 作为 loss function，可插入任何 nn.Module

### 与曦源的关联（重点）

1. **直接实战工具**：
   - 我们写 BC / VLA fine-tuning 时，可以把 cross-entropy loss 替换为 `DROLoss(ambiguity='Wasserstein', eps=0.05)`
   - 一行代码加 robust 训练
   - **复现 ReMix DRO 的关键依赖**

2. **与 RobustVLA 协同的工程方案**：
   - RobustVLA 的多臂 bandit 是 ad-hoc 的，可用 dro library 的 `DROCustom` 替换
   - 让 worst-case 扰动选择有理论保证
   
3. **对 SmolVLA / OpenVLA fine-tuning 的低门槛接入**：
   - SmolVLA 是 PyTorch 模型，可以直接用 dro 的 `DROLoss` 模块
   - 工程实现路径清晰：fork SmolVLA → 在 loss 处加一行 → 启动训练
   
4. **可写进开题报告**：
   - 「采用 Stanford & Columbia 2025 发布的 dro Python 库，实现 worst-case 训练」
   - 体现「使用最新工具 + 站在理论基础上」的研究态度

### 待解问题

1. **大规模（>1M sample）下 Wasserstein DRO 的可行性**：dro 库主要面向中等规模数据；OXE 970k demos 接近上限，需测试
2. **DRO 与 SGD mini-batch 的兼容性**：理论 DRO 是 full-batch 公式，mini-batch 下 robust radius 估计需调整
3. **与 LoRA / PEFT 集成**：只更新 LoRA weights 时，DRO 的对偶变量也要相应调整

### 复现挑战

- **依赖**：cvxpy, scipy, sklearn；与 PyTorch 训练栈整合需写 wrapper
- **eps 选择**：cross-validation 慢，需启发式或 prior 知识
- **GPU 加速**：库目前主要 CPU-based，大规模 VLA 训练需 PyTorch 后端

%% end my-thoughts %%

## 🔗 关联笔记

- **B6 DRO 三件套**：[[2024_ReMix_DRO]], [[2025_DRO_Library]], [[2024_DRO_Theory_Survey]]
- **B5 IL 鲁棒**：[[2020-10_RobustIL-NoisyDemos_Tangkaratt]], [[2022-06_RobustIL-EnvDynamics]]
- **VLA 工程接入**：[[2026_RobustVLA_Guo]], [[2024_OpenVLA]], [[2025_SmolVLA]]

## 📌 Action Items

- [ ] 安装 dro 库，跑 README tutorial
- [ ] 写一个 minimal example：用 DROLoss 替换 BC 的 MSE，在 LIBERO 单任务上验证
- [ ] 测试：DROLoss + LoRA fine-tuning SmolVLA 的训练稳定性
- [ ] 把 dro 库引用纳入开题报告参考文献

%% Import Date: 2026-05-26 %%
