---
create time: 2026-05-27T14:10:00
tags:
  - 曦源项目
  - 版本二plus
  - 主方案
  - FSF
  - FocusedSpatialForcing
  - VGGT
  - FastVGGT
  - Layer24
  - FocusMask
  - 多源融合
status: 工作版v1
audience: 项目组 + 导师 + 师哥 + workshop paper §3 Method
---

# 01 · 主方案 · FocusedSpatialForcing on OpenVLA-OFT

> **方法代号**：**FSF**（FocusedSpatialForcing）
> **一句话**：以冻结 FastVGGT 为 teacher，对 OpenVLA-OFT Layer 24 视觉 token 做余弦相似度对齐；对齐 loss 的 per-token 权重由方向词 GT × SAM2 × DINOv2 CLS attention 三源融合的 focus mask 调制（前景 1.0 / 上下文 0.5 / 背景 0.1）；与 v2 的 InSpire 显式方向词提示叠加。

## § 1 · 方法学完整描述

### 1.1 主架构（继承 v2 + 增量 3 个组件）

```
                输入 RGB (224×224) + language instruction
                            │
        ┌───────────────────┼───────────────────────────────┐
        │                   │                                 │
        ▼                   ▼                                 ▼
   FastVGGT teacher     OpenVLA-OFT 7B + LoRA r=16        SAM2 / DINOv2 (Focus Mask sources)
   (冻结、离线缓存)      ──────────────────────             ──────────────────
        │             SigLIP+DINOv2 patch embeddings              │
        │                   │                                     │
        ▼                   ▼ (256 patch token)                   │
   Aggregator Layer 23   Llama2-7B Layer 1-23                     │
   feature f_3D          (32 层中的 Layer 24 取 hidden state x^V)   │
   (256×1024)            │                                        │
        │                   ▼                                     │
        │             x^V ∈ R^(256×4096)                          │
        │                   │                                     │
        └─────────┬─────────┘                                     │
                  │     ┌─── BN + 2-layer MLP                     │
                  │     │    + 2D Sinusoidal PE                   │
                  │     └─── proj(x^V) ∈ R^(256×1024)             │
                  │                  │                            │
                  │                  ▼                            │
                  └────────► Cosine Similarity ◄──── m_i (focus weight)
                                     │                            │
                                     │                            │
                                     ▼                            │
                                L_FSF (加权余弦)                   │
                                                                  │
        Llama2-7B Layer 24-32 (剩余 LLM 层)                       │
                  │                                               │
       ┌──────────┴───────────────┐                               │
       ▼                          ▼                               │
   Direction Head (7-class)   Action Head (L1 reg)                │
       │                          │                               │
       ▼                          ▼                               │
   L_direction (CE)          L_action (L1)                        │
                                                                  │
                                                                  ▼
                                       Focus Mask Generation
                                       m_i = F1(m_A, m_B, m_C) ∈ [0.1, 1.0]^256
                                       (背景 0.1 / 上下文 0.5 / 前景 1.0 三段量化)

      L_total = L_action + α·L_direction + β·L_FSF
                α=0.1                   β=0.5
```

### 1.2 三层互补的监督栈

| 层位 | 监督类型 | 监督目标 | 继承自 | 数据流 |
|---|---|---|---|---|
| **VLA Layer 24** | FSF：VGGT teacher align + focus mask | dense 3D-aware feature alignment | **v2plus 主创新** | 每个 patch token 一个余弦相似度 |
| **output head (direction)** | direction cross-entropy | 显式方向 grounding | **v2 InSpire** | 每个 timestep 一个 7-class |
| **output head (action)** | L1 regression | 控制信号回归 | v2 baseline | 每个 timestep 7-DoF action chunk |

**关键设计原则**：
1. **FSF 仅在训练时使用，推理时只用 student VLA** → plug-and-play，部署不增加 latency 与依赖
2. **FastVGGT teacher 完全冻结**（Spatial Forcing 原论文消融已证微调反而变差）
3. **Focus mask 注入到 L_FSF 内部 per-token 权重**，**不**注入到 attention 计算或 action loss → 避免破坏原 attention 结构
4. **三个 loss 是加法关系**（非互斥）：α 与 β 经过消融确定，默认 0.1 / 0.5

## § 2 · Loss 精确数学表达

### 2.1 总损失

$$\boxed{\mathcal{L}_{total} = \mathcal{L}_{action} + \alpha \cdot \mathcal{L}_{direction} + \beta \cdot \mathcal{L}_{FSF}}$$

推荐初值与消融区间：

| 参数 | 初值 | 消融区间 | 依据 |
|---|---|---|---|
| $\alpha$ | 0.1 | {0.05, 0.1, 0.2, 0.5} | InSpire 论文未给精确值；v2 上消融已表明 0.1 平衡 |
| $\beta$ | 0.5 | {0.1, 0.3, 0.5, 0.8} | Spatial Forcing 原论文最优 α=0.5；>0.8 过度耦合 |

### 2.2 Action loss（继承 OpenVLA-OFT）

OpenVLA-OFT 采用 L1 回归（非 token 离散化），对 7-DoF action chunk（chunk size H=8 步）：

$$\mathcal{L}_{action} = \frac{1}{H \cdot 7} \sum_{t=1}^{H} \sum_{k=1}^{7} \big| a_{t,k}^{pred} - a_{t,k}^{gt} \big|$$

### 2.3 Direction loss（继承 InSpire）

每步 episode 都有 1 个方向词 GT $d_t^{gt} \in \{\text{right, left, up, down, front, back, grasped}\}$。从 LLM 最后一层取 "Answer:" token 位置后的 hidden state $h_t$，过 7-class linear head：

$$\mathcal{L}_{direction} = -\frac{1}{T} \sum_{t=1}^{T} \log P_{\theta}(d_t^{gt} \mid h_t)$$

### 2.4 FSF loss（v2plus 核心创新公式）

令第 $i$ 个 patch token（共 $N=256$ 个 visual token，14×14 patch grid + …实际 OpenVLA-OFT 配置以 256 token 为基础）：

- Student 投影后特征：$\tilde{x}_i^V = \text{MLP}(\text{BN}(x_i^{V,L24})) + E_i \in \mathbb{R}^{1024}$
- Teacher 特征：$f_i^{3D,L23} \in \mathbb{R}^{1024}$（FastVGGT aggregator penultimate layer，离线预提取）
- Focus mask：$m_i \in [0.1, 1.0]$（多源融合后的三段量化）

$$\boxed{\mathcal{L}_{FSF} = -\frac{1}{\sum_{i=1}^{N} m_i} \sum_{i=1}^{N} m_i \cdot \cos\Big( \tilde{x}_i^V,\; f_i^{3D,L23} \Big)}$$

其中 $\cos(u,v) = \dfrac{u \cdot v}{\|u\|_2 \|v\|_2}$。分母 $\sum m_i$ 归一化保证不同 mask 分布下 loss scale 一致——这是 Spatial Forcing 原论文 $\frac{1}{N}$（均匀）的自然推广。

#### 与 Spatial Forcing 原版的对比

| 维度 | Spatial Forcing 原版 | **v2plus FSF** |
|---|---|---|
| 损失公式 | $-\frac{1}{N} \sum_i \cos(\tilde{x}_i^V, f_i^{3D})$ | $-\frac{1}{\sum m_i} \sum_i m_i \cos(\tilde{x}_i^V, f_i^{3D})$ |
| Per-token 权重 | $m_i \equiv 1$（均匀） | $m_i \in \{0.1, 0.5, 1.0\}$（任务相关） |
| 物理直觉 | 全图都监督 | "focus 在重要位置，背景弱" |
| **来源** | Li et al. 2025 (2510.12276) | 师哥 2026-05-27 提示 + 本项目方法学 |

## § 3 · Focus Mask 多源融合的具体实现

### 3.1 三个来源的设计

#### Source A · 方向词 GT 倒推 2D Gaussian Mask（成本最低，与 v2 工具复用）

**输入**：
- 当前帧的 robot eef 3D pose $\mathbf{p}_{eef}$（来自 LIBERO demo 或真机 leader/follower pose）
- 当前帧的 target object 3D pose $\mathbf{p}_{obj}$（同上）
- 相机内参 $K$ 与外参 $[R | t]$（LIBERO MuJoCo 渲染时已知；真机需标定）

**算法**：

```python
def focus_mask_source_A(p_eef, p_obj, K, R, t, H=224, W=224, patch_grid=14):
    # 1. 3D pose 投影到 2D image
    u_eef, v_eef = project_3d_to_2d(p_eef, K, R, t)
    u_obj, v_obj = project_3d_to_2d(p_obj, K, R, t)
    
    # 2. 224×224 pixel grid 上生成两个 2D 高斯
    SIGMA_EEF = 30   # px
    SIGMA_OBJ = 40   # px (object 略大)
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    G_eef = np.exp(-((x_grid-u_eef)**2 + (y_grid-v_eef)**2) / (2*SIGMA_EEF**2))
    G_obj = np.exp(-((x_grid-u_obj)**2 + (y_grid-v_obj)**2) / (2*SIGMA_OBJ**2))
    m_A_pixel = np.maximum(G_eef, G_obj)  # (224, 224), [0, 1]
    
    # 3. 下采样到 14×14 patch grid (SigLIP/DINOv2 patch=16×16 → 14×14)
    m_A_patch = m_A_pixel.reshape(patch_grid, H//patch_grid, patch_grid, W//patch_grid).mean(axis=(1,3))
    return m_A_patch.flatten()  # (196,) — 注：196 是 14×14；OpenVLA-OFT 实际 256 token 含 cls
```

**优点**：完全免费（pose 已在 demo 中），与 v2 方向词 GT 推导工具 100% 复用。
**缺点**：高斯假设过强，对非凸物体（如长棒）覆盖不全；只覆盖 2 个中心，不能描述物体的全部形状。

#### Source B · SAM2 物体硬 Mask（与 v3 工具协同）

**输入**：当前帧 RGB；Task 中 target object 文本（如 "alphabet soup"，从 LIBERO task name 解析）

**算法**（离线一次性预提取）：

```python
def focus_mask_source_B(image, target_text, sam2_model, grounding_dino):
    # 1. GroundingDINO 找物体 bbox
    bbox = grounding_dino(image, text=target_text)  # (x_min, y_min, x_max, y_max)
    
    # 2. SAM2 用 bbox 提示分割
    M_B_binary = sam2_model.segment(image, bbox_prompt=bbox)  # (224, 224), {0, 1}
    
    # 3. 形态学膨胀 (dilation kernel=5) 包含物体边界周围细节
    M_B_dilated = scipy.ndimage.binary_dilation(M_B_binary, structure=np.ones((5,5)))
    
    # 4. 下采样到 14×14 patch grid (patch 被覆盖比例 >= 0.5 时设为 1)
    m_B_patch = (M_B_dilated.reshape(14, 16, 14, 16).mean(axis=(1,3)) >= 0.5).astype(float)
    return m_B_patch.flatten()  # (196,), {0, 1}
```

**追踪策略**：
- LIBERO 仿真：相机固定，物体集合有限 → 每 episode 仅在 keyframe（首帧 / 抓取瞬间 / 释放瞬间）跑 SAM2，中间帧用光流或 SAM2 video propagation 跨帧传播
- 真机：相机可能轻微振动 → 每 5 帧重新跑 SAM2，中间帧光流

**优点**：边界精确，对各种形状鲁棒。
**缺点**：依赖 SAM2 的开放词汇分割质量；LIBERO 渲染图与 SAM2 训练分布有差距（v3 已验证 mIoU ≈ 0.6-0.75，可接受）。

#### Source C · DINOv2 CLS Attention（免费 saliency）

**输入**：当前帧 RGB；DINOv2 ViT-L/14（OpenVLA-OFT vision encoder 之一，已在 pipeline 中）

**算法**：

```python
def focus_mask_source_C(image, dinov2_model):
    # 1. DINOv2 前向，取最后一层 self-attention 矩阵
    with torch.no_grad():
        attn = dinov2_model.get_last_self_attention(image)
        # attn shape: (1, H_heads, 257, 257) — 256 patch + 1 CLS
    
    # 2. CLS-to-patch attention，多头平均
    attn_cls = attn[0, :, 0, 1:].mean(dim=0)  # (256,)
    
    # 3. min-max 归一化到 [0, 1]
    m_C = (attn_cls - attn_cls.min()) / (attn_cls.max() - attn_cls.min() + 1e-6)
    return m_C.cpu().numpy()  # (256,) — 已在 patch grid 上
```

**优点**：零额外计算（DINOv2 本就在 OpenVLA-OFT 前向中），无需额外标注。
**缺点**：DINOv2 saliency 对显著物体好，但对 task-relevant 物体不一定（可能突出装饰物或干扰物）。

### 3.2 三源融合策略（4 种候选，主路 F1）

| 策略 | 公式 | 优点 | 缺点 |
|---|---|---|---|
| **F1 加权平均（主路推荐）** | $m_i = \text{clip}(0.3 m_i^A + 0.4 m_i^B + 0.3 m_i^C, 0.1, 1.0)$ → 三段量化 | 平滑、稳定 | 权重需调 |
| F2 Max 融合 | $m_i = \max(m_i^A, m_i^B, m_i^C)$ | 召回高（任一源说重要就重要） | 易过拟合显著区域 |
| F3 学习 gating | $m_i = \sigma(g_\theta([m_i^A, m_i^B, m_i^C]))$（小 MLP 学权重） | 数据驱动 | 引入额外可训参数，过拟合风险 |
| F4 阶梯量化（3-level，直接对应师哥语言） | $m_i \in \{0.1, 0.5, 1.0\}$ 按 max 分级 | 与师哥"前景1/上下文0.5/背景0.1"直接对应 | 边界硬，可能不平滑 |

**主路 F1 详细**（A3 ablation 验证此为最优）：

```python
def fuse_focus_masks_F1(m_A, m_B, m_C, w_A=0.3, w_B=0.4, w_C=0.3,
                        thr_fg=0.7, thr_ctx=0.3, fg=1.0, ctx=0.5, bg=0.1):
    m_raw = w_A * m_A + w_B * m_B + w_C * m_C
    
    # 三段量化
    m = np.where(m_raw >= thr_fg, fg,
        np.where(m_raw >= thr_ctx, ctx, bg))
    
    # 可选高斯平滑（kernel=3）让边界过渡平滑（H5 ablation 测试）
    # m = scipy.ndimage.gaussian_filter(m.reshape(14, 14), sigma=0.5).flatten()
    return m
```

**关键超参与消融区间**：

| 参数 | 初值 | 消融区间 | 依据 |
|---|---|---|---|
| $w_A, w_B, w_C$ | 0.3, 0.4, 0.3 | $\{0.2,0.3,0.4\}^3$（保持 $\sum=1$） | SAM2 mask 精度最高，故 $w_B$ 略大 |
| $\text{thr}_{fg}$ | 0.7 | {0.5, 0.6, 0.7, 0.8} | 阈值越高，前景越严 |
| $\text{thr}_{ctx}$ | 0.3 | {0.2, 0.3, 0.4} | 上下文阈值 |
| 前景 weight | 1.0 | 固定 | Spatial Forcing 默认 |
| 上下文 weight | 0.5 | 固定（与 fg/bg 形成 3 段阶梯） | 师哥语言对应 |
| 背景 weight | 0.1 | {0.0, 0.05, 0.1, 0.3} | 师哥强调"弱但非零"；0.0 等价于硬 mask |

### 3.3 Focus Mask Ablation 设计（H5 验证）

| Ablation 编号 | mask 来源 | 目的 |
|---|---|---|
| **M-None** | $m_i \equiv 1$（=SF 原版） | 对照基线，验证 focus 是否有用 |
| M-A | 仅 Source A（方向词 GT） | 验证最便宜的单源是否足够 |
| M-B | 仅 Source B（SAM2） | 验证精确分割单源是否足够 |
| M-C | 仅 Source C（DINOv2） | 验证免费 saliency 单源是否足够 |
| M-AB | A + B 加权融合 | 测两源（最便宜 + 最精确）够不够 |
| **M-ABC（主路）** | A + B + C F1 融合 | 主提交 |
| M-Random | 随机 mask（同前景占比） | 验证"focus location"重要而非"mask diversity" |

预期 H5 结论：M-ABC > M-A ≈ M-B ≈ M-C > M-None > M-Random。

## § 4 · VGGT Teacher 配置与离线缓存

### 4.1 从 FastVGGT 取哪一层 feature

FastVGGT 继承 VGGT 的 24 层 aggregator transformer。我们的层位选择：

| 候选 | 维度 | 含义 | 推荐度 |
|---|---|---|---|
| Aggregator Layer 12（middle） | 1024 | 中层 spatial-aware feature | ⭐⭐ |
| **Aggregator Layer 23（penultimate，主选）** | **1024** | **最终空间表征，深度/点云信息丰富** | **⭐⭐⭐⭐⭐** |
| DPT depth head logits | per-pixel | 显式 depth | ⭐⭐ |
| DPT point map | per-pixel 3D | 显式 3D 点云 | ⭐⭐⭐ |

**主选 Layer 23**：依据 Spatial Forcing 原论文（§3.2）："we extract VGGT feature $f^{3D}$ from the penultimate transformer layer of VGGT aggregator"。

**Ablation A1**：跑 {Layer 12, Layer 18, Layer 23} 三种 teacher 层，主路报 Layer 23。

### 4.2 维度匹配

| 端点 | 维度 |
|---|---|
| FastVGGT Aggregator Layer 23 输出 | $N=256$ patch × 1024 dim |
| OpenVLA-OFT Llama2-7B Layer 24 hidden | $N=256$ visual token × **4096** dim |
| 投影 MLP | 4096 → 2048 → **1024**（与 FastVGGT 对齐） |

**投影模块** `FSFProjector`（~30 行代码）：

```python
class FSFProjector(nn.Module):
    def __init__(self, student_dim=4096, hidden_dim=2048, teacher_dim=1024, n_patch=256):
        super().__init__()
        self.bn = nn.BatchNorm1d(student_dim)
        self.mlp = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim),
        )
        # 2D sinusoidal positional encoding (固定，非可学习)
        self.register_buffer('pos_enc', build_2d_sinusoidal_pe(n_patch, teacher_dim))
    
    def forward(self, x_student):  # x_student: (B, N=256, d_S=4096)
        B, N, _ = x_student.shape
        x = self.bn(x_student.reshape(B*N, -1)).reshape(B, N, -1)
        x = self.mlp(x)             # (B, N, d_T=1024)
        x = x + self.pos_enc[None]  # 广播加 PE
        return x


def fsf_loss(student_feat, teacher_feat, focus_mask):
    cos = F.cosine_similarity(student_feat, teacher_feat, dim=-1)  # (B, N)
    weighted = (focus_mask * cos).sum(dim=-1) / focus_mask.sum(dim=-1).clamp(min=1e-6)
    return -weighted.mean()
```

### 4.3 离线缓存策略（**关键工程决策**）

#### 为什么必须缓存

- LIBERO 26,000 demo × 平均 50 frame = **1.3M frame**
- 每 frame FastVGGT 前向 ≈ 30ms，共 **11 GPU-h**（一次性）
- 训练每 epoch 都跑一遍 VGGT → 1100 GPU-h（× 100 epoch），**不可承受**
- 离线缓存：一次性 11 GPU-h → 训练时 IO 而非 compute

#### 缓存格式：WebDataset (tar shards)

| 选项 | 优点 | 缺点 | 推荐度 |
|---|---|---|---|
| HDF5（单大文件） | 随机访问、压缩好 | 多进程并发限制 | ⭐⭐⭐ |
| **WebDataset (tar shards)（主选）** | **DataLoader 友好、shuffle 快、并发好** | 学习曲线 | **⭐⭐⭐⭐⭐** |
| 每 demo 一个 .npy/.pt | 简单 | 文件多、IO 低效 | ⭐⭐⭐ |

**存储结构**：

```
vggt_cache/
├── libero_spatial/
│   ├── shard_0000.tar    # 1000 episode 的 feature
│   ├── shard_0001.tar
│   └── ...
├── libero_object/
├── libero_goal/
└── libero_long/
```

每个 tar 包内：

```
episode_0001/
  frame_0000.feat.npy    # shape (256, 1024) bf16
  frame_0001.feat.npy
  ...
```

#### 磁盘空间估算与优化

- 单 frame feature: $256 \times 1024 \times 2$ bytes (bf16) = **0.5 MB**
- 总 frame: 1.3M
- **总占用：650 GB**（bf16）

**优化策略**（应对学校存储配额）：
1. **fp8 量化** → 325 GB（轻微精度损失，需 P0 验证）
2. **仅缓存 LIBERO-Spatial 1 suite** → 162 GB（先单 suite 验证 FSF）
3. **学校大存储节点** / **阿里云 OSS 冷存储**（~¥30/月 / 100 GB） → 1 年 ¥360 < ¥400 实验耗材预算

**Cache 失效条件**：
- VGGT teacher 换版本 → 全部重新缓存
- LIBERO demo 重生成 → 受影响 suite 重新缓存

## § 5 · 3 个变体设计（紧约束版）

### 5.1 变体配置

由于 2 人 + ¥10K 紧约束，从 v2plus 设计原稿的 4 变体降级为 3 变体（去掉单独 +FSF，用 A3 ablation 中 M-None 替代）：

| 变体 | Prompt | Loss | Backbone | 训练数据 |
|---|---|---|---|---|
| **V1: vanilla** | 原 LIBERO instruction | $\mathcal{L}_{action}$ | OpenVLA-OFT + LoRA r=16 | LIBERO 4 suite |
| **V2: +InSpire** | 前置方向词 prompt | $\mathcal{L}_{action} + 0.1 \mathcal{L}_{direction}$ | 同上 + 7-class direction head | 同上 + 方向词 GT |
| **V3: +InSpire+FSF（主创新）** | 同 V2 | $\mathcal{L}_{action} + 0.1 \mathcal{L}_{direction} + 0.5 \mathcal{L}_{FSF}$ | 同上 + FSF projector | 同上 + 缓存 VGGT feature + focus mask |

**变体间关系**：
- V1 = v2 的 vanilla baseline（向后兼容）
- V2 = v2 的 +InSpire 变体（向后兼容，对照 H1/H3）
- V3 = v2plus 主创新（H4 验证）

**V2 与 V3 都使用方向词 prompt** → V3 的 FSF 增益是在 InSpire 基础上的"叠加增益"，对应 H2 的形式化。

### 5.2 推理协议（FSF 是训练时 only）

| 变体 | 推理时 Direction 输出 | 推理时 FSF | Latency 增量 |
|---|---|---|---|
| V1 | ❌ | ❌ | 0 |
| V2 | 是（output 1 token） | ❌ | +5-10ms |
| V3 | 是 | ❌（FSF 是训练时 only） | +5-10ms |

**关键工程优势**：FSF 仅在训练时使用，推理时只用 student VLA，**不需要 VGGT 也不需要 mask** → plug-and-play 部署友好。

### 5.3 LoRA 训练配置

- **target_modules**：q_proj, k_proj, v_proj, o_proj
- **r=16, alpha=32, dropout=0.05**
- **显存预算**：< 40 GB（单卡 A100）
- **训练数据**：LIBERO-Spatial / Object / Goal / Long 4 suites（共 ~26000 trajectory）
- **训练时间**：单卡 8-12h / variant / seed

## § 6 · 5 个核心假设的形式化

### 6.1 H1（仿真鲁棒性增益）

**命题**：令 $M \in \{V1, V2, V3\}$，$d \in \{\text{layout, viewpoint, state, instruction, light, texture, noise}\}$（LIBERO-Plus 7 维）。定义 **perturbation drop**：

$$\Delta_{M,d} = \text{SR}_{\text{clean}}(M) - \text{SR}_d(M)$$

**可证伪声明**：在至少 2 个扰动维度上，V3 (+InSpire+FSF) 的 perturbation drop **显著小于** V1 (vanilla)：

$$\Delta_{V3, d} < \Delta_{V1, d} - \tau, \quad \tau = 5\text{pp}, \quad p < 0.05 \quad \text{(配对 t-test, Bonferroni)}$$

**预期显著维度**：基于先验机制分析，最可能显著的维度是 **viewpoint** 与 **state**（因 3D 表征对齐对几何变化敏感）；最不可能显著的维度是 **instruction**（LIBERO-Plus 论文已报该维度对所有 VLA 都不显著）。

**预期 effect size**：
- v2 +InSpire 在 viewpoint 上 +3-5pp（保守估计）
- v2plus +FSF 叠加 +3-5pp（SF 原论文报 long horizon +3.8pp）
- 合计 V3 在 viewpoint 上 +6-10pp vs V1

### 6.2 H2（叠加效应）

**命题**：V3 (+InSpire+FSF) 的 7 维平均 SR 显著优于 V2 (+InSpire) 与单独 +FSF：

$$\text{SR}_{V3} > \max(\text{SR}_{V2}, \text{SR}_{+\text{FSF only}}) + 1.5\text{pp}$$

由于紧约束没有单独 +FSF 变体，**H2 退化为半实证假设**：用 A3 中 M-None（=SF 原版均匀 FSF）作为 +FSF only 代理。

**Fallback**：如果 H2 不成立但 V3 > V2，仍然有意义（说明 FSF 在 InSpire 基础上有增量）。

### 6.3 H3（Failure Predictor，继承 v2）

**命题**：用 V2/V3 的方向词预测时序训练 LightGBM 二分类器，预测最终成败 AUC ≥ 0.65：

$$\text{AUC}\big(f(\bar{\text{acc}}, \text{var}(\text{acc}), \bar{\text{conf}}, \text{flips}, ...) \to y\big) \ge 0.65 \quad \text{(5-fold CV)}$$

**理论辩护**：InSpire 方向词预测的**离散切换**（discrete flip）比连续 entropy 更易锁定 critical moment——对应 Averaging Trap 论文（arXiv:2603.18342）的核心 finding。

**预期**：
- v2 估计 0.65-0.75
- v2plus 由于 FSF 训练让 hidden state 更 3D-aware，可能进一步提升

### 6.4 H4（真机鲁棒性增益，v2plus 真机假设）

**命题**：V3 在真机 3 任务 × 3 扰动维度 × 30 episode 评测中，相对 V1 的扰动 SR drop 平均小 ≥ 5pp（p < 0.10，真机 sample 少，p 阈值放宽）：

$$\overline{\Delta_{V3}^{real}} < \overline{\Delta_{V1}^{real}} - 5\text{pp}, \quad p < 0.10$$

**真机扰动 cell**：3 任务 × 3 扰动 = 9 cell；每 cell 30 episode；每 cell SR std ≈ 9-12pp；5pp 差异需要 30 episode × 3 任务 ≈ power 0.6（适中）。

### 6.5 H5（多源 focus 优于单源，v2plus 核心创新假设）

**命题**：在 A3 ablation 中，M-ABC（三源融合）的平均 SR 高于任一单源（M-A, M-B, M-C）+ 1pp：

$$\text{SR}_{M\text{-}ABC} > \max(\text{SR}_{M\text{-}A}, \text{SR}_{M\text{-}B}, \text{SR}_{M\text{-}C}) + 1\text{pp}$$

同时 **三源融合显著优于 M-None（SF 原版均匀监督）** 与 M-Random（随机 mask）：

$$\text{SR}_{M\text{-}ABC} > \text{SR}_{M\text{-}None} + 1.5\text{pp}, \quad p < 0.05$$

$$\text{SR}_{M\text{-}ABC} > \text{SR}_{M\text{-}Random} + 3\text{pp}, \quad p < 0.05$$

**意义**：H5 是 v2plus 区别于 SF 原论文最直接的"方法学贡献"实证。如果 H5 不成立但 M-ABC > M-None 仍 holds，说明 focus 是有用的、但融合方式无差异——也是有价值的 finding。

## § 7 · 主评测设计

### 7.1 LIBERO-Plus 7 维主评测（仿真）

**配置**：LIBERO-Plus 7 维 × 3 变体 × 3 seed × 300 instance/cell（紧约束下 v2 的 500 降到 300）

| 项 | 数值 |
|---|---|
| Cell 数 | 7 × 3 = **21 cell** |
| 每 cell episode | 300 × 3 = 900 episode |
| 总 episode | 21 × 900 = **18,900** |
| 平均每 episode 推理时长 | 30s（OpenVLA-OFT 报告） |
| **总 GPU-h** | 18,900 × 30s / 3600 ≈ **158 GPU-h** |

### 7.2 真机评测（v2plus 新增轴）

**配置**（用户确认中等 scope）：3 任务 × 50 demo 训练 × 3 扰动维度 × 30 episode/cell

| 任务（建议） | 难度 | 物理意义 |
|---|---|---|
| **T1: Pick-place（block to bowl）** | 简单 | 测基础抓放 |
| **T2: Push（block to target line）** | 中等 | 测推动力控制 |
| **T3: Pick-place（mug to plate, occluded）** | 难 | 测部分遮挡下视觉鲁棒性 |

**3 个扰动维度**（与 LIBERO-Plus 部分对应）：
- D1: **光照变化**（实验室灯 / 暗光 / 侧射强光）
- D2: **背景纹理**（白桌 / 木桌布 / 杂乱）
- D3: **初始位置**（5×5 cm 网格，9 个起点抽 3 个）

**评测计数**：
- 训练数据：3 任务 × 50 demo = 150 demo（用 LIBERO 训练好的 LoRA 在真机上少量 fine-tune）
- 评测：3 任务 × 3 扰动 × 30 episode/cell + 3 任务 × clean × 30 episode = **360 episode**
- 单 episode 真机执行 30-60s，含人工 reset
- **总人工时间**：360 × 45s = 4.5 小时（纯执行）+ reset 4.5 小时 + 标注 2 小时 ≈ **11-12 小时人工 / 1.5 天**
- GPU：本地 RTX 4090 推理（如实验室有），**不进云 GPU 账户**

### 7.3 Ablation 实验

| Ablation | 配置 | Cell 数 | GPU-h | 验证目标 |
|---|---|---|---|---|
| **A1: VGGT layer 选择** | VGGT Layer {12, 18, 23} | 3 × 3 维子集 × 3 seed × 200 inst | 15 h | Layer 23 最优 |
| **A2: SF 监督层位** | OpenVLA-OFT Layer {8, 16, 24, 32} | 4 × 3 维子集 × 3 seed × 200 inst | 20 h | Layer 24 最优 |
| **A3: Focus mask 来源（H5 核心）** | M-None/A/B/C/AB/ABC/Random | 7 × 3 维子集 × 3 seed × 200 inst | 35 h | 多源融合优于单源 |
| **A4: β 扫描** | β ∈ {0.1, 0.3, 0.5, 0.8} | 4 × 3 维子集 × 3 seed × 200 inst | 20 h | β=0.5 最优 |
| **A5: 背景 weight 扫描** | bg_w ∈ {0, 0.05, 0.1, 0.3} | 4 × 3 维子集 × 3 seed × 200 inst | 20 h | 0.1 最优 |
| **A6: 视觉利用 ablation（继承 v2）** | token dropout 3 强度 × patch mask 2 模式 × attn vis | 同 v2 | 25 h | 机制层证据 |
| **A7: 融合策略** | F1/F2/F3/F4 | 4 × 3 维子集 × 3 seed × 200 inst | 20 h | 选最佳融合策略 |
| **A8: 方向词粒度** | 6 vs 7 vs 12 方向 | 3 × 3 维子集 × 3 seed × 200 inst | 15 h | 7 类适中 |
| **总 Ablation GPU-h** | | | **170 h** | |

**Ablation 优先级**（紧约束下可砍）：
- **必须做（P2）**：A1, A2, A3（核心 3 个，55 h）
- **可降级（P3）**：A4, A5 用粗扫（粗 2 个值，10 h）
- **可放弃（stretch）**：A7, A8（35 h，进 stretch goal）

## § 8 · 6 个执行 Phase（M0-M12）

详见 [[03_第一步两周任务清单]] 与 [[05_风险与缓解]]。

### P0 · 环境与真机准备（M0，2026-05-末 → 06-初；2 周）

- D1-D3: OpenVLA-OFT + LoRA 环境装通
- D4-D5: FastVGGT 仓库装通（github.com/OpenHelix-Team/FastVGGT），在 5 张 LIBERO 渲染图上前向测试
- D6-D7: GroundedSAM + SAM2 装通
- D8-D9: 真机平台（实验室已有）测试，跑 1 个 demo 录制流程
- D10-D14: 方向词 GT 标注脚本 + Focus mask 3 源 pipeline 实现

**N0 末决策**：4 仓库装通 + FastVGGT 在 LIBERO 上 mIoU ≥ 0.5 → 主路；否则 fallback VGGT 原版

### P1 · 复现 + Sanity（M1-M3，06-08）

- M1: V1/V2 LoRA 训练 + FastVGGT 离线缓存（11 GPU-h）
- M2: Focus mask 3 源在 100 帧上目视验证 + FSF projector 代码 + V3 sanity check 训练
- M3: V3 完整 LoRA 训练（4 suite × 3 seed）+ clean LIBERO 评测 V1/V2/V3 + 真机数据采集（3 任务 × 50 demo）

**N1 末决策**：V3 clean SR ≥ V1 - 3pp & VGGT 缓存完整 → Go P2；退化 > 10pp → 切 v2 only

### P2 · 仿真主评测 + 真机标注（M4-M6，09-11）

- M4: 3 变体 × 3 维（viewpoint/state/texture）LIBERO-Plus 评测（80 GPU-h）+ 真机标注
- M5: 剩余 4 维评测（60 GPU-h）+ A1/A2 ablation
- M6: A3 mask 来源 ablation（H5 核心）+ 真机 V1 训练（150 demo 少量 fine-tune）

**N2 末决策**：H1 ≥ 2 维度显著 & H5 M-ABC > M-None → Go P3；完全无增益 → negative finding workshop

### P3 · 机制分析 + 真机评测（M7-M9，12-02）

- M7: A6 视觉利用 ablation + A4/A5 ablation + 真机 V2/V3 训练
- M8: 真机 360 episode 评测 + Attention 可视化对比 V1/V3
- M9: A7/A8 stretch ablation + 视觉利用 ablation 数据汇总 + 论文图制作

**N3 末决策**：H4 真机增益 ≥ 5pp & 视觉 ablation 显示 V3 对前景 patch 依赖更高 → Go P4；退化 → 真机降为 case study

### P4 · Failure Predictor + 论文写作起步（M10-M11，03-04）

- M10: 收集 2k 配对 episode + 训练 LightGBM + 与 attention entropy baseline 对比
- M11: 论文 §1 Intro + §2 Related + §3 Method 初稿（≈2500 字）+ 实验图表第一版

**N4 末决策**：AUC ≥ 0.65 & 论文数据齐全 → Go P5；不齐 → 转 ICLR/CCC

### P5 · 写作冲刺 + 投稿（M12，05）

- W1-W2: §4 Experiment + §5 Discussion
- W3: 完整论文初稿（5-6 页）+ 内部 review
- W4: 投稿（CoRL 2027 workshop 或 ICLR 2027 Robot Learning workshop）

## § 9 · 与 v2 / v3 的关系总结

| 资源 | v2plus 中的处理 |
|---|---|
| v2 的 OpenVLA-OFT + LoRA 训练 pipeline | **100% 复用** |
| v2 的 方向词 GT 自动标注脚本 | **100% 复用**（同时作为 Focus mask Source A） |
| v2 的 LIBERO-Plus 7 维评测脚本 | **100% 复用**（3 变体共享） |
| v2 的 7-class direction cls head | **100% 复用**（V2/V3 共用） |
| v2 的视觉利用 ablation 工具 | **100% 复用**（全 3 变体） |
| v2 的 LightGBM failure predictor | **100% 复用**（H3 继承） |
| v2plus 新增：FastVGGT 离线缓存 pipeline | **新增**（11 GPU-h 一次性） |
| v2plus 新增：FSF projector 模块（~30 行代码） | **新增** |
| v2plus 新增：Focus mask 3 源融合 | **部分新**（Source A 复用 v2 工具） |
| v2plus 新增：真机数据采集与标注 | **新增**（实验室已有真机平台） |
| v3 的 GroundedSAM/SAM2 工具链 | **部分复用**（作为 Focus mask Source B） |
| v3 的 attention-mask IoU 计算 | **不复用**（v2plus 主路不依赖此） |

## § 10 · 主方案的预期 paper 章节映射

详见 [[08_最终方案_FSF详细技术报告]] § 8.2。简表如下：

| Paper § | 字数 | 对应本文档章节 |
|---|---|---|
| §1 Intro | 500 | § 1（架构） + § 6（H 假设） |
| §2 Related | 700 | [[06_关联资料索引]] |
| §3 Method | 1000 | § 1 + § 2（loss）+ § 3（mask）+ § 4（VGGT 缓存） |
| §4 Experiment | 1500 | § 7（评测）+ Appendix（ablation 详细） |
| §5 Discussion | 500 | [[05_风险与缓解]] + [[02_备选方案]] |

%% Updated: 2026-05-27 %%
