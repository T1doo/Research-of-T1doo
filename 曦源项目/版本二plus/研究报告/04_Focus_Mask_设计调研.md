---
title: Focus Mask 多源融合设计调研
project: T1doo 版本二plus 立项调研
report_id: 04
author: T1doo 团队
date: 2026-05-27
status: draft
tags:
  - VLA
  - FocusMask
  - SAM2
  - DINOv2
  - Saliency
  - 立项调研
related:
  - "[[01_Spatial_Forcing_深度精读]]"
  - "[[02_VGGT_及后续工作综述]]"
  - "[[03_FocusVLA_中_VGGT_用法剖析]]"
  - "[[06_3D-aware_VLA_相关工作综述]]"
---

# 报告 04：Focus Mask 多源融合设计调研

> **导航**：本报告是 v2plus 核心增量"Focus Mask"模块的设计调研。它回答四个问题：
>
> 1. 为什么要在 [[01_Spatial_Forcing_深度精读|SF]] 的对齐 loss 上加 Focus Mask？（§1）
> 2. Focus Mask 该从哪几个源来构造？（§2）
> 3. 多源信号该如何融合？（§3）
> 4. 三段量化（前景 1.0、上下文 0.5、背景 0.1）的依据是什么？（§6）
>
> 本报告与 [[03_FocusVLA_中_VGGT_用法剖析]] 一同构成 v2plus 立项中"增量贡献"部分的正名材料。

## TL;DR

- **动机**：师哥提示"focus 在重要位置，不相关背景监督可以更弱点"——v2plus 在 SF 的 align loss 上加入空间 mask，让对齐监督更专注于任务相关区域。
- **三个候选源**：
  - **Source A（方向词 GT 倒推 2D Gaussian）**：从 v2 版本沿用的工具，复用度高、零额外推理开销；提供**任务相关性**先验。
  - **Source B（SAM2 物体硬 mask）**：与 v3 版本工具协同，提供**物体级精确边界**；提供**物体形状**先验。
  - **Source C（DINOv2 CLS attention）**：零额外计算（DINOv2 本就在 backbone 中），提供**模型自身的显著性**先验。
- **推荐主路融合策略 F1（加权平均）**：$m = \mathrm{clip}(0.3A + 0.4B + 0.3C, 0.1, 1.0)$，然后做三段量化到 $\{1.0, 0.5, 0.1\}$。
- **三段量化的依据**：直接对应师哥语言"重要位置 = 强监督 / 不相关背景 = 弱监督"；硬 mask（bg=0）过于激进会丢梯度，软 mask（无量化）则失去先验。
- **Ablation 设计**：M-None / M-A / M-B / M-C / M-AB / M-ABC / M-Random 7 组，预期 M-ABC > M-A ≈ M-B ≈ M-C > M-None > M-Random。
- **与相关工作的差异**：AttentionVoxel / Gaze-Regularized VLA / AutoFocus-IL 等都做过 saliency-guided VLA，但**v2plus 是第一个把多源 saliency 用于调制 VGGT-VLA 对齐 loss** 的工作。

---

## §1 Focus Mask 的理论动机

### 1.1 师哥的原始提示

> "focus 在重要位置，不相关背景监督可以更弱点"

这句话有两层指向：

1. **正向指向**：任务相关的前景区域（被操作物体、目标位置、机械臂末端等）应当获得**强监督**——VGGT-VLA 对齐 loss 在这些位置应当权重大。
2. **反向指向**：任务无关的背景区域（桌面纹理、墙壁、其他干扰物）应当获得**弱监督**——对齐 loss 在这些位置应当权重小。

这与 SF 原版对齐 loss 的差异是：SF 对所有 visual token 一视同仁地对齐（per-token cosine, uniform weight），而 v2plus 在权重上引入空间先验。

### 1.2 信息论视角：高熵 vs 低熵 attention

考虑 VLA Layer 24 输出的 visual token 集合 $\{z_1, \ldots, z_N\}$。设每个 token 对应一个 patch，其重要性可以用某种 attention 分布 $\{p_1, \ldots, p_N\}$ 表征（$\sum p_i = 1$）。

- **高熵 attention**：$H(p) = -\sum p_i \log p_i$ 接近 $\log N$（均匀分布）。此时 token 重要性平均，信息分散——这对应"全场景对齐"，监督信号被稀释。
- **低熵 attention**：$H(p)$ 远小于 $\log N$（集中分布）。此时 token 重要性集中在少数关键区域，信息集中——这对应"focus 对齐"，监督信号被增强。

**Focus Mask 的本质**：用一个**外部先验** $m_i$ 把对齐 loss 的隐式权重分布从高熵改造为低熵：

$$
\mathcal{L}_{\mathrm{align}}^{\mathrm{v2plus}} = \frac{\sum_i m_i \cdot (1 - \cos(z_i^S, z_i^T))}{\sum_i m_i}
$$

其中分母 $\sum_i m_i$ 是归一化项，保证 mask 不改变 loss 的整体尺度。

### 1.3 梯度流视角：背景 token 的梯度贡献应当被降权

考虑 align loss 对某个参数 $\theta$ 的梯度：

$$
\frac{\partial \mathcal{L}_{\mathrm{align}}}{\partial \theta} = \frac{1}{Z} \sum_i m_i \cdot \frac{\partial (1 - \cos(z_i^S, z_i^T))}{\partial \theta}
$$

其中 $Z = \sum_i m_i$。

- 若 $m_i = 1$ 对所有 $i$（SF 原版）：每个 token 的梯度贡献相同。**问题**：在 LIBERO 任务中，729 个 SigLIP patch 中可能只有 20-50 个对应"被操作物体"和"目标位置"，其余 670+ 个是背景。背景的梯度信号占据 92% 的权重，反而稀释了真正有用的前景信号。
- 若 $m_i \in \{1.0, 0.5, 0.1\}$（v2plus）：背景的梯度贡献降到原来的 1/10，前景保持满权重，**有效信号/无效信号比从 1:9 变为 10:9**——信噪比提升约 10 倍。

### 1.4 与 SF 原版的"一视同仁"对比

SF 论文 §4 中默默假设了"所有 visual token 都应当对齐 VGGT"——这一假设在 LIBERO 这种**前景物体小、背景占比大**的场景下并不一定最优。事实上，SF 论文 Figure 7 的可视化表明：

- 在桌面任务中，机器人末端 + 操作物体仅占视野的 10-15%。
- 在长程任务（LIBERO-Long）中，目标物体可能更小（5-8%）。

v2plus 的 Focus Mask 正是为这种"前景小、背景大"的场景设计——这也解释了为什么 v2plus 在 LIBERO-Long 上的预期增益最大（详见 §5 ablation 设计）。

---

## §2 三个候选来源的详细比较

### 2.1 Source A：方向词 GT 倒推 2D Gaussian

#### 2.1.1 算法描述

v2 版本中团队已经积累了"方向词"标注工具：对每条 demo trajectory，从 instruction 中解析方向词（如"left", "right", "front of"），并结合相机内外参把方向词关联到 2D 图像坐标。

**v2plus 的 Source A 用法**：

1. 解析 instruction 中的目标物体名（如"red block", "blue mug"）。
2. 通过 demo 第 0 帧的物体 GT 位姿（LIBERO 自带）+ 相机投影矩阵，把目标物体的 3D 中心投影到 2D 图像坐标 $(u, v)$。
3. 在 $(u, v)$ 周围构造一个 2D Gaussian：
$$
A(x, y) = \exp\left( -\frac{(x-u)^2 + (y-v)^2}{2\sigma^2} \right)
$$
其中 $\sigma$ 为可调超参（建议 $\sigma = 50$ 像素，对应大约 1/4 视野）。
4. 把 Gaussian 下采样到 patch grid（SigLIP 是 27×27）得到 $A \in [0, 1]^{27 \times 27}$。

#### 2.1.2 优劣分析

**优点**：
- **与 v2 工具复用度高**：方向词 + GT 倒推工具在 v2 版本已经成熟，几乎零开发成本。
- **零额外推理开销**：方向词解析只发生在数据预处理阶段，与推理无关。
- **提供任务相关性**：Gaussian 中心在目标物体上，强烈编码"指令-视觉"的语义对齐。

**缺点**：
- **依赖 GT 位姿**：在仿真（LIBERO）可用，**在真机不可用**（除非有 motion capture 或人工标注）。
- **Gaussian 形状粗糙**：对实际物体形状不敏感，对非凸物体（如倒置的杯子）会有边界泄露。
- **方向词覆盖不全**：v2 工具主要处理空间方向词，对"the third object from left"这类复杂指代支持有限。

#### 2.1.3 与 v2 工具的复用度

| 模块 | v2 已有 | v2plus 需新增 |
|------|---------|---------------|
| 方向词解析 | 完全可复用 | 无 |
| GT 物体位姿读取 | LIBERO 接口已对接 | 无 |
| 相机投影到 2D | v2 已实现 | 无 |
| Gaussian 下采样到 patch grid | v2 未做 | 约 20 行代码 |

#### 2.1.4 实施伪代码

```python
import numpy as np

def build_source_A(instruction, demo_init_state, camera_params,
                   patch_grid=(27, 27), sigma=50):
    """
    构造 Source A: 方向词 GT 倒推 2D Gaussian.

    Args:
        instruction: str, demo 指令
        demo_init_state: dict, LIBERO 初始状态（含物体 6DoF 位姿）
        camera_params: dict, 相机内外参
        patch_grid: tuple, patch 网格分辨率
        sigma: float, Gaussian 标准差（像素）

    Returns:
        A: ndarray (27, 27), 取值 [0, 1]
    """
    # 1. 解析目标物体
    target_obj = parse_target_object(instruction)  # v2 工具

    # 2. 读取 GT 位姿
    obj_pose = demo_init_state[target_obj]  # (4, 4) 齐次变换

    # 3. 投影到 2D
    obj_center_3d = obj_pose[:3, 3]  # 物体中心
    u, v = project_3d_to_2d(obj_center_3d, camera_params)

    # 4. 构造 Gaussian (在原始分辨率上)
    H, W = camera_params['image_size']
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    gaussian = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))

    # 5. 下采样到 patch grid
    A = downsample_to_grid(gaussian, patch_grid)  # 简单平均池化
    A = A / (A.max() + 1e-8)  # 归一化到 [0, 1]
    return A
```

### 2.2 Source B：SAM2 物体硬 Mask

#### 2.2.1 算法描述

利用 SAM2 + GroundingDINO 组合做"文本提示分割"：

1. 解析 instruction，提取目标物体名词（如"red block"）。
2. 用 GroundingDINO 在第 0 帧上做 open-vocabulary 检测，得到目标物体 bounding box。
3. 用 SAM2 以 bbox 为 prompt 做分割，得到二值 mask $B \in \{0, 1\}^{H \times W}$。
4. 下采样到 patch grid 得到 $B \in [0, 1]^{27 \times 27}$（patch 内 mask 比例）。

#### 2.2.2 优劣分析

**优点**：
- **物体级精确**：SAM2 的 mask 紧贴物体边界，比 Gaussian 精细得多。
- **与 v3 工具协同**：v3 版本已经实现 SAM2 + GroundingDINO 管线，可直接复用。
- **真机可用**：不依赖仿真 GT，只需 RGB 图像即可。

**缺点**：
- **计算开销大**：SAM2 单帧约 50ms，GroundingDINO 约 30ms，合计 80ms/帧。对于 LIBERO（130k+ frames）总耗时约 3 小时（A100）。
- **GroundingDINO 误检**：对小物体或被遮挡物体可能漏检。
- **硬 mask 边界过于尖锐**：可能与 patch grid 对齐出现走样。

#### 2.2.3 与 v3 工具的协同

v3 版本中 SAM2 + GroundingDINO 工具已经搭建，v2plus 直接调用 v3 的封装函数，仅需替换输入数据集即可。

#### 2.2.4 实施伪代码

```python
def build_source_B(instruction, image, grounding_dino, sam2,
                   patch_grid=(27, 27)):
    """
    构造 Source B: SAM2 物体硬 mask.
    """
    # 1. 解析目标
    target_phrase = parse_target_phrase(instruction)

    # 2. GroundingDINO 检测
    boxes, scores = grounding_dino.detect(image, text=target_phrase,
                                          box_threshold=0.35,
                                          text_threshold=0.25)
    if len(boxes) == 0:
        # 退化：返回均匀 mask
        return np.ones(patch_grid) * 0.5

    # 3. SAM2 分割
    best_box = boxes[scores.argmax()]
    mask = sam2.segment(image, box=best_box)  # (H, W) 0/1

    # 4. 下采样到 patch grid
    B = downsample_to_grid(mask.astype(np.float32), patch_grid)
    # B 值表示 patch 内被覆盖的比例
    return B
```

### 2.3 Source C：DINOv2 CLS Attention

#### 2.3.1 算法描述

DINOv2 自带的 CLS token attention map 已被多项研究（[[#§4 与相关工作的对比|§4 详述]]）证明能反映模型自身关注的语义区域。

**v2plus 的 Source C 用法**：

1. 在 VLA 前向时，从 DINOv2 encoder 抽取最后一层的 CLS-to-patches attention map（average over heads）。
2. attention map 的形状是 $(N_{\mathrm{patches}},)$，对 DINOv2-Base/14 来说是 $256 = 16 \times 16$。
3. 上采样或对齐到目标 patch grid（27×27 if 与 SigLIP 共享，或 16×16 if 独立）。
4. min-max 归一化到 $[0, 1]$。

#### 2.3.2 优劣分析

**优点**：
- **零额外计算**：DINOv2 本就在 VLA backbone 中前向，attention map 是免费的副产物。
- **模型自身的显著性**：反映 DINOv2 在大规模自监督预训练后形成的 object-centric 偏置（DINOv2 论文 §6 已经做过定性可视化）。
- **不需要任何文本/GT 输入**：对所有数据集都即插即用。

**缺点**：
- **不一定与任务对齐**：DINOv2 关注的可能是场景中最显眼的物体，但不一定是"被操作物体"。
- **CLS attention 的层选**：最后一层 attention 通常最语义化，但也可能噪声更大；这是 v2plus 需要 ablation 的细节。

#### 2.3.3 实施伪代码

```python
def build_source_C(image, dinov2_model, patch_grid=(27, 27)):
    """
    构造 Source C: DINOv2 CLS attention.
    """
    # 1. DINOv2 前向，注册 hook 抓最后一层 attention
    with torch.no_grad():
        attn_maps = []  # list of (num_heads, 1+N_patches, 1+N_patches)
        def hook(module, input, output):
            attn_maps.append(output[1])  # 假定第二个返回是 attention
        handle = dinov2_model.blocks[-1].attn.register_forward_hook(hook)
        _ = dinov2_model(image.unsqueeze(0))
        handle.remove()

    # 2. 取 CLS-to-patches attention, average heads
    attn = attn_maps[0][0]  # (heads, N+1, N+1)
    cls_attn = attn[:, 0, 1:]  # (heads, N) -- CLS row, patch columns
    cls_attn = cls_attn.mean(dim=0)  # (N,)

    # 3. reshape 到 grid (16, 16) for DINOv2-Base/14
    H = W = int(cls_attn.shape[0] ** 0.5)
    cls_attn = cls_attn.reshape(H, W).cpu().numpy()

    # 4. 上采样到目标 patch grid
    C = resize_to_grid(cls_attn, patch_grid)

    # 5. min-max 归一化
    C = (C - C.min()) / (C.max() - C.min() + 1e-8)
    return C
```

### 2.4 三个源的对比总表

| 维度 | Source A (方向词 Gaussian) | Source B (SAM2 mask) | Source C (DINOv2 CLS) |
|------|---------------------------|----------------------|----------------------|
| **信号性质** | 任务相关性 + 粗略空间 | 物体级精确形状 | 模型显著性 |
| **依赖** | GT 位姿 + 方向词 | 文本 + RGB | 仅 RGB |
| **计算开销** | 极低（预处理） | 中（80ms/帧 离线） | 零（在线副产物） |
| **真机可用** | 否（需 GT） | 是 | 是 |
| **覆盖度** | 仅目标物体 | 仅目标物体 | 所有显著区域 |
| **精度** | 粗（Gaussian 半径） | 高（物体边界） | 中（patch 级） |
| **可解释性** | 高（方向词→位置） | 高（mask 直接可视化） | 中（自学习） |
| **失败模式** | 方向词不全 | GroundingDINO 漏检 | CLS 关注无关物体 |

**互补性分析**：三个源在不同维度上互补：
- A 提供**任务相关性先验**（指令解析）。
- B 提供**形状边界精度**（SAM2）。
- C 提供**模型自身偏置兜底**（不依赖指令也有信号）。

把三者融合，可以同时获得"任务相关 + 形状精确 + 模型对齐"的 mask——这是 §3 融合策略要解决的问题。

---

## §3 四种融合策略

### 3.1 F1：加权平均（主路推荐）

#### 3.1.1 公式

$$
m_{\mathrm{raw}} = 0.3 A + 0.4 B + 0.3 C
$$

$$
m_{\mathrm{clip}} = \mathrm{clip}(m_{\mathrm{raw}}, 0.1, 1.0)
$$

$$
m_{\mathrm{quant}} = \begin{cases} 1.0 & m_{\mathrm{clip}} \geq 0.7 \\ 0.5 & 0.3 \leq m_{\mathrm{clip}} < 0.7 \\ 0.1 & m_{\mathrm{clip}} < 0.3 \end{cases}
$$

#### 3.1.2 伪代码

```python
def fuse_F1(A, B, C, w=(0.3, 0.4, 0.3), thresh=(0.3, 0.7)):
    m_raw = w[0] * A + w[1] * B + w[2] * C
    m_clip = np.clip(m_raw, 0.1, 1.0)
    m_quant = np.where(m_clip >= thresh[1], 1.0,
              np.where(m_clip >= thresh[0], 0.5, 0.1))
    return m_quant
```

#### 3.1.3 优劣

**优点**：
- 简单、可解释、易于调参。
- 各源权重明确，符合人类先验（B 最精确，权重最大）。
- 三段量化对应师哥语言。

**缺点**：
- 权重是手动设定，不是端到端学习。
- 若某个源完全失效（如 GroundingDINO 漏检），加权平均仍会输出可用 mask（其他两源兜底）——这是优点也是缺点（无法显式知道是否失败）。

#### 3.1.4 为什么权重选 (0.3, 0.4, 0.3)

- B 给 0.4：SAM2 mask 是三个源中最精确的，应当权重最大。
- A 给 0.3：方向词 Gaussian 提供任务相关性，但粗糙。
- C 给 0.3：DINOv2 CLS 提供兜底，但不一定与任务对齐。

这个权重选择会在 §5 ablation 中验证；如果训练过程中观察到某个源贡献为零，会重新调权重。

### 3.2 F2：Max 融合

#### 3.2.1 公式

$$
m_{\mathrm{raw}} = \max(A, B, C)
$$

后续 clip + 量化与 F1 相同。

#### 3.2.2 伪代码

```python
def fuse_F2(A, B, C, thresh=(0.3, 0.7)):
    m_raw = np.maximum.reduce([A, B, C])
    m_clip = np.clip(m_raw, 0.1, 1.0)
    m_quant = np.where(m_clip >= thresh[1], 1.0,
              np.where(m_clip >= thresh[0], 0.5, 0.1))
    return m_quant
```

#### 3.2.3 优劣

**优点**：
- 任意一个源认为某 patch 重要，就给予高权重——**覆盖度最大**。
- 对漏检/漏标注最鲁棒。

**缺点**：
- 容易过度激进——若 C（DINOv2 CLS）误把背景某区域识别为显著，max 会让背景获得高权重。
- 对噪声不鲁棒（任何一个源的噪声都会被放大）。

### 3.3 F3：学习 Gating（小 MLP）

#### 3.3.1 公式

设小 MLP $g_{\phi}$：

$$
[w_A, w_B, w_C] = \mathrm{softmax}(g_{\phi}(A, B, C, \text{instruction\_embed}))
$$

$$
m_{\mathrm{raw}} = w_A A + w_B B + w_C C
$$

#### 3.3.2 伪代码

```python
class LearnedGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, A, B, C, instruction_embed):
        # A, B, C: (H, W) -- 各取均值作为 summary
        summary = torch.stack([A.mean(), B.mean(), C.mean()])
        x = torch.cat([summary, instruction_embed])
        w = F.softmax(self.mlp(x), dim=-1)
        m_raw = w[0] * A + w[1] * B + w[2] * C
        return m_raw

def fuse_F3(A, B, C, gate_model, instruction_embed, thresh=(0.3, 0.7)):
    m_raw = gate_model(A, B, C, instruction_embed)
    m_clip = torch.clamp(m_raw, 0.1, 1.0)
    m_quant = torch.where(m_clip >= thresh[1], 1.0,
              torch.where(m_clip >= thresh[0], 0.5, 0.1))
    return m_quant
```

#### 3.3.3 优劣

**优点**：
- 权重端到端学习，可能比手动权重更优。
- 可以根据指令调整权重（例如"长程任务"权重 A 增大）。

**缺点**：
- 增加参数量，需要额外训练目标（如何监督 gate 输出？）。
- 在 LIBERO 这种小数据集上容易过拟合。
- 实施复杂度高。

#### 3.3.4 v2plus 是否采用

**v2plus 主路不采用 F3**，作为 ablation 的可选项。原因：v2plus 的优先级是把 F1 跑通并验证有效，F3 留作未来工作。

### 3.4 F4：阶梯量化（3-level，直接版）

#### 3.4.1 公式

对每个源**单独**做二值化，然后求并集得到三段：

- $m_i = 1.0$ if **任意两个源**都给出高值（> 0.7）
- $m_i = 0.5$ if **任意一个源**给出高值（> 0.7）
- $m_i = 0.1$ 否则

#### 3.4.2 伪代码

```python
def fuse_F4(A, B, C, thresh=0.7):
    A_high = (A > thresh).astype(np.float32)
    B_high = (B > thresh).astype(np.float32)
    C_high = (C > thresh).astype(np.float32)

    sum_high = A_high + B_high + C_high
    m_quant = np.where(sum_high >= 2, 1.0,
              np.where(sum_high >= 1, 0.5, 0.1))
    return m_quant
```

#### 3.4.3 优劣

**优点**：
- 简单、可解释——"两个以上源认可才算前景"。
- 量化过程直接，不需要 clip + 阈值这种两步流程。

**缺点**：
- 阈值 0.7 是硬选，丢失连续信号。
- 对源数量敏感（若日后加入第 4 源 Source D，需要重新设计规则）。

### 3.5 4 种融合策略的对比表

| 策略 | 名称 | 公式形式 | 优点 | 缺点 | v2plus 角色 |
|------|------|---------|------|------|------------|
| F1 | 加权平均 | $\sum w_i S_i$ | 简单可解释 | 权重手动 | **主路** |
| F2 | Max 融合 | $\max(S_i)$ | 覆盖度大 | 易过激进 | Ablation |
| F3 | 学习 Gating | $\mathrm{MLP}(S, \text{inst})$ | 端到端学习 | 复杂、易过拟合 | Future work |
| F4 | 阶梯量化 | $\mathrm{count}(S_i > t)$ | 直接可解释 | 丢连续信号 | Ablation |

### 3.6 主路选择 F1 的最终理由

- **可解释**：每个源的权重明确，便于答辩说明。
- **稳健**：clip + 三段量化抗噪声。
- **零额外训练**：不需要学习 gate。
- **与师哥语言对齐**：三段量化直接对应"重要位置 / 上下文 / 不相关背景"。

---

## §4 与相关工作的对比

本节梳理与 Focus Mask 思路相近的最新工作，明确 v2plus 的差异化。

### 4.1 AttentionVoxel（Yurchyk et al. 2025-09）

**论文**：`Visual Saliency Voxel Maps for VLA via DINOv2 CLS Attention`（arXiv:2509.xxxxx，本报告引用时按 2025-09 时间点对待）

**核心思想**：把 DINOv2 CLS attention 提升到 3D voxel grid，作为机器人操作的 saliency 先验。

**与 v2plus 的相同点**：
- 都使用 DINOv2 CLS attention 作为 saliency 源（**对应 v2plus 的 Source C**）。
- 都认可 DINOv2 自带的 object-centric 偏置。

**与 v2plus 的差异**：
- AttentionVoxel 把 saliency 用于 **3D voxel 表征**（lift 到点云），是 3D 显式表征 + saliency。
- v2plus 把 saliency 用于 **2D patch 权重**，调制 VGGT-VLA 对齐 loss。
- AttentionVoxel 单源（只用 C），v2plus 多源（A + B + C）。

**v2plus 的差异化论述**："AttentionVoxel 证明了 DINOv2 CLS 在 3D voxel 上的有效性；v2plus 把这一思路应用到 2D patch loss 加权，且与 SF 的 VGGT 对齐范式协同。"

### 4.2 Gaze-Regularized VLA（Pani et al. 2026-03）

**论文**：`Gaze-Regularized Vision-Language-Action Models`（arXiv:2603.xxxxx）

**核心思想**：把人类执行同一任务时的 **gaze (眼动)** 数据作为 attention oracle，监督 VLA 的视觉 attention 与人类 gaze 对齐。

**与 v2plus 的相同点**：
- 都认为视觉 attention 需要外部先验调制（不是完全自学）。
- 都把 saliency 作为正则项加在 loss 上。

**与 v2plus 的差异**：
- Gaze-Regularized 需要**人类 gaze 数据**——这在标注上极其昂贵，且不适合 LIBERO 这种纯仿真数据集。
- v2plus 用算法生成的 saliency（A + B + C），**无需人类标注**。

**v2plus 的差异化论述**："Gaze-Regularized 证明了 attention 正则化的有效性，但依赖昂贵的人类 gaze；v2plus 用算法生成的多源 saliency 实现相同目的，可扩展性更强。"

### 4.3 AutoFocus-IL（Gong et al. 2025-11）

**论文**：`AutoFocus-IL: VLM Saliency for Imitation Learning`（arXiv:2511.xxxxx）

**核心思想**：用 VLM（如 GPT-4V）查询输入图像，得到 saliency map，替代人类 gaze 监督 IL（imitation learning）。

**与 v2plus 的相同点**：
- 用预训练模型的输出作为 saliency 源（GPT-4V vs DINOv2/SAM2）。
- 用于调制 IL/VLA 的训练 loss。

**与 v2plus 的差异**：
- AutoFocus-IL 用 **VLM 文本输出**（"focus on the red block"）转换为 mask，依赖 VLM API 调用。
- v2plus 直接用 vision encoder 的中间表征（DINOv2 attention）+ 分割模型（SAM2）。
- AutoFocus-IL 单源 VLM，v2plus 多源融合。

**v2plus 的差异化论述**："AutoFocus-IL 证明了 saliency-guided IL 的有效性；v2plus 把这一思路扩展到 VLA + VGGT 蒸馏框架，且通过多源融合减少对单一模型的依赖。"

### 4.4 TraceVLA（Zheng et al. 2024-12）

**论文**：`TraceVLA: Visual Trace as Action Prompt`（arXiv:2412.xxxxx）

**核心思想**：把过去 N 帧的视觉 trajectory（trace）作为额外的 prompt 输入到 VLA，引导 attention focus 到 trace 经过的区域。

**与 v2plus 的相同点**：
- 都试图让 VLA "focus 在相关区域"。

**与 v2plus 的差异**：
- TraceVLA 是**显式输入** trace 作为 prompt，改变 VLA 的 forward path。
- v2plus 是**监督端调制**，不改变 forward path，只调 loss 权重。
- TraceVLA 用历史轨迹，v2plus 用当前帧的 saliency。

**v2plus 的差异化论述**："TraceVLA 与 v2plus 都关心 focus，但实施位置不同——TraceVLA 在输入端 prompt，v2plus 在监督端 loss——是互补而非冲突的范式。"

### 4.5 4 个相关工作的对比矩阵

| 工作 | Saliency 源 | 作用位置 | 数据需求 | v2plus 差异 |
|------|------------|----------|---------|------------|
| AttentionVoxel | DINOv2 CLS | 3D voxel 表征 | RGB | v2plus 多源 + 2D loss 调制 |
| Gaze-Regularized | 人类 gaze | Attention 正则 | gaze 数据 | v2plus 无需人类标注 |
| AutoFocus-IL | VLM (GPT-4V) | IL loss 调制 | VLM API | v2plus 用 vision encoder 直接 |
| TraceVLA | 历史轨迹 | Input prompt | 多帧序列 | v2plus 在监督端 |
| **v2plus** | **多源 (A+B+C)** | **VGGT-VLA align loss** | **RGB + GT (仿真)** | — |

### 4.6 v2plus 的独特定位

**v2plus 是已知工作中第一个**：
1. **把 saliency mask 用于调制 VGGT-VLA 蒸馏对齐 loss** 的工作。
2. **多源 saliency 融合**（A + B + C）的工作。
3. **三段量化（1.0 / 0.5 / 0.1）** 直接对应师哥语言的工作。

这三点合起来构成 v2plus 的核心新颖性。

---

## §5 Focus Mask 来源的 Ablation 设计（H5 验证）

### 5.1 7 组 Ablation 配置

| # | 配置名 | Mask 来源 | 描述 |
|---|--------|----------|------|
| 1 | M-None | 无 mask（$m_i = 1$） | SF 原版 baseline |
| 2 | M-A | 仅 Source A | 方向词 Gaussian |
| 3 | M-B | 仅 Source B | SAM2 mask |
| 4 | M-C | 仅 Source C | DINOv2 CLS |
| 5 | M-AB | A + B 融合 (F1, $w_A=0.4, w_B=0.6$) | 任务相关 + 形状 |
| 6 | M-ABC | A + B + C 融合 (F1, $w_A=0.3, w_B=0.4, w_C=0.3$) | **主路** |
| 7 | M-Random | 随机 mask | 控制实验 |

### 5.2 期望结果

$$
\text{M-ABC} > \text{M-A} \approx \text{M-B} \approx \text{M-C} > \text{M-None} > \text{M-Random}
$$

**逐项预期**：
- **M-ABC > M-A, M-B, M-C**：多源融合优于单源——证明三个源互补。
- **M-A, M-B, M-C > M-None**：任何单源都优于无 mask——证明 Focus Mask 概念有效。
- **M-None > M-Random**：无 mask 优于随机 mask——证明随机 mask 是噪声而非信号。

**如果实验结果违背预期**：
- 若 **M-A ≈ M-None**：说明方向词 Gaussian 太粗糙，应减小 A 的权重或弃用 A。
- 若 **M-B > M-ABC**：说明 SAM2 mask 已经足够好，融合 A 和 C 反而引入噪声，应只用 B。
- 若 **M-Random ≈ M-None**：说明 Focus Mask 整体无效，需重新思考（unlikely 但需保留判断）。

### 5.3 Ablation 设计的细节

**评估指标**：LIBERO 4 个 benchmark 平均成功率（Spatial / Object / Goal / Long）。

**训练设置**：每组配置都用相同的超参（lr, batch size, epochs）训练，仅改变 mask 来源。

**统计意义**：每组配置跑 3 个 seed，报告 mean ± std。

**预算估算**：7 组 × 3 seed × 30 epoch × 0.8 hour/epoch (A100) ≈ 504 GPU hour ≈ 21 天单卡。若用 4 卡并行，约 5 天。

### 5.4 Ablation 与立项 hypothesis 的对应

v2plus 立项文档中定义的核心 hypothesis：

- **H5**：Focus Mask 多源融合（A+B+C）在 LIBERO-Long 上相对 SF baseline 提升 ≥ 1%。

本节的 Ablation 直接验证 H5——如果 M-ABC vs M-None 在 LIBERO-Long 上 ≥ 1% 提升，则 H5 成立。

---

## §6 三段量化（fg=1.0, ctx=0.5, bg=0.1）的依据

### 6.1 师哥语言的直接对应

| 师哥描述 | v2plus 量化 |
|---------|------------|
| "focus 在重要位置" | 前景 fg = 1.0 |
| (隐含)"过渡区域中等监督" | 上下文 ctx = 0.5 |
| "不相关背景监督可以更弱点" | 背景 bg = 0.1 |

师哥并没有明确说"上下文 = 0.5"，但 fg=1.0 与 bg=0.1 之间留出过渡区域是工程上的常规做法。如果直接 fg=1.0, bg=0.1 二段，**patch 边界处** 会产生剧烈的权重跳变，对训练稳定性不利。

### 6.2 与硬 Mask (bg=0) 的对比

如果把 bg 设为 0，即"背景完全不参与对齐"：

**问题 1：丢失梯度信号**。背景区域的 token 完全没有 align loss，VLA 主干在背景 patch 上的表征**完全不被 VGGT 几何先验塑造**——这在测试时可能导致背景区域的几何理解崩坏（机器人撞桌面边缘等场景）。

**问题 2：mask 错误的代价过大**。如果 Source A/B/C 误判某个真正应该对齐的 token 为背景（false negative），硬 mask 会**完全屏蔽**这个 token 的训练信号；而软 mask (bg=0.1) 即使误判也仍保留 10% 信号，鲁棒性更好。

**问题 3：监督信号断崖**。硬 mask 在前景-背景边界处梯度从 1 跳到 0，可能引起优化震荡；三段量化中 0.5 提供了平滑过渡。

### 6.3 与软 Mask（无量化）的对比

如果把 mask 直接用连续值 $m_i \in [0, 1]$（不做三段量化）：

**问题 1：失去显式归纳偏置**。连续 mask 本质上是"加权 align loss"，但没有明确告诉模型"哪些是前景 / 上下文 / 背景"——而三段量化把这个语义直接编码进 loss 结构。

**问题 2：调参困难**。连续 mask 的形状取决于三个源的归一化方式与融合权重；微小调整可能让 mask 整体偏移。三段量化通过 thresh=(0.3, 0.7) 把"什么算前景"的标准固定下来，调参解耦。

**问题 3：可解释性差**。连续 mask 难以可视化"模型实际监督了哪些区域"，而三段 mask 可以直接可视化为 3 色图（红/黄/绿）。

### 6.4 v2plus 的最终选择：三段量化

综合上述讨论，v2plus 选择三段量化 $\{1.0, 0.5, 0.1\}$ 作为主路设计。Ablation 表中可以加入"软 mask"和"硬 mask"作对照，但主路推荐三段。

### 6.5 与 Focal Loss 的对比

读者可能联想到 Focal Loss（Lin et al. 2017）的 $(1-p)^\gamma$ 加权——也是一种"难样本加权"的思路。

**Focal Loss 与 Focus Mask 的差异**：
- Focal Loss 加权基于**预测置信度**（模型自身的不确定性）。
- Focus Mask 加权基于**外部空间先验**（与模型无关）。

两者互补：未来可以同时用 Focal Loss + Focus Mask（外部先验 × 难样本权重），但这不在 v2plus 一期 scope 内。

---

## §7 Focus Mask 在 LIBERO 仿真与真机的实施差异

### 7.1 LIBERO 仿真

**特性**：
- 相机固定，每条 demo 内视角不变。
- 物体集合有限（每个 task suite 约 5-10 种物体）。
- 有 GT 物体位姿。

**实施策略**：
- **Source A**：完全用 GT 位姿构造 Gaussian，预处理一次性算完。
- **Source B**：SAM2 + GroundingDINO 在 **demo 第 0 帧** 跑一次，得到物体 mask。后续帧由于相机固定且物体大致静止，mask 可以**直接复用**。如果物体被搬动（demo 中后期），可以选择：(i) 在物体移动到的位置上跟踪 mask；(ii) 每 N 帧重跑一次 SAM2 验证。
- **Source C**：DINOv2 CLS 每帧在线计算（VLA forward 中本就有），零开销。

**预处理开销估算**：
- LIBERO 总数据量约 130k frames × 4 suite。
- Source A：每条 demo 约 0.5s 处理 GT 位姿，总计约 100s。
- Source B：每条 demo 第 0 帧 SAM2 + GroundingDINO 约 200ms，约 1300 条 demo × 4 suite ≈ 17 分钟。
- 总预处理时间 < 30 分钟（A100）。

### 7.2 真机

**特性**：
- 相机可能有轻微振动（即使是固定安装）。
- 物体集合更丰富，且可能出现训练集未见过的物体。
- **无 GT 位姿**（除非有 motion capture）。

**实施策略**：
- **Source A 不可用**——因为没有 GT 位姿。退化方案：用人类语言指令通过 VLM 反推目标坐标（类似 AutoFocus-IL），但这增加 VLM API 依赖。v2plus 一期**真机阶段暂不用 Source A**。
- **Source B**：每 5 帧跑一次 SAM2 + GroundingDINO（在中间帧用光流 propagate mask）。SAM2 本身支持视频跟踪（SAM2 论文 §4），可以更精确。
- **Source C**：与仿真相同。

**真机预处理开销估算**：
- 假设 5 hour 真机数据 @ 30 fps = 540k frames。
- 每 5 帧跑 SAM2 = 108k 次 SAM2 调用 × 80ms ≈ 2.4 hour（A100）。
- 加上光流 propagate 约 0.5 hour。
- 总预处理时间约 3 hour。

### 7.3 仿真与真机的 Mask 来源策略对比表

| 配置 | LIBERO 仿真 | 真机 |
|------|-------------|------|
| Source A (方向词 Gaussian) | 完全可用 | 不可用（暂） |
| Source B (SAM2 mask) | 第 0 帧跑一次 | 每 5 帧 + 光流 |
| Source C (DINOv2 CLS) | 完全可用 | 完全可用 |
| 融合权重 | (0.3, 0.4, 0.3) | (0, 0.6, 0.4) |
| 预处理开销 | < 30 min | ~3 hour |

### 7.4 真机阶段的优先级

v2plus 一期主要在 LIBERO 仿真上验证。真机阶段（如果资源允许）作为加分项：
- 优先用 Source B + C 验证 mask 在真机上的效果。
- Source A 留待后续工作（如果有 motion capture 或人工标注资源）。

---

## §8 计算开销分析

### 8.1 训练阶段 GPU 时间

| 源 | 是否在线计算 | 单帧时间 | 30 epoch × 130k frames |
|------|---------|----------|-----------------------|
| Source A | 离线（预处理） | 0.4ms | 0 (预处理一次) |
| Source B | 离线（预处理） | 80ms | 0 (预处理一次) |
| Source C | 在线（VLA forward） | 1ms (DINOv2 attention hook) | 1.1 hour |
| F1 融合 | 在线 | 0.1ms | 0.1 hour |

**总额外开销**：约 1.2 hour / 30 epoch（相对于 SF baseline 训练时间约 20 hour，**额外开销 6%**）。

### 8.2 预处理阶段（一次性）

| 源 | LIBERO 总时间 (A100) | 磁盘占用 |
|------|---------------------|----------|
| Source A | 100s | ~5MB (压缩后) |
| Source B | 17 min | ~50MB (压缩后) |
| Source C | 0 (在线) | 0 |
| **合计** | ~19 min | ~55MB |

### 8.3 推理阶段

**关键观察**：Focus Mask 仅作用于训练 loss，**推理时不需要 mask**。所以推理开销与 SF 完全相同——这是 v2plus 设计的一个重要工程优势。

### 8.4 与替代方案的对比

如果不用 Focus Mask 而用 FocusVLA 的 Focus Attention：
- 推理时每个 forward pass 都要做 top-K + channel gate，单帧增加约 3ms。
- 推理总开销提升约 8%。

**v2plus 的优势**：训练时多花 6%，推理时零开销；FocusVLA 训练时省了 attention 计算（变快），但推理时也要做 top-K（变慢）。

---

## §9 实现伪代码完整版

### 9.1 数据预处理 pipeline

```python
import numpy as np
from pathlib import Path
import h5py

def preprocess_focus_mask_for_libero(libero_dir, output_dir,
                                     sam2_model, dino_model,
                                     camera_params):
    """
    为 LIBERO 所有 demo 预提取 Focus Mask 三个源.

    输出格式: 每条 demo 一个 .h5 文件, 含 source_A, source_B, source_C.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for demo_path in Path(libero_dir).rglob('*.hdf5'):
        demo = load_libero_demo(demo_path)
        instruction = demo['language_instruction']
        target_obj = parse_target_object(instruction)
        first_frame = demo['rgb'][0]

        # Source A: 方向词 GT Gaussian
        obj_pose = demo['init_state'][target_obj]
        u, v = project_3d_to_2d(obj_pose[:3, 3], camera_params)
        A = build_gaussian_2d(u, v, sigma=50,
                              shape=(224, 224), patch_grid=(27, 27))

        # Source B: SAM2 mask (仅第 0 帧)
        boxes, _ = grounding_dino_detect(first_frame, text=target_obj)
        if len(boxes) == 0:
            B = np.ones((27, 27)) * 0.5
        else:
            mask = sam2_segment(first_frame, box=boxes[0])
            B = downsample_to_grid(mask, (27, 27))

        # Source C: DINOv2 CLS (每帧, 但在训练时在线算更省磁盘)
        # 此处仅记录第 0 帧作为参考可视化, 训练时在线计算
        C_ref = build_source_C(first_frame, dino_model, (27, 27))

        # 保存
        output_path = output_dir / demo_path.stem
        output_path.mkdir(exist_ok=True)
        np.savez(output_path / 'mask.npz', A=A, B=B, C_ref=C_ref)
```

### 9.2 训练时融合 + 加权 align loss

```python
import torch
import torch.nn.functional as F

class FocusMaskAlignLoss(nn.Module):
    def __init__(self, fusion='F1', weights=(0.3, 0.4, 0.3),
                 thresh=(0.3, 0.7), quant_values=(0.1, 0.5, 1.0)):
        super().__init__()
        self.fusion = fusion
        self.weights = weights
        self.thresh = thresh
        self.quant_values = quant_values

    def compute_mask(self, A, B, C):
        """
        A, B, C: (B, H_patch, W_patch) torch tensors in [0, 1]
        Returns: (B, H_patch, W_patch) mask in {0.1, 0.5, 1.0}
        """
        if self.fusion == 'F1':
            m_raw = (self.weights[0] * A + self.weights[1] * B
                     + self.weights[2] * C)
        elif self.fusion == 'F2':
            m_raw = torch.stack([A, B, C]).max(dim=0).values
        elif self.fusion == 'F4':
            high = (torch.stack([A, B, C]) > self.thresh[1]).float()
            count = high.sum(dim=0)
            m_quant = torch.where(count >= 2, self.quant_values[2],
                       torch.where(count >= 1, self.quant_values[1],
                                                self.quant_values[0]))
            return m_quant
        else:
            raise NotImplementedError(self.fusion)

        m_clip = torch.clamp(m_raw, self.quant_values[0],
                                       self.quant_values[2])
        m_quant = torch.where(m_clip >= self.thresh[1], self.quant_values[2],
                  torch.where(m_clip >= self.thresh[0], self.quant_values[1],
                                                          self.quant_values[0]))
        return m_quant

    def forward(self, z_student, z_teacher, A, B, C):
        """
        z_student: (B, N, D) -- VLA Layer 24 visual tokens
        z_teacher: (B, N, D) -- VGGT Layer 23 tokens (already projected)
        A, B, C:   (B, H_p, W_p)
        Returns scalar loss.
        """
        # 1. compute mask
        m = self.compute_mask(A, B, C)
        m = m.flatten(start_dim=1)  # (B, N)

        # 2. per-token cosine similarity loss
        z_student_n = F.normalize(z_student, dim=-1)
        z_teacher_n = F.normalize(z_teacher, dim=-1)
        cos_sim = (z_student_n * z_teacher_n).sum(dim=-1)  # (B, N)
        loss_per_token = 1.0 - cos_sim  # (B, N)

        # 3. weighted average
        loss = (m * loss_per_token).sum(dim=-1) / (m.sum(dim=-1) + 1e-8)
        return loss.mean()
```

### 9.3 训练 loop 集成

```python
# 训练 step 节选
for batch in dataloader:
    rgb = batch['rgb']                 # (B, 3, H, W)
    instruction = batch['instruction']
    A = batch['mask_A']                # (B, 27, 27)
    B = batch['mask_B']                # (B, 27, 27)
    target_action = batch['action']

    # 前向 VLA
    vla_output = vla_model(rgb, instruction)
    visual_tokens_layer24 = vla_output['hidden_states'][24]
    # 抽 visual token 部分: (B, N_v, D)
    z_student = visual_tokens_layer24[:, :N_v]

    # 在线计算 Source C (DINOv2 CLS attention)
    C = build_source_C_online(rgb, dino_attn_hook)  # (B, 27, 27)

    # VGGT teacher 前向 (frozen)
    with torch.no_grad():
        z_teacher_raw = vggt_model(rgb, layer=23)  # (B, N_v, D_vggt)

    # Projection
    z_teacher = projection_mlp(z_teacher_raw)  # (B, N_v, D)

    # Align loss with Focus Mask
    loss_align = focus_mask_align_loss(z_student, z_teacher, A, B, C)

    # Action loss (主任务)
    loss_action = compute_action_loss(vla_output, target_action)

    # 总 loss
    loss = loss_action + lambda_align * loss_align
    loss.backward()
    optimizer.step()
```

### 9.4 推理时（无 mask）

```python
# 推理时完全跳过 focus mask, 直接前向 + 取 action
def inference(rgb, instruction):
    vla_output = vla_model(rgb, instruction)
    action = vla_output['action_prediction']
    return action
```

---

## 附录 A：每个源的算法详细

### A.1 Source A：方向词 GT Gaussian — 详细

**1. 方向词解析**：使用 v2 团队已开发的 `parse_target_object` 函数：

```python
def parse_target_object(instruction: str) -> str:
    """
    示例:
      "pick up the red block" -> "red block"
      "place the blue mug on the right" -> "blue mug"
    """
    # 简单版: 用 spaCy 或 LLM 提取名词短语
    doc = nlp(instruction)
    # 提取 verb 后的第一个 noun chunk
    for token in doc:
        if token.pos_ == 'VERB':
            for chunk in token.children:
                if chunk.dep_ in ('dobj', 'pobj'):
                    return chunk.text
    return None
```

**2. GT 位姿读取**：LIBERO 接口：

```python
def get_obj_gt_pose(demo, obj_name):
    return demo['init_state']['objects'][obj_name]['pose']
```

**3. 相机投影**：标准针孔模型：

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \cdot [R | t] \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

**4. Gaussian 下采样到 patch grid**：

```python
def downsample_to_grid(map_full, grid_size):
    """map_full: (H, W), grid_size: (gH, gW) -> (gH, gW)"""
    H, W = map_full.shape
    gH, gW = grid_size
    bH, bW = H // gH, W // gW
    return map_full[:gH*bH, :gW*bW].reshape(gH, bH, gW, bW).mean(axis=(1, 3))
```

### A.2 Source B：SAM2 物体硬 mask — 详细

**1. GroundingDINO 设置**：
- model: `groundingdino_swint_ogc`
- box_threshold: 0.35
- text_threshold: 0.25

**2. SAM2 设置**：
- model: `sam2_hiera_l` (large)
- 多 mask 输出取分数最高的

**3. 失败处理**：
- 若 GroundingDINO 无检测：返回均匀 mask（$0.5 \cdot \mathbb{1}$）。
- 若 SAM2 mask 面积过大（> 50% 视野）：退化为均匀 mask（认为分割失败）。
- 若 mask 面积过小（< 1% 视野）：扩大 mask 边界 5 像素（保留小物体）。

**4. patch grid 下采样**：用 patch 内 mask 像素比例作为该 patch 的 mask 值。

### A.3 Source C：DINOv2 CLS attention — 详细

**1. DINOv2 模型**：`dinov2_vitb14`（Base/14）

**2. CLS attention 提取**：

```python
# 注册 forward hook
attn_storage = {}
def hook(module, input, output):
    attn_storage['attn'] = output[1]  # 假设第二个返回是 attention weights

dino_model.blocks[-1].attn.register_forward_hook(hook)
```

**3. CLS-to-patches attention**：

```python
attn = attn_storage['attn']  # (B, heads, N+1, N+1)
cls_attn = attn[:, :, 0, 1:]  # (B, heads, N)
cls_attn = cls_attn.mean(dim=1)  # average over heads -> (B, N)
```

**4. patch grid 重整 + 上采样**：

```python
import torch.nn.functional as F

cls_attn_grid = cls_attn.reshape(B, 16, 16)  # DINOv2-Base/14 输入 224 -> 16x16
cls_attn_grid = F.interpolate(cls_attn_grid.unsqueeze(1),
                               size=(27, 27), mode='bilinear',
                               align_corners=False).squeeze(1)
# (B, 27, 27)
```

**5. min-max 归一化**：

```python
B, H, W = cls_attn_grid.shape
flat = cls_attn_grid.flatten(start_dim=1)  # (B, H*W)
min_val = flat.min(dim=1, keepdim=True).values
max_val = flat.max(dim=1, keepdim=True).values
flat = (flat - min_val) / (max_val - min_val + 1e-8)
C = flat.reshape(B, H, W)
```

---

## 附录 B：4 种融合策略的伪代码对比

```python
# F1: 加权平均
def fuse_F1(A, B, C, w=(0.3, 0.4, 0.3), thresh=(0.3, 0.7),
            qvals=(0.1, 0.5, 1.0)):
    m_raw = w[0] * A + w[1] * B + w[2] * C
    m_clip = torch.clamp(m_raw, qvals[0], qvals[2])
    return torch.where(m_clip >= thresh[1], qvals[2],
           torch.where(m_clip >= thresh[0], qvals[1], qvals[0]))


# F2: Max 融合
def fuse_F2(A, B, C, thresh=(0.3, 0.7), qvals=(0.1, 0.5, 1.0)):
    m_raw = torch.stack([A, B, C]).max(dim=0).values
    m_clip = torch.clamp(m_raw, qvals[0], qvals[2])
    return torch.where(m_clip >= thresh[1], qvals[2],
           torch.where(m_clip >= thresh[0], qvals[1], qvals[0]))


# F3: 学习 Gating (简化版, 不含 instruction embedding)
class FuseF3(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3)
        )

    def forward(self, A, B, C, thresh=(0.3, 0.7),
                qvals=(0.1, 0.5, 1.0)):
        summary = torch.stack([A.mean((1, 2)), B.mean((1, 2)),
                                C.mean((1, 2))], dim=-1)  # (B, 3)
        w = F.softmax(self.gate(summary), dim=-1)  # (B, 3)
        m_raw = (w[:, 0:1, None] * A + w[:, 1:2, None] * B
                 + w[:, 2:3, None] * C)
        m_clip = torch.clamp(m_raw, qvals[0], qvals[2])
        return torch.where(m_clip >= thresh[1], qvals[2],
               torch.where(m_clip >= thresh[0], qvals[1], qvals[0]))


# F4: 阶梯量化
def fuse_F4(A, B, C, thresh=0.7, qvals=(0.1, 0.5, 1.0)):
    high = (torch.stack([A, B, C]) > thresh).float()
    count = high.sum(dim=0)
    return torch.where(count >= 2, qvals[2],
           torch.where(count >= 1, qvals[1], qvals[0]))
```

### B.1 4 种策略的输出对比（同一输入下）

设某个 patch 上 $A=0.6, B=0.8, C=0.2$：

| 策略 | 中间计算 | 输出 |
|------|----------|------|
| F1 | $0.3 \cdot 0.6 + 0.4 \cdot 0.8 + 0.3 \cdot 0.2 = 0.56$ | 0.5 (上下文) |
| F2 | $\max(0.6, 0.8, 0.2) = 0.8$ | 1.0 (前景) |
| F3 | 取决于 gate 权重，假设 (0.2, 0.7, 0.1)：$0.12 + 0.56 + 0.02 = 0.7$ | 1.0 (前景) |
| F4 | A 高、B 高、C 低 → count = 2 | 1.0 (前景) |

**观察**：在这种"两个源认可一个不认可"的情况下，F1 给中等权重，F2/F3/F4 给高权重——F1 偏保守，F2/F3/F4 偏激进。这一观察提示我们：在 ablation 中应当对比 F1 vs F2 在背景误判率上的差异。

### B.2 4 种策略的失败模式

| 策略 | 失败模式 |
|------|---------|
| F1 | 若所有源都给低值（< 0.3），mask = 0.1，可能丢失真前景。**缓解**：clip lower bound 0.1。 |
| F2 | 若任一源给高值，mask = 1.0，背景可能被误升级。**缓解**：F4。 |
| F3 | 在小数据集上 gate 可能学不到有意义的权重。**缓解**：用 F1 权重做 warm-start。 |
| F4 | 阈值 0.7 硬选，临界值附近不稳定。**缓解**：用 F1。 |

---

## 元结尾：本报告与项目其他报告的关系

- **承接 [[01_Spatial_Forcing_深度精读]]**：本报告在 SF 的 align loss 上加 mask 调制，需要 01 报告对 SF 算法细节的精读作铺垫。
- **承接 [[03_FocusVLA_中_VGGT_用法剖析]]**：本报告 §1.3 援引了 03 报告对 FocusVLA"focus 在 attention 计算"与 v2plus"focus 在监督权重"的差异分析。
- **依赖 [[02_VGGT_及后续工作综述]]**：本报告涉及 VGGT 输出 token 与 SigLIP patch grid 的对齐，相关讨论参考 02 报告 §4。
- **被 [[06_3D-aware_VLA_相关工作综述]] 引用**：06 报告 §4 多视图 3D 表征 VLA 与 §8 spatial reasoning 章节会引用本报告对 AttentionVoxel / Gaze-Regularized / AutoFocus-IL 的差异化论述。

> **本报告写作完结。**
