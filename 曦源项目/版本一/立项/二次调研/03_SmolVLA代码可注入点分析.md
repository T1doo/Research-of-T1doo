---
create time: 2026-05-22
tags:
  - 曦源项目
  - 立项调研
  - 二次调研
  - 代码分析
---

# SmolVLA 代码可注入点分析

> [!info] 范围
> 基于 HuggingFace `lerobot` 仓库 main 分支（SHA `c0a2e981`，2026-05-22 拉取）以及 `gakakulicc/RobustVLA` 仓库 master 分支的源码精读，定位 AAC（推理时） + RobustVLA（训练时） + LoRA(r=16) 在 SmolVLA 代码上的具体注入点。所有路径均相对仓库根目录，函数签名与行号均为实际抓取结果。

> [!note] 关键发现
> 1. `SmolVLAPolicy` 是 PreTrainedPolicy 的实现，**已经内建 PEFT/LoRA 支持**（`_get_default_peft_targets`），且默认 target_modules 已经覆盖动作专家的 Q/V projection 以及 5 个核心 MLP；
> 2. SmolVLA 内置 **Real-Time Chunking (RTC)** 机制（`rtc_processor`），是 chunk overlap 与异步推理的 in-tree 实现，AAC 可以与之协同；
> 3. 异步推理 stack 在 `src/lerobot/async_inference/` 是完整 gRPC 客户端-服务端架构，`robot_client.py` 内已经有 `_aggregate_action_queues` 钩子用于合并多个 chunk，AAC 的截断与提前重新预测自然映射进来；
> 4. RobustVLA 在 RobustVLA 仓库 `openpi/src/openpi/models/pi0.py` 中以 `_compute_adversarial_action_loss`（PGD L∞）和 `_compute_adversarial_image_loss` 的方式实现，将其移植到 PyTorch 版 SmolVLA 是 1:1 的逻辑转换。

---

## 1. 仓库结构概览

```
lerobot/                              (HuggingFace 官方,main 分支)
├── src/lerobot/
│   ├── policies/
│   │   └── smolvla/                  ← 主战场（全部源文件）
│   │       ├── __init__.py           (851 B)
│   │       ├── configuration_smolvla.py    (5.9 KB, 159 行)
│   │       ├── modeling_smolvla.py         (36.7 KB, 916 行)  ← 核心
│   │       ├── processor_smolvla.py        (3.7 KB, 103 行)
│   │       ├── smolvlm_with_expert.py      (23.4 KB, 561 行)  ← VLM+Expert 拼接
│   │       └── README.md
│   ├── policies/rtc/                 ← Real-Time Chunking 处理器（chunk 复用机制）
│   ├── async_inference/              ← 异步推理 stack
│   │   ├── policy_server.py          (16.9 KB, 439 行)
│   │   ├── robot_client.py           (20.4 KB, 517 行)
│   │   ├── helpers.py
│   │   ├── configs.py
│   │   └── constants.py
│   ├── scripts/
│   │   ├── lerobot_train.py          (24.2 KB, 605 行)  ← 训练入口
│   │   ├── lerobot_eval.py
│   │   └── lerobot_rollout.py
│   ├── optim/  configs/  datasets/  envs/  model/  processor/  rl/  transforms/  utils/

RobustVLA/                            (gakakulicc/RobustVLA, master 分支)
├── openpi/                           ← Physical-Intelligence/openpi fork
│   └── src/openpi/
│       ├── models/
│       │   ├── pi0.py                ← compute_loss + PGD 对抗 (448 行,JAX/Flax)
│       │   ├── pi0_backup.py
│       │   ├── lora.py
│       │   └── gemma.py
│       ├── robust_util/
│       │   ├── ucb_augmentation_balancer.py    ← UCB 多臂老虎机 (449 行)
│       │   ├── ucb_controlled_transforms.py
│       │   └── img_util.py                     ← 7 种视觉扰动算子
│       ├── prompt_augmentation.py    ← 语言提示词扰动 (85 行)
│       └── training/
├── openvla/                          ← OpenVLA 复现
├── prompts/                          ← 语言扰动 JSON 字典
└── LIBERO/                           ← LIBERO eval
```

---

## 2. SmolVLA 模型架构在代码中的拆分

### 2.1 VLM backbone (`SmolVLMWithExpertModel`)

- **位置**：`src/lerobot/policies/smolvla/smolvlm_with_expert.py:72` 起，`class SmolVLMWithExpertModel(nn.Module)`
- **作用**：把 SmolVLM2 (默认 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) 与「动作专家」（一个层数更少、宽度 ×0.75 的 Gemma-like Transformer）拼装在一起；VLM 处理图像/语言/状态，Expert 处理动作 token，两者通过 cross-attention 交互。
- **是否锁参**：默认 `freeze_vision_encoder=True`, `train_expert_only=True`（见 `configuration_smolvla.py:69-71`）。`set_requires_grad`（`smolvlm_with_expert.py:150`）按 config 决定。

### 2.2 Action expert 与 flow matching

- **类**：`VLAFlowMatching`（`modeling_smolvla.py:541`，nn.Module）
- **训练 loss**：`VLAFlowMatching.forward`（`modeling_smolvla.py:774-810`）：
  - `noise = sample_noise(shape, device)` (line 778-779)
  - `time = sample_time(bsize, device)` 使用 Beta(1.5, 1) 分布（line 632）
  - `x_t = time_expanded * noise + (1 - time_expanded) * actions`（line 785）
  - `u_t = noise - actions`（line 786，flow matching 的目标向量场）
  - `v_t = action_out_proj(suffix_out)`（line 808，模型预测的向量场）
  - `losses = F.mse_loss(u_t, v_t, reduction='none')`（line 809）
- **推理 sampling**：`VLAFlowMatching.sample_actions`（line 812-881）使用 Euler 法做 `num_steps=10` 步反向积分：
  - `dt = -1.0 / num_steps`（line 845）
  - 主循环 `for step in range(num_steps)` (line 848)：调用 `denoise_step` 得 `v_t`，更新 `x_t = x_t + dt * v_t`（line 876）
  - `denoise_step` 内部复用 prefix 的 KV cache（line 836-843 在循环开始前算）
- **chunk_size = 50, num_steps = 10, max_action_dim = 32**（`configuration_smolvla.py:29, 42, 63`）

### 2.3 Policy 封装：`SmolVLAPolicy`

- **位置**：`modeling_smolvla.py:226`，`class SmolVLAPolicy(PreTrainedPolicy)`
- **关键接口**：
  - `forward(batch, noise=None, time=None, reduction="mean")` → (line 358-413) **训练入口**，返回 `(loss, loss_dict)`；已经支持 `reduction="none"` 用于 RA-BC 加权
  - `select_action(batch, ...)` → (line 325-350) **同步推理入口**，内部用 deque 管理 chunk，触发条件是 `_check_get_actions_condition()`（队列空）
  - `predict_action_chunk(batch, ...)` → (line 313-322) **异步/批量推理入口**，被 policy_server 调用
  - `_get_action_chunk(batch, noise=None, **kwargs)` → (line 276-304) 内部统一入口，调 `model.sample_actions`
- **PEFT/LoRA 默认 targets**：`_get_default_peft_targets`（line 495-504）已经预设：
  ```python
  target_modules = r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj
                       |model\.(state_proj|action_in_proj|action_out_proj
                                 |action_time_mlp_in|action_time_mlp_out))"
  ```

### 2.4 训练循环入口

- **位置**：`src/lerobot/scripts/lerobot_train.py:66` 起，`def update_policy(...)` 是单步更新；`def train(cfg, accelerator)` 在 line 165 是主入口
- **损失定义**：`loss, output_dict = policy.forward(batch)`（line 128，无加权）或 `per_sample_loss, output_dict = policy.forward(batch, reduction="none")`（line 113，加权场景）
- **已经内置 `sample_weighter`**：`update_policy` 的 line 102-126 已经支持 per-sample loss × weights，加和归一化。**这是 RobustVLA per-sample 权重 / UCB bandit 的天然挂载点。**

### 2.5 推理入口

- **同步**：`SmolVLAPolicy.select_action`（行 325），用 `deque(maxlen=n_action_steps)` 维护内部 chunk 队列，队列空就重新计算
- **异步**：`src/lerobot/async_inference/`：
  - `policy_server.py:PolicyServer._predict_action_chunk`（行 330-405）：5 阶段 pipeline：raw→preprocess→`predict_action_chunk`→postprocess→`TimedAction`
  - `robot_client.py:RobotClient.receive_actions`（行 269-360）：gRPC 接收，调 `_aggregate_action_queues` 把新 chunk 合并到 queue（默认聚合 = 用新动作覆盖）
  - `robot_client.py:RobotClient._aggregate_action_queues`（行 224-267）：**`aggregate_fn` 可注入**——这就是 AAC 截断后合并 chunk 的核心钩子
- **内置 RTC**：`VLAFlowMatching.sample_actions` 内部（行 860-872）根据 `self._rtc_enabled()` 调 `rtc_processor.denoise_step`，传 `prev_chunk_left_over` / `inference_delay` / `execution_horizon`。AAC 的"提前重新预测"机制可以走 RTC 的同一管线。

---

## 3. AAC（Adaptive Action Chunking）注入点

> AAC 的核心：在 flow-matching sampling 后，用 **N 次重采样的方差** 度量每一步动作的不确定度，自适应截断 chunk 长度 h* < H=50。

### 注入点 A1：`SmolVLAPolicy.predict_action_chunk` 增加 N 次重采样

- **文件:函数**：`modeling_smolvla.py:313` `SmolVLAPolicy.predict_action_chunk`
- **现状**：单次调 `_get_action_chunk(batch, noise)` 返回 1 个 chunk，形状 `(B, chunk_size=50, action_dim)`
- **修改方案**：增加新方法 `predict_action_chunk_aac`，并行 N 次（默认 N=20）采样，对每个时间步算样本协方差和高斯熵
- **伪代码**：
  ```python
  @torch.no_grad()
  def predict_action_chunk_aac(
      self, batch, n_samples=20, xi=0.05, h_min=5
  ) -> tuple[Tensor, int, Tensor]:
      self.eval()
      batch = self._prepare_batch(batch)
      self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

      # 1) 准备 batch 复制 N 份（B*N）以便单次 forward
      images, img_masks = self.prepare_images(batch)
      state = self.prepare_state(batch)
      lang_tokens = batch[OBS_LANGUAGE_TOKENS]; lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
      bsize = state.shape[0]; H = self.config.chunk_size; D = self.config.max_action_dim
      # 给每个样本采 N 份 noise
      noise = self.model.sample_noise((bsize*n_samples, H, D), state.device)
      state_rep = state.repeat_interleave(n_samples, dim=0)
      # ... images/lang 同样 repeat_interleave

      chunks = self.model.sample_actions(images_rep, ..., state_rep, noise=noise)
      chunks = chunks.view(bsize, n_samples, H, D)

      # 2) per-step 高斯微分熵 H_t = 0.5 * log det(2*pi*e * Sigma_t)
      cov = chunks.var(dim=1) + 1e-8                          # (B, H, D),对角近似
      entropy = 0.5 * torch.log(2*math.pi*math.e * cov).sum(-1)  # (B, H)

      # 3) 检测熵跳变 E[h+1] - E[h] > xi
      delta = entropy[:, 1:] - entropy[:, :-1]                # (B, H-1)
      mask = (delta > xi).cumsum(dim=-1) == 0                 # 截断前
      h_star = mask.sum(dim=-1).clamp(min=h_min).int() + 1    # (B,)

      # 4) 返回 mean chunk 的前 h_star 步（或取第 0 个样本）
      mean_chunk = chunks.mean(dim=1)                         # (B, H, D)
      return mean_chunk, h_star, entropy
  ```
- **影响异步**：返回 `h_star` 后，server 端需把 chunk 截短到 `h_star`，client 队列加速消费

### 注入点 A2：将 AAC 嵌入 `_get_action_chunk` 的 kwargs 管线

- **文件:函数**：`modeling_smolvla.py:276` `SmolVLAPolicy._get_action_chunk`
- **现状**：单次 `self.model.sample_actions(...)`
- **修改方案**：扩展 `ActionSelectKwargs`（行 76）加 `aac_n_samples: int`, `aac_xi: float`；在 `_get_action_chunk` 内做 N-shot 采样并裁剪
- **伪代码**：
  ```python
  def _get_action_chunk(self, batch, noise=None, **kwargs):
      ...
      n = kwargs.pop("aac_n_samples", 1)
      xi = kwargs.pop("aac_xi", None)
      if n > 1:
          actions, h_star, _ = self._sample_n_and_truncate(
              images, img_masks, lang_tokens, lang_masks, state, n, xi, **kwargs
          )
          # 仅保留 h_star 步, 其余位置以 NaN/掩码标记
          ...
      else:
          actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)
      return actions[:, :, :self.config.action_feature.shape[0]]
  ```

### 注入点 A3：在 `VLAFlowMatching.sample_actions` 内并行化 N 次去噪

- **文件:函数**：`modeling_smolvla.py:812` `VLAFlowMatching.sample_actions`
- **现状**：bsize 维度做去噪，KV cache 在循环外计算一次复用
- **修改方案**：把 N 折叠进 batch 维度，**prefix KV cache 共享**（只算一次），显著省时间
- **伪代码**：
  ```python
  def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                     noise=None, n_samples: int = 1, **kwargs):
      bsize = state.shape[0]
      # prefix 只算一次
      prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
      _, past_kv = self.vlm_with_expert.forward(..., fill_kv_cache=True)
      # KV cache 复制 N 份(沿 batch 维)
      past_kv = self._replicate_kv_cache(past_kv, n_samples)
      prefix_pad_masks = prefix_pad_masks.repeat_interleave(n_samples, 0)
      # noise 是 (B*N, H, D)
      if noise is None:
          noise = self.sample_noise(
              (bsize*n_samples, self.config.chunk_size, self.config.max_action_dim),
              state.device)
      x_t = noise
      for step in range(self.config.num_steps):  # 10
          v_t = self.denoise_step(prefix_pad_masks, past_kv, x_t, timestep=...)
          x_t = x_t + dt * v_t
      return x_t.view(bsize, n_samples, self.config.chunk_size, -1)
  ```
- **风险**：KV cache 复制内存 ×N，VLM 在 SmolVLM2-500M 下 prefix ~150 token × hidden 960 → 每样本 KV 增量约 60MB；N=20 时 +1.2GB，需 12-16GB 显存才稳

### 注入点 A4：熵估计与自适应截断工具函数（新增）

- **文件:函数**：`modeling_smolvla.py` 顶部新增 `def adaptive_chunk_size(entropy, xi, h_min=5)`
- **现状**：无
- **修改方案**：抽离 AAC 截断逻辑为独立函数（便于消融实验），输入 `entropy: Tensor[B, H]`，输出 `h_star: Tensor[B]`
- **伪代码**：
  ```python
  def adaptive_chunk_size(entropy: Tensor, xi: float, h_min: int = 5,
                          h_max: int | None = None) -> Tensor:
      """根据高斯熵增量检测截断点 h*。entropy: (B, H), xi: 阈值."""
      delta = entropy[:, 1:] - entropy[:, :-1]                # (B, H-1)
      # 第一个 delta > xi 的位置
      exceeded = (delta > xi).int()
      first_exceed = exceeded.argmax(dim=-1)                  # (B,)
      # 若全 0,说明从未超阈值,用 H
      never = exceeded.sum(-1) == 0
      h_star = torch.where(never, torch.full_like(first_exceed, entropy.shape[1]),
                            first_exceed + 1)
      h_star = h_star.clamp(min=h_min, max=h_max or entropy.shape[1])
      return h_star
  ```

### 注入点 A5：与 RTC 协同——把 AAC 截断结果回填到 `prev_chunk_left_over`

- **文件:函数**：`modeling_smolvla.py:355` `_rtc_enabled` + `modeling_smolvla.py:867` 调用 `rtc_processor.denoise_step` 的位置
- **现状**：RTC 用 `prev_chunk_left_over`（剩余的旧 chunk）保证轨迹平滑，假设 chunk 长度固定 = H
- **修改方案**：当 AAC 把 chunk 截断到 h*，把 `[h*, H)` 范围标记为「需要新预测」，RTC 在下一轮 inference 时把这部分用新 noise 重新去噪而不是复用
- **伪代码**：
  ```python
  # 在 SmolVLAPolicy._get_action_chunk 末尾
  if aac_n_samples > 1 and self._rtc_enabled():
      # h_star: (B,) 标记每个样本的有效长度
      # 把超出部分的 prev_chunk_left_over 清零,告诉 RTC 这里不能复用
      self.rtc_processor.invalidate_beyond(h_star)
  ```

### 注入点 A6：异步推理 server 端按 `h*` 裁剪 TimedAction 列表

- **文件:函数**：`async_inference/policy_server.py:322` `PolicyServer._get_action_chunk`
- **现状**：`return chunk[:, : self.actions_per_chunk, :]` 固定截到 `actions_per_chunk`（client 配置）
- **修改方案**：调 `predict_action_chunk_aac` 拿到 `h_star`，按 `min(h_star, actions_per_chunk)` 截
- **伪代码**：
  ```python
  def _get_action_chunk(self, observation):
      if self.config.use_aac:
          chunk, h_star_b, _ = self.policy.predict_action_chunk_aac(
              observation, n_samples=self.config.aac_n_samples, xi=self.config.aac_xi)
          # batch size=1 在 server 上
          h_star = int(h_star_b.item())
          h_eff = min(h_star, self.actions_per_chunk)
          return chunk[:, :h_eff, :]
      else:
          chunk = self.policy.predict_action_chunk(observation)
          if chunk.ndim != 3:
              chunk = chunk.unsqueeze(0)
          return chunk[:, : self.actions_per_chunk, :]
  ```

### 注入点 A7：robot_client 提前触发下一次推理（must-go）

- **文件:函数**：`async_inference/robot_client.py:269` `RobotClient.receive_actions` 中的 `self.must_go.set()` 行 337，以及 `control_loop_action`
- **现状**：`must_go` 在每次收到新 chunk 时被 set，等队列空再触发下次推理
- **修改方案**：当本 chunk 是 AAC 截断版本（带 `h_star` metadata），到达 `h_star - margin` 步时主动 `must_go.set()`，让 policy_server 提前并行去算下一 chunk
- **伪代码**：
  ```python
  def control_loop_action(self, verbose=False):
      with self.action_queue_lock:
          self.action_queue_size.append(self.action_queue.qsize())
      ...
      # 新增:若 AAC 模式且队列剩余 < margin,提前催 server
      if self.config.use_aac and self.action_queue.qsize() <= self.config.aac_prefetch_margin:
          self.must_go.set()
      action = self._pop_next_action()
      return self._action_tensor_to_action_dict(action)
  ```

### 注入点 A8：配置项接入

- **文件:函数**：`configuration_smolvla.py:26` `SmolVLAConfig` dataclass + `async_inference/configs.py`
- **现状**：无 AAC 字段
- **修改方案**：增加 4 个字段
- **伪代码**：
  ```python
  # in SmolVLAConfig
  use_aac: bool = False
  aac_n_samples: int = 20
  aac_xi: float = 0.05       # 熵增量阈值
  aac_h_min: int = 5         # 最小 chunk 长度
  # in PolicyServerConfig
  use_aac: bool = False
  aac_n_samples: int = 20
  aac_xi: float = 0.05
  aac_prefetch_margin: int = 3
  ```

---

## 4. RobustVLA 注入点

> RobustVLA 的核心：**output robust** = PGD L∞ 找最坏 action noise η 让 flow matching loss 最大；**input robust** = 视觉/语言扰动后 action 应保持一致；多扰动用 SW-UCB 多臂老虎机选择。

### 注入点 R1：训练 loss 加 output-robust 项（动作 PGD）

- **文件:函数**：`modeling_smolvla.py:358` `SmolVLAPolicy.forward` + `modeling_smolvla.py:774` `VLAFlowMatching.forward`
- **现状**：单次 MSE flow matching loss（line 809）
- **参考实现**：`RobustVLA/openpi/src/openpi/models/pi0.py:254` `_compute_adversarial_action_loss`（JAX 版）
- **修改方案**：在 `SmolVLAPolicy.forward` 同时算 clean loss 与 PGD adversarial loss，加权相加
- **伪代码**：
  ```python
  def forward(self, batch, noise=None, time=None, reduction="mean"):
      ...
      images, img_masks = self.prepare_images(batch)
      state = self.prepare_state(batch); actions = self.prepare_action(batch)
      lang_tokens = batch[OBS_LANGUAGE_TOKENS]; lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

      # 1) clean flow matching
      clean_losses = self.model.forward(images, img_masks, lang_tokens, lang_masks,
                                         state, actions, noise, time)
      total_loss = clean_losses.mean() * self.config.clean_loss_weight

      # 2) RobustVLA action PGD
      if self.training and self.config.adv_training_action:
          eta = self._pgd_action_noise(
              images, img_masks, lang_tokens, lang_masks, state, actions,
              noise, time,
              epsilon=self.config.adv_epsilon_action,    # e.g. 0.03
              alpha=self.config.pgd_alpha_action,        # e.g. 0.01
              steps=self.config.pgd_steps_action,        # e.g. 3
          )
          adv_losses = self.model.forward(images, img_masks, lang_tokens, lang_masks,
                                           state, actions + eta, noise, time)
          total_loss = total_loss + adv_losses.mean() * self.config.action_loss_weight
      return total_loss, {"loss": total_loss.item(), ...}
  ```
- **显存影响**：每个 PGD step ≈ 1 次前向 + 1 次反向（只对 η 反传）。3 步 + clean 共 4 次前向，显存峰值 ≈ ×2~×2.5（无需保存所有 step 的激活，只保留最新）

### 注入点 R2：新增 `_pgd_action_noise` 方法

- **文件:函数**：`modeling_smolvla.py` 中 `SmolVLAPolicy` 类内新增
- **现状**：无
- **修改方案**：参照 `RobustVLA pi0.py:_compute_adversarial_action_loss`（行 254-288）用 PyTorch 重写
- **伪代码**：
  ```python
  def _pgd_action_noise(self, images, img_masks, lang_tokens, lang_masks,
                        state, actions, noise, time, *,
                        epsilon: float, alpha: float, steps: int) -> Tensor:
      """L∞ PGD: 找最坏的 action 扰动 η ∈ [-ε, ε]^A."""
      # 初始化:Bernoulli{-1,+1} * ε(参照 RobustVLA pi0.py:257)
      eta = (torch.randint(0, 2, actions.shape, device=actions.device,
                            dtype=actions.dtype) * 2 - 1) * epsilon
      eta = eta.detach().requires_grad_(True)
      for _ in range(steps):
          adv_losses = self.model.forward(
              images, img_masks, lang_tokens, lang_masks,
              state, actions + eta, noise, time)
          loss = adv_losses.mean()
          grad = torch.autograd.grad(loss, eta, retain_graph=False)[0]
          with torch.no_grad():
              eta = eta + alpha * grad.sign()
              eta = eta.clamp_(-epsilon, epsilon)
          eta = eta.detach().requires_grad_(True)
      return eta.detach()
  ```

### 注入点 R3：训练 loss 加 input-robust 项（图像 PGD / consistency）

- **文件:函数**：`modeling_smolvla.py:358` `SmolVLAPolicy.forward`
- **现状**：无图像扰动
- **参考实现**：`RobustVLA/openpi/src/openpi/models/pi0.py:291` `_compute_adversarial_image_loss`
- **修改方案**：对图像加 PGD 扰动 δ，要求扰动后预测 v_t 仍然贴近原 u_t
- **伪代码**：
  ```python
  if self.training and self.config.adv_training_image:
      delta = self._pgd_image_noise(
          images, img_masks, lang_tokens, lang_masks, state, actions,
          noise, time, epsilon=self.config.adv_epsilon_image)   # e.g. 8/255
      adv_images = [torch.clamp(img + d, -1, 1) for img, d in zip(images, delta)]
      adv_losses = self.model.forward(adv_images, img_masks, lang_tokens, lang_masks,
                                       state, actions, noise, time)
      total_loss = total_loss + adv_losses.mean() * self.config.img_loss_weight
  ```

### 注入点 R4：语言扰动注入（提示词替换）

- **文件:函数**：`processor_smolvla.py` 的 tokenize 阶段；或者更早在 `data_processing` 阶段
- **现状**：tokenizer 直接 encode 原始 task 字符串
- **参考实现**：`RobustVLA/openpi/src/openpi/prompt_augmentation.py:PromptAugmenter`
- **修改方案**：训练时用 JSON 字典做随机同义替换；增加 `PromptAugmenter` 钩子到数据预处理 pipeline
- **伪代码**：
  ```python
  # processor_smolvla.py 加一个 step
  class SmolVLAProcessor:
      def __init__(self, ..., prompt_augmenter: PromptAugmenter | None = None):
          self.prompt_augmenter = prompt_augmenter
      def _process_language(self, task: str, train: bool) -> str:
          if train and self.prompt_augmenter is not None:
              task = self.prompt_augmenter.augment_prompt(task)
          return task
  ```

### 注入点 R5：UCB 多臂老虎机选扰动模态

- **文件:函数**：训练步前的 batch 预处理，挂在 `lerobot_train.py:165` `train` 主循环的 dataloader 旁
- **现状**：无
- **参考实现**：`RobustVLA/openpi/src/openpi/robust_util/ucb_augmentation_balancer.py:UCBAugmentationBalancer`（449 行，11 个 arms = 7 视觉 + 3 语言 + 1 no_aug）
- **修改方案**：每个 step 先 UCB 选 arm，应用对应扰动再算 loss，loss 改善量回灌 UCB 统计
- **伪代码**：
  ```python
  # in train() main loop
  ucb = UCBAugmentationBalancer(window_size=100, exploration_coeff=1.0, ...)

  for step in range(num_steps):
      batch = next(dl_iter)
      arm = ucb.select_arm()                                # 选扰动类型
      aug_batch = ucb.apply_augmentation(batch, arm)        # 应用到 batch
      loss_before = policy.forward(batch)[0].item()
      loss_after  = policy.forward(aug_batch)[0].item()
      improvement = loss_before - loss_after                # 负值代表扰动让 loss 上升
      ucb.update(arm, reward=-improvement)                  # 鼓励让模型变难的扰动
      # 真正训练步用 aug_batch
      train_metrics, _ = update_policy(train_metrics, policy, aug_batch, optimizer, ...)
  ```

### 注入点 R6：把 RobustVLA loss 接入 `update_policy` 的 sample_weighter

- **文件:函数**：`scripts/lerobot_train.py:102-128`
- **现状**：`sample_weighter` 存在，但默认未启用；`policy.forward(batch, reduction="none")` 已实现，返回 per-sample loss
- **修改方案**：实现 `RobustVLASampleWeighter`（参照 `sample_weighter` 接口），其 `compute_batch_weights` 返回基于每个样本最坏 η 范数的权重——高 η 范数样本权重大（hard mining）
- **伪代码**：
  ```python
  class RobustVLASampleWeighter:
      def __init__(self, alpha=1.0):
          self.alpha = alpha
      def compute_batch_weights(self, batch):
          # 复用 PGD 结果:η_norm 作为 hardness 信号
          # 与 update_policy 解耦,只返回权重张量
          ...
          weights = torch.softmax(self.alpha * eta_norm_per_sample, dim=0) * batch_size
          stats = {"mean_weight": weights.mean().item(), "max_weight": weights.max().item()}
          return weights, stats
  ```

### 注入点 R7：配置项（`SmolVLAConfig` 扩展）

- **文件:函数**：`configuration_smolvla.py:26`
- **现状**：无
- **修改方案**：增加 RobustVLA 字段
- **伪代码**：
  ```python
  # in SmolVLAConfig
  # ---- RobustVLA output robust ----
  adv_training_action: bool = False
  adv_epsilon_action: float = 0.03   # L∞ 半径
  pgd_alpha_action: float = 0.01
  pgd_steps_action: int = 3
  action_loss_weight: float = 1.0
  clean_loss_weight: float = 1.0
  # ---- RobustVLA input robust ----
  adv_training_image: bool = False
  adv_epsilon_image: float = 0.0314  # ≈ 8/255
  pgd_alpha_image: float = 0.008
  pgd_steps_image: int = 3
  img_loss_weight: float = 0.5
  # ---- UCB ----
  use_ucb_balancer: bool = False
  ucb_window_size: int = 100
  ucb_exploration_coeff: float = 1.0
  prompt_augmentation_dir: str = "./prompts"
  ```

### 注入点 R8：双循环兼容性——内层 PGD 不更新主参数

- **文件:函数**：`modeling_smolvla.py:_pgd_action_noise`（R2 新增）
- **现状**：—
- **修改方案**：PGD 内层只对 η 求导，主网络参数保留计算图但不调 `optimizer.step()`；用 `torch.autograd.grad(..., create_graph=False, retain_graph=False)` 节省显存
- **伪代码**：
  ```python
  # 见 R2:torch.autograd.grad(loss, eta, retain_graph=False)
  # 关键点:η 在每个内层 step 重新 detach().requires_grad_(True)
  # 主参数的 grad 不会被累积,因为外层 loss 是从 actions+η.detach() 重新算的
  ```

---

## 5. LoRA attach 建议

### 5.1 哪些 layer 加 LoRA

SmolVLA 已经在 `_get_default_peft_targets`（`modeling_smolvla.py:495-504`）给出默认 target 正则表达式：

```python
target_modules = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj"   # 动作专家所有层的 QV
    r"|model\.(state_proj|action_in_proj|action_out_proj"   # 3 个状态/动作 projector
    r"|action_time_mlp_in|action_time_mlp_out))"            # 2 个时间-动作融合 MLP
)
```

- **覆盖范围**：动作专家所有层的 Q/V projection + 5 个轻量 projector（约占可训参数的 60-80%）
- **没有覆盖**：VLM backbone 的 self-attn（默认全冻），expert 的 O_proj / K_proj / MLP gate-up-down
- **建议 rank**：曦源 r=16 与默认 PEFT preset 兼容。若要扩展到 K/O/MLP，可改为：

```python
target_modules = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|k|v|o)_proj"
    r"|model\.vlm_with_expert\.lm_expert\..*\.(gate|up|down)_proj"
    r"|model\.(state_proj|action_in_proj|action_out_proj"
    r"|action_time_mlp_in|action_time_mlp_out))"
)
```

### 5.2 与 PEFT 库的集成

- **`PreTrainedPolicy` 父类已经有 `_validate_peft_config`**（见 `modeling_smolvla.py:506-515`）和 `_get_default_peft_targets` 钩子，PEFT 已经是 in-tree 支持
- **配置示例**：
  ```python
  from peft import LoraConfig
  peft_config = LoraConfig(
      r=16, lora_alpha=32, lora_dropout=0.05,
      bias="none", task_type="FEATURE_EXTRACTION",
      target_modules=policy._get_default_peft_targets()["target_modules"],
      modules_to_save=policy._get_default_peft_targets()["modules_to_save"],
  )
  policy.model = get_peft_model(policy.model, peft_config)
  ```

### 5.3 与 RobustVLA worst-case 双循环的兼容性

- **内循环找 η 时**：所有参数（包括 LoRA A/B）都 freeze，只对 η 求导。`_pgd_action_noise`（R2）内部 `torch.autograd.grad(loss, eta, ...)` 只指定 η 作为 grad target，**不会更新 LoRA 参数**
- **外循环更新参数时**：只 LoRA A/B（rank 矩阵）参与 `optimizer.step()`，PEFT 会自动把非-LoRA 参数的 `requires_grad=False`。RobustVLA 的 `clean_loss + adv_loss` 加权和正常 backward
- **风险**：PGD 内层每步算图必须 retain 主网络计算图，与 LoRA 兼容；但若 N=20 + PGD steps=3 + 动作 expert 16 层，激活峰值显著，需 grad checkpointing
- **建议**：先用 `freeze_vision_encoder=True, train_expert_only=True` + LoRA r=16 跑 RobustVLA，确认显存预算后再考虑是否扩 K/O proj

---

## 6. 异步推理 stack 集成方式

### 6.1 异步推理的代码位置

- **核心模块**：`src/lerobot/async_inference/`
  - `policy_server.py` (439 行)：gRPC 服务端，包 `PolicyServer(services_pb2_grpc.AsyncInferenceServicer)` (line 64)
  - `robot_client.py` (517 行)：gRPC 客户端，`RobotClient` (line 83) 维护本地 `action_queue: Queue`
  - `helpers.py` (297 行)：`raw_observation_to_observation` 等辅助
  - `configs.py`：`PolicyServerConfig` + `RobotClientConfig`

- **架构**（精读结论）：
  ```
  Robot (RobotClient)         Server (PolicyServer)
      │                              │
      │  send_observation (gRPC)     │
      ├─────────────────────────────►│
      │                              │  PolicyServer.SendObservations
      │                              │      └─> _predict_action_chunk
      │                              │          └─> policy.predict_action_chunk
      │  GetActions (gRPC streaming) │              └─> sample_actions (flow match × 10 steps)
      │◄─────────────────────────────┤
      │  receive_actions             │
      │      └─> _aggregate_action_queues   ← 关键钩子
      │          └─> action_queue.put
      │                              │
      │  control_loop_action         │
      │      └─> action_queue.get   │
      │      └─> robot.send_action  │
  ```

### 6.2 AAC 截断如何 hook 到 async queue

**信号流**：当 AAC 决定 h* < H 时：

1. **server 侧**（`policy_server.py:_get_action_chunk`）按 h* 裁剪 TimedAction 列表，仅发送 h* 个动作给 client
2. **client 侧**（`robot_client.py:_aggregate_action_queues`）正常合并到 queue；由于 chunk 短，queue 会更快被消费完
3. **触发提前预测**：在 `control_loop_action`（line 370）增加 prefetch 逻辑——当 `qsize ≤ aac_prefetch_margin` 时 set `must_go` 事件，server 看到事件就立刻收下一帧 obs 算下个 chunk
4. **`aggregate_fn` 协同 RTC**：当新 chunk 到达且与现有 queue 重叠（client 还没消费完旧的），`_aggregate_action_queues` 用 `aggregate_fn(old, new)` 合并；可注入「指数平滑」或 RTC 的 partial-overwrite 策略

**代码修改位置汇总**：

| 修改点 | 文件:函数 | 行号 | 内容 |
|---|---|---|---|
| AAC 推理入口 | `modeling_smolvla.py:SmolVLAPolicy` | 新增 | `predict_action_chunk_aac()` |
| Server 截断 | `policy_server.py:PolicyServer._get_action_chunk` | 322 | 按 h* 裁剪 |
| Client prefetch | `robot_client.py:RobotClient.control_loop_action` | 370 | `must_go` 触发条件 |
| 配置注入 | `async_inference/configs.py:PolicyServerConfig` | — | `use_aac, aac_n_samples, aac_xi` |
| RTC 协同 | `policies/rtc/rtc_processor.py` | — | `invalidate_beyond(h_star)` |

---

## 7. 实现工作量估算

| 模块 | 修改/新增行数 | 难度 | 风险 |
|------|---------|------|------|
| **AAC 集成** |  |  |  |
| - `predict_action_chunk_aac` + `_sample_n_and_truncate` (A1, A2) | ~80 行 | 中 | xi 阈值与 h_min 需要调优 |
| - `VLAFlowMatching.sample_actions` 并行 N 次重写 (A3) | ~50 行 | 中 | KV cache 复制内存 ×N |
| - `adaptive_chunk_size` 工具函数 (A4) | ~20 行 | 低 | — |
| - RTC 协同 + invalidate (A5) | ~30 行 | 中 | 需理解 RTC 内部 |
| - server 截断 + client prefetch (A6, A7) | ~40 行 | 中 | 多线程时序，需端到端测试 |
| - 配置项 (A8) | ~15 行 | 低 | — |
| **AAC 小计** | **~235 行** | — | — |
| **RobustVLA 集成** |  |  |  |
| - `_pgd_action_noise` (R2) | ~40 行 | 中 | 显存 +30%~+100% |
| - `_pgd_image_noise` (R3) | ~50 行 | 中 | 与 R2 类似 |
| - `forward` 改造 (R1) | ~30 行 | 中 | — |
| - 语言扰动 hook (R4) | ~30 行 | 低 | 依赖 prompts 字典 |
| - UCB balancer 移植到 PyTorch (R5) | ~250 行 | 中-高 | JAX→PyTorch 重写 |
| - `RobustVLASampleWeighter` (R6) | ~50 行 | 低 | 接口已存在 |
| - 配置项 (R7) | ~25 行 | 低 | — |
| **RobustVLA 小计** | **~475 行** | — | — |
| **LoRA wrapper** | ~30 行 | 低 | PEFT 已成熟，默认 target 已给 |
| **异步推理 hook** | ~50 行 | 中-高 | gRPC 协议 + 多线程时序 |
| **测试与消融** | ~150 行 | — | 必需 |
| **合计** | **~940 行** | — | — |

---

## 8. 关键依赖与版本

- `huggingface/lerobot` main 分支 SHA：`c0a2e981` (2026-05-22)
- `gakakulicc/RobustVLA` master 分支：fork 自 `Physical-Intelligence/openpi`（**JAX/Flax 实现**，需要全部 1:1 翻译到 PyTorch）
- `torch >= 2.4`（lerobot 要求，使用 `torch.compile`）
- `transformers >= 4.45`（SmolVLM2 支持）
- `peft >= 0.13`（LoraConfig + get_peft_model）
- `accelerate >= 0.34`（lerobot_train.py 使用 `accelerator.autocast/backward`）
- `grpcio` + `protobuf`（async_inference 用，已在 lerobot 依赖中）
- VLM 权重：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`（5.0 亿 VLM + ~0.45B 动作 expert = 0.95B 总参；但论文/项目主轴报告 SmolVLA 总参 0.45B，对应 `num_vlm_layers=16, expert_width_multiplier=0.75`，实际可调）

---

## 9. 风险盘点

1. **最大风险：RobustVLA 全套是 JAX/Flax**，从 `_compute_adversarial_action_loss` 到 `UCBAugmentationBalancer` 都需 1:1 翻译到 PyTorch；UCB 的 449 行需仔细处理（含 sliding window EMA、modality-floor 投影）
2. **显存：AAC N=20 + PGD steps=3** 同时启用时，峰值显存 ≈ baseline ×3，需要 24GB+ 显卡；建议 LoRA + gradient checkpointing
3. **AAC 与 RTC 互操作**：RTC 假设固定 chunk 长度，AAC 动态截断会让 RTC 的 `prev_chunk_left_over` 含无效尾部，需新增 `invalidate_beyond` API
4. **异步推理时序**：截断后过早 `must_go.set()` 会让 server queue 拥堵，需要 prefetch_margin 调参
5. **PEFT + RobustVLA 双循环**：PGD 内层算图保留，激活显存可能炸；需 grad checkpointing + 较小 PGD steps（≤3）

---

## 参考

- lerobot 仓库 https://github.com/huggingface/lerobot
- lerobot SmolVLA 模型源码 https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla
- lerobot 异步推理 stack https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference
- RobustVLA 仓库 https://github.com/gakakulicc/RobustVLA
- RobustVLA pi0 实现 https://github.com/gakakulicc/RobustVLA/blob/master/openpi/src/openpi/models/pi0.py
- RobustVLA UCB 实现 https://github.com/gakakulicc/RobustVLA/blob/master/openpi/src/openpi/robust_util/ucb_augmentation_balancer.py
- 一次调研 [[01_方向一_动态自适应推理]] [[03_方向三_鲁棒性微调]]
