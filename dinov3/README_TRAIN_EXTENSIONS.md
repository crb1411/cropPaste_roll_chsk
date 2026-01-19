# DINOv3 新增训练逻辑说明（草稿）

本文用于记录我在 DINOv3 训练流程中新增/改造的关键逻辑，便于后续复盘与对外说明。
本次先完成第一部分：DTCH‑SK（双温度累计历史 Sinkhorn）。

---

## 1. DTCH‑SK（从 SK → CH‑SK → DTCH‑SK 的演进）

### 1.1 背景与演进脉络
最初版本只有标准 Sinkhorn（SK）。该算法在 `Q ∈ R^{B×K}` 上施加行/列归一化约束，
可理解为：
- **行约束**：每个样本的分配和为常数，保证样本内分配可比较；
- **列约束**：每个原型的总质量为常数，保证跨样本的原型均衡。

当 `K=65536` 且 `B` 很小（极端 `B=1`）时，列约束会把每个维度推向 `1/K`，
使得单样本分配趋于均匀，难以形成“尖锐”的输出。随着 `B` 增大，
列约束由更多样本共同承担，单样本就拥有更大的自由度去保持尖锐分配。
本质上，这是 **`B×K` 自由度不足** 与 **大 K 原型均衡** 之间的矛盾。

因此我先设计了 **SK‑Cache** 来模拟“更大 batch”，但缓存历史样本会显著占用显存。

随后改进为 **CH‑SK（Cumulative History SK）**：
- 不再保存完整历史样本，而仅维护 `history_Q ∈ R^K` 的累计统计；
- 以“累计列质量”替代显存昂贵的队列；
- 从 SK 到 CH‑SK，模型精度提升 **5%+**，且能在单机单卡上进行大规模预训练。

### 1.2 理论动机：尖锐性与均衡性的解耦
训练目标希望同时满足：
1) **单样本分配要尖锐**：尖锐分配带来更高的信息量和更强的区分性，降低模型坍塌风险；  
2) **跨样本统计要均衡**：每个原型在大量数据上被均衡使用，避免“少数原型独大”。

CH‑SK 虽然解决了小 batch 的均衡问题，但仍在 sharp 空间里更新 history。
这会导致单个 batch 的极大值对 history 产生过大冲击，
使得历史统计难以真正均衡（尤其在混合精度/设备差异下）。

因此我进一步提出 **DTCH‑SK（双温度累计历史 SK）**：
- 在一个更“软”的温度空间完成 SK 与 history 更新，使统计更平滑；  
- 在 SK 之后通过幂次恢复尖锐度，确保输出仍然“尖”。

这相当于把“均衡”与“尖锐”拆到两个不同的温度空间内完成。

### 1.3 具体实现（代码层面）
新增类 `DTCH_SK`，继承 `CH_SK`：
- 代码位置：`dinov3/loss/dtch_sk.py`
- 复用历史机制：`dinov3/loss/ch_sk.py`

主要步骤如下：
1. **强制 fp32 计算**
   - `teacher_output = teacher_output.float()`  
   - 避免 BF16 下的 exp/softmax 数值不稳定。

2. **软化温度用于 SK**
   - `logits_temp = teacher_output / (teacher_temp * dt_temp_scale)`
   - `exp` 后得到 `Q_batch_soft`（注意 clamp），用于 SK 与 history。

3. **history_Q 初始化/更新**
   - history 只用 `Q_batch_soft` 统计，避免被 boost 或过尖锐分布污染。
   - 若 checkpoint 不含 history_Q 或为 NaN，会自动用当前 batch 初始化。

4. **保持原有 boost 逻辑**
   - 仍然根据 history_Q 做 prototype boosting（`boost_alpha / boost_w_max / boost_threshold_divisor / boost_eps`）。
   - boost 只作用于 SK 输入，不影响 history 更新。

5. **SK 之后再做“再尖锐化”**
   - `Q_assign = Q[:, -B_batch:].t()`  
   - `Q_assign = Q_assign.pow(dt_exp_power)`  
   - 再按 K 维归一化，保证每个样本分配和为 1。

6. **返回值**
   - 返回 `Q_assign`，保持为 fp32（更稳定）。

### 1.4 配置项（可调）
在配置中新增/沿用以下参数：
- `ch_sk.dt_temp_scale`：SK 的温度缩放（>1 时更软）
- `ch_sk.dt_exp_power`：SK 后的幂次尖锐化强度
- `ch_sk.boost_*`：沿用 CH‑SK 的 boost 系列参数

### 1.5 集成位置
DTCH‑SK 被用于：
- `dinov3/loss/dino_cls_loss_cache_global.py`  
  DINO cls 的 Sinkhorn teacher
- `dinov3/loss/ibot_patch_loss.py`  
  iBOT patch 的 Sinkhorn teacher

即 **cls 与 patch 的 Sinkhorn 都统一走 DTCH‑SK**，从而稳定整个训练链路。

### 1.6 效果与预期
- history_Q 更稳定，减少数值漂移；
- SK 过程在多卡/不同设备上更稳；
- 仍保持 sharp assignment 的效果（通过 pow 再尖锐化）。

---

后续部分（crop‑resize loss、patch roll、bridge loss）将在下一步补充。
