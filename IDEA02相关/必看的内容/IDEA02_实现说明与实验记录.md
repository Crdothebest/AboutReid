# IDEA02 实现说明与实验记录

> 记录时间：2026-04-10
> 基于 AboutReid 项目实际运行结果

---

## 一、IDEA02 是什么

IDEA02 = **两阶段文本语义引导**，在 IDEA01（多尺度滑动窗口 + MoE）的基础上叠加文本先验：

- **训练阶段**：跨模态文本融合（CMTF）——用文本信号统一三模态的全局语义表征
- **推理阶段**：模态内语义引导（IMSG）——用文本信号对每个模态的视觉特征做针对性微调

---

## 二、文本特征：为什么用离线预编码

### 当前做法：离线预编码

运行一次 `tools/encode_texts.py`，把所有图像对应的文本描述预先编码为 512 维向量，存成 `_feat.pt` 文件：

```
data/datasets/RGBNT201/text/
├── train_RGB_feat.pt    # {jpg文件名: tensor[512]}
├── train_NI_feat.pt
├── train_TI_feat.pt
├── test_RGB_feat.pt
├── test_NI_feat.pt
└── test_TI_feat.pt
```

训练时 DataLoader 直接读这些文件，每个样本返回固定的 `[512]` 向量，**CLIP 文本编码器不参与训练**。

### 为什么要离线而不是实时编码

| 原因 | 说明 |
|------|------|
| **速度** | CLIP 文本编码器每次 forward 约 10ms，全量数据集每 epoch 要额外花几分钟；离线编码后运行时零开销 |
| **稳定性** | 文本向量固定，排除了文本编码随机性对训练的干扰 |
| **工程简洁** | DataLoader 只需读 Tensor，不需要在训练主循环中维护文本编码器的 device/dtype |

### 和 IDEA 原作者的区别

原作者使用 **可学习 Prompt Tuning**：在文本描述前插入 4 个可学习 token（`X X X X`），这 4 个 token 的 embedding 在训练过程中由 ReID 损失函数（ID Loss + Triplet Loss）持续更新，相当于让文本向量朝着「对跨模态匹配最有利」的方向偏移。

```
原作者：f("★★★★ 穿白衣服") → 向量随训练变化，每 batch 必须重新编码
我们：  f("X X X X 穿白衣服") → 向量固定，训练前编码一次
```

**当前选择离线方案的原因**：
- 可学习 Prompt 需要把文本编码器接入训练图，改动 DataLoader、Optimizer、模型 forward，工程量大
- 实测证明：固定的模态感知前缀（`in the visible spectrum...` 等）已经能提供足够的语义信号
- 66.8% 的结果验证了当前方案的有效性

可学习 Prompt 可作为未来改进方向，预期能进一步提升性能。

---

## 三、训练阶段 CMTF 详解

**位置**：AAM 三模态融合之后，BNNeck 之前

**流程**：

```
三路文本各自通过独立适配器（512 → 1536）
         ↓
三路结果取平均 → text_modulator [B, 1536]
         ↓
gate = sigmoid(text_modulator)          # 每维压到 0~1
fuse = fuse + 0.3 × fuse × gate         # 门控残差增强
```

**为什么三路文本用独立适配器而非共享**：
RGB/NIR/TIR 三种模态的文本描述侧重点完全不同——RGB 描述颜色，NIR 描述纹理，TIR 描述温度。共享适配器会丢失这种差异性，独立适配器保留了各模态文本的语义特异性。

**融合方法选择 residual 而非 attention 的原因**：
- `attention` 模式包含两个 MultiheadAttention + 一个 MLP，参数量大，5 epoch 收敛慢
- `residual` 模式只有一个 2 层 MLP（512→256→512），结构简单，5 epoch 即可充分收敛
- 实测：residual 模式 5 epoch 达到 66.8%，attention 模式 5 epoch 只有 28.6%

---

## 四、推理阶段 IMSG 详解

**位置**：AAM 融合之前，逐模态独立处理

**流程（以 RGB 为例）**：

```
RGB_cash [B, 129, 512]（CLIP patch tokens）
    ↓ 取 CLS token → [B, 512]
    → LayerNorm → visual_normed

RGB文本 [B, 512]
    → LayerNorm → text_adapter（Linear）→ text_aligned

concat([visual_normed, text_aligned]) [B, 1024]
    → 2层MLP → Sigmoid → guidance [B, 512]
    → unsqueeze(1) → [B, 1, 512]（广播到序列维度）

RGB_enhanced = RGB_cash + RGB_cash × guidance × 0.1
```

NIR、TIR 同理，三路**互不干扰**。

**关键设计选择**：

| 设计 | 原因 |
|------|------|
| 每模态用自己的文本 | 不同模态文本语义不同，混用会互相干扰 |
| 增强幅度 0.1（极小） | 推理时文本质量未知，宁可少改不能改错 |
| 在 AAM 融合前处理 | 融合后无法单独校正退化模态；融合前逐模态精准补偿 |
| 门控基于 CLS token 生成后广播 | CLS token 是全局语义代表，用它生成门控信号，再作用到全序列 |

---

## 五、修复的 Bug（2026-04-10）

在实现过程中发现 IMSG 完全未生效，根本原因是 4 个 bug：

### Bug 1：`make_dataloader.py` — val_loader 未加载文本特征

```python
# 修复前
val_set = RGBNT201DatasetWrapper(val_data, val_transforms, use_text_features=True)
# 修复后
_val_feat_dir = os.path.join(root_dir, 'datasets', 'RGBNT201', 'text')
val_set = RGBNT201DatasetWrapper(val_data, val_transforms, use_text_features=True, feat_dir=_val_feat_dir)
```

另外 val_collate_fn 中文本列表未转为 Tensor，导致 LayerNorm crash：

```python
# 修复前
text_features = {'RGB': list(text_rgbs), ...}
# 修复后
text_features = {'RGB': torch.stack(list(text_rgbs)), ...}  # 若元素为 Tensor
```

### Bug 2：`engine/processor.py` — 验证循环未传 text_features 给模型

```python
# 修复前
feat = model(img, cam_label=camids, view_label=target_view)
# 修复后
feat = model(img, cam_label=camids, view_label=target_view, text_features=text_features)
```

### Bug 3：`modeling/make_model.py` — SafeModalGuidance 不支持 3D 输入

CLIP backbone 输出 `[B, seq_len, 512]`，SafeModalGuidance 原来只处理 `[B, 512]`：

```python
# 修复后：取 CLS token 生成门控信号，再广播回序列维度
is_seq = visual_feat.dim() == 3
cls_feat = visual_feat[:, 0] if is_seq else visual_feat
# ... 生成 guidance [B, 512] ...
if is_seq:
    guidance = guidance.unsqueeze(1)  # [B, 1, 512] 广播
```

拼接前也加了维度保护：

```python
def _get_global(feat):
    return feat[:, 0] if feat.dim() == 3 else feat
fuse = torch.cat([_get_global(RGB_enhanced), _get_global(NI_enhanced), _get_global(TI_enhanced)], dim=-1)
```

### Bug 4：`cross_modal_attention.py` — residual 模式收到多余参数

```python
# 修复前
return TextResidualFusion(**kwargs)  # input_dim 传入但 TextResidualFusion 不接受
# 修复后
residual_kwargs = {k: v for k, v in kwargs.items() if k != 'input_dim'}
return TextResidualFusion(**residual_kwargs)
```

---

## 六、实验结果

### 最终对比（RGBNT201，5 epoch）

| 方法 | mAP | Rank-1 | Rank-5 | Rank-10 |
|------|-----|--------|--------|---------|
| IDEA01（无文本） | 57.1% | 61.7% | — | — |
| IDEA02（CMTF，IMSG 未生效） | 28.6% | 27.4% | 49.0% | 62.0% |
| **IDEA02（CMTF + IMSG，全修复）** | **66.8%** | **71.8%** | **82.2%** | **87.4%** |

IMSG 修复前后从 28.6% → 66.8%，说明推理阶段模态内引导是主要贡献来源。

### 运行命令

```bash
python train_net.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    MODEL.USE_TEXT_FUSION True \
    MODEL.TEXT_FUSION_METHOD residual \
    MODEL.USE_MODAL_GUIDANCE True \
    DATASETS.USE_TEXT_FEATURES True \
    SOLVER.MAX_EPOCHS 60 \
    OUTPUT_DIR outputs/idea02_60ep
```

### 查看训练状态（其他终端）

```bash
# 连接服务器
ssh -p 22736 root@connect.cqa1.seetacloud.com

# 实时跟踪日志（过滤进度条）
tail -f /tmp/idea02_60ep.log | grep -v BATCH_GET

# 只看关键指标
grep -E "Epoch [0-9]+ done|mAP|Rank-1|Best mAP" /tmp/idea02_60ep.log

# 查看最新进度
grep -v "BATCH_GET\|batch/s\|batch]" /tmp/idea02_60ep.log | tail -20

# 确认进程还在运行
ps aux | grep train_net | grep -v grep

# GPU 状态
nvidia-smi
```

---

## 七、消融实验设计（待做）

按论文实验设计，还需要以下对照组：

| 组别 | USE_TEXT_FUSION | USE_MODAL_GUIDANCE | 目的 |
|------|----------------|-------------------|------|
| A | False | False | 纯 IDEA01 基线（已有：57.1%） |
| B | True | False | 验证 CMTF 单独贡献 |
| C | False | True | 验证 IMSG 单独贡献 |
| D | True | True | 完整 IDEA02（已有：66.8%） |

---

*文档生成时间：2026年4月10日*
