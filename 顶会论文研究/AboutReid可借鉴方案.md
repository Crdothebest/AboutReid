# AboutReid 可借鉴方案 — 基于顶会论文的改进路线

**参考论文**: TOP-ReID (AAAI 2024) | IDEA (CVPR 2025) | CLIP-ReID (AAAI 2023) | ShapeSpeak (arXiv 2025)  
**当前基线**: IDEA02，mAP=66.8%，Rank-1=71.8%（5 epoch），推测 60 epoch 可达 ~75–78%  
**不改动的部分**: Mamba/MM_SS2D 结构、AAM 结构

---

## 当前方案的问题诊断

| 问题 | 描述 | 来源分析 |
|------|------|----------|
| 文本融合是"浅层"的 | 文本只在 AAM 输出后加权，不影响 ViT 编码过程 | 对比 IDEA 的 InverseNet 深度融合 |
| alignment loss 太弱 | 逐样本 cosine 距离不能利用批内负样本 | 对比 CLIP-ReID 的 InfoNCE |
| 文本过滤器未区分模态语义 | NIR/TIR 的 text_adapter 可能学到颜色维度 | 对比 ShapeSpeak 的模态自适应过滤 |
| 无跨模态重建监督 | 模型没有被迫学习跨模态对应关系 | 对比 TOP-ReID 的 CRM |

---

## 改进方案（按优先级排序）

### 方案 A: 升级 Alignment Loss 为 InfoNCE（推荐优先实施）

**改动位置**: `engine/processor.py`，`modeling/make_model.py`  
**改动难度**: ★☆☆☆☆（极低）  
**预期收益**: mAP +1–2%  
**理论依据**: CLIP-ReID 证明 InfoNCE 比 pair-wise cosine 距离更有效

**当前代码**（processor.py 中）:
```python
# 当前: 逐样本 cosine alignment
align_loss = 1.0 - (v * t).sum(dim=-1).mean()
```

**修改为** InfoNCE:
```python
def compute_info_nce_loss(visual_feat, text_feat, temperature=0.07):
    """批内对比: 正样本是同一行人的 (视觉, 文本) 对"""
    B = visual_feat.size(0)
    v = F.normalize(visual_feat, dim=-1)   # [B, 512]
    t = F.normalize(text_feat, dim=-1)     # [B, 512]
    
    logits = v @ t.T / temperature         # [B, B]
    labels = torch.arange(B, device=v.device)
    
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2.0
```

在 make_model.py 中，对 RGB/NIR/TIR 三路分别计算，最终 text_align_loss 取平均。

---

### 方案 B: 模态自适应文本过滤器（中等优先）

**改动位置**: `modeling/make_model.py`  
**改动难度**: ★★☆☆☆  
**预期收益**: mAP +1–2%（尤其对 NIR/TIR 提升明显）  
**理论依据**: ShapeSpeak 证明不同模态需要不同维度的文本信息

**当前实现**:
```python
self.text_adapters = nn.ModuleDict({
    'RGB': nn.Linear(512, 512),
    'NIR': nn.Linear(512, 512),
    'TIR': nn.Linear(512, 512),
})
```

**升级为带门控的过滤器**:
```python
class ModalTextFilter(nn.Module):
    def __init__(self, text_dim=512, feat_dim=512, dropout=0.1):
        super().__init__()
        # 门控: 学习哪些文本维度对当前模态有效
        self.gate = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Sigmoid()
        )
        # 变换: 将过滤后的文本投影到特征空间
        self.transform = nn.Sequential(
            nn.Linear(text_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, text_feat):
        gate_weights = self.gate(text_feat)          # [B, 512] ∈ [0,1]
        filtered_text = text_feat * gate_weights      # 维度级别过滤
        return self.transform(filtered_text)          # [B, feat_dim]

# 在 MambaPro.__init__ 中:
self.text_adapters = nn.ModuleDict({
    'RGB': ModalTextFilter(512, 512),
    'NIR': ModalTextFilter(512, 512),
    'TIR': ModalTextFilter(512, 512),
})
```

**核心差别**: 原来是 `nn.Linear`（全通，不过滤），现在是 `gate * text`（维度级别选择性过滤）

---

### 方案 C: 预 AAM 跨模态文本注意力（高收益，中等改动）

**改动位置**: `modeling/make_model.py`（在 AAM 之前插入）  
**改动难度**: ★★★☆☆  
**预期收益**: mAP +2–4%（核心改动，最有潜力）  
**理论依据**: TOP-ReID 的跨模态交互 + IDEA 的深度文本注入

**设计思路**: 在三路视觉 token 进入 AAM 之前，用文本特征做一次 cross-attention 增强，让视觉 token 带着文本语义进入 Mamba 融合。

```
RGB/NIR/TIR 视觉 token [B, 129, 512]
         ↓
 轻量跨模态文本注意力 (新增)
  - 文本特征作为 K/V
  - 视觉 token 作为 Q
         ↓
 文本增强的视觉 token [B, 129, 512]
         ↓
 原有 AAM (Mamba, 不改动)
         ↓
 融合输出
```

**关键模块代码**:
```python
class PreAAMTextAttention(nn.Module):
    """轻量单头跨模态注意力，文本为 K/V，视觉 token 为 Q"""
    def __init__(self, dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_proj = nn.Linear(dim, dim)  # 文本维度适配
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))  # 初始化小值，稳定训练
    
    def forward(self, visual_tokens, text_feat):
        # visual_tokens: [B, 129, 512]
        # text_feat: [B, 512] → 扩展为 [B, 1, 512]
        text_kv = self.text_proj(text_feat).unsqueeze(1)  # [B, 1, 512]
        
        attended, _ = self.cross_attn(
            query=visual_tokens,  # [B, 129, 512]
            key=text_kv,          # [B, 1, 512]
            value=text_kv         # [B, 1, 512]
        )
        # 残差连接 + 可学习缩放
        return self.norm(visual_tokens + self.scale * attended)
```

在 `MambaPro.__init__` 中:
```python
if self.use_text_fusion:
    self.pre_aam_text_attn = nn.ModuleDict({
        'RGB': PreAAMTextAttention(512, 4),
        'NIR': PreAAMTextAttention(512, 4),
        'TIR': PreAAMTextAttention(512, 4),
    })
```

在 `MambaPro.forward` 中（AAM 之前）:
```python
if self.use_text_fusion and text_features is not None:
    RGB_vis = self.pre_aam_text_attn['RGB'](RGB_vis, text_features['RGB'])
    NI_vis = self.pre_aam_text_attn['NIR'](NI_vis, text_features['NIR'])
    TI_vis = self.pre_aam_text_attn['TIR'](TI_vis, text_features['TIR'])
# 然后送入 AAM
```

---

### 方案 D: 跨模态重建辅助 Loss（可选）

**改动位置**: `engine/processor.py`，`modeling/make_model.py`  
**改动难度**: ★★★☆☆  
**预期收益**: mAP +0.5–1.5%（收益较小但稳定）  
**理论依据**: TOP-ReID CRM + ShapeSpeak TVCR

在 make_model.py 的 forward 中，用 RGB 全局特征预测 NIR 全局特征：
```python
# 在 MambaPro 中新增
self.cross_modal_reconstructor = nn.ModuleDict({
    'RGB_to_NIR': nn.Linear(512, 512),
    'RGB_to_TIR': nn.Linear(512, 512),
    'NIR_to_RGB': nn.Linear(512, 512),
})

# forward 中
self.cross_recon_loss = None
if training:
    pred_ni = self.cross_modal_reconstructor['RGB_to_NIR'](RGB_global)
    pred_ti = self.cross_modal_reconstructor['RGB_to_TIR'](RGB_global)
    self.cross_recon_loss = (
        F.mse_loss(pred_ni, NI_global.detach()) +
        F.mse_loss(pred_ti, TI_global.detach())
    ) / 2
```

---

## 实施路线图

```
优先级 1 (最快见效):
  方案 A: InfoNCE loss 替换 cosine loss
  → 只改 processor.py 中约 10 行代码
  → 预期 mAP +1–2%

优先级 2 (中等代价高回报):
  方案 B: 模态自适应文本过滤器
  → 改 make_model.py 中 text_adapters 定义
  → 预期 mAP +1–2%

优先级 3 (核心创新，毕设亮点):
  方案 C: 预 AAM 文本注意力
  → 新增 PreAAMTextAttention 模块
  → 不改动 Mamba/AAM 结构
  → 预期 mAP +2–4%

优先级 4 (可选增强):
  方案 D: 跨模态重建 loss
  → 追加辅助 loss
  → 预期 mAP +0.5–1.5%
```

**累积预期收益**: 方案 A+B+C 组合，预计 mAP 从当前 66.8%（5 epoch）提升至 **72–74%**（5 epoch）。  
60 epoch 跑完后，对比基线 ~75%，目标可达 **80–83%**。

---

## 对比表：当前 vs 改进后

| 模块 | 当前 | 改进后（方案 A+B+C） |
|------|------|---------------------|
| Alignment Loss | cosine (逐样本) | InfoNCE (批内对比) |
| 文本过滤 | nn.Linear (全通) | ModalTextFilter (门控过滤) |
| 文本注入位置 | AAM 输出后（浅层） | AAM 输入前（深层，方案 C） + AAM 输出后 |
| 文本注入机制 | 加法 (additive) | Cross-Attention + 加法（双层） |
| 跨模态监督 | 无 | 可选重建 loss（方案 D） |

---

## 注意事项

1. **方案 C 新增参数量**: 3 个 PreAAMTextAttention 模块，每个约 2×512×512 = 0.5M 参数，共 1.5M，相对主干 86M 影响极小。
2. **训练稳定性**: `scale = nn.Parameter(tensor(0.1))` 初始化确保新模块在训练初期贡献极小，不破坏预训练权重的作用。
3. **不改动 Mamba**: 所有改动都在 Mamba/AAM 的上下游，避免破坏已调优的 Mamba 结构。
4. **温度参数**: InfoNCE 的 temperature=0.07 是 CLIP 原始设置，可根据验证集调整为 0.1–0.2。
