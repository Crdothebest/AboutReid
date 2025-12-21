# 模态专家权重分析使用指南

## 🎯 实验目的

通过定量分析，验证模型在面对物理特性迥异的模态（RGB、NI、TI）时，是否具备**"因材施教"**的尺度选择能力。

---

## 📋 实验设计

### 数据采集方法

1. **加载模型**：使用验证集表现最好的权重文件（Best .pth）

2. **固定输入测试**：
   - **步骤一**：仅激活测试集中的 RGB 模态数据，运行推理，记录所有样本在 MoE 层输出的 Router 权重 $W_{rgb} \in \mathbb{R}^{N \times 3}$，计算其平均值
   - **步骤二**：对 NI 模态重复上述过程，得到平均权重 $W_{ni}$
   - **步骤三**：对 TI 模态重复上述过程，得到平均权重 $W_{ti}$

3. **统计维度**：
   - **平均值（Mean）**：展示每个模态对专家的平均选择倾向
   - **标准差（Standard Deviation）**：展示样本间的选择多样性

---

## 🚀 Linux 命令行

### 基本使用

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python analyze_modality_expert_weights.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --output_dir outputs/modality_expert_analysis/79.4mAP_model
```

### 限制样本数量（快速测试）

```bash
python analyze_modality_expert_weights.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --num_samples 500 \
    --output_dir outputs/modality_expert_analysis/79.4mAP_model_quick
```

---

## 📊 输出结果

### 生成的文件

1. **`modality_expert_weights_stacked_bar.png`** - 分组堆叠柱状图
   - X 轴：三个模态（RGB、NI、TI）
   - Y 轴：权重占比 (0% - 100%)
   - 颜色分层：每个柱子内部由三种颜色组成，分别对应 Scale 4×4、Scale 8×8、Scale 16×16
   - 标注：在柱子上方标注该模态下主导专家的百分比数值

2. **`modality_expert_weights_radar.png`** - 雷达图
   - 三个轴：三个专家（Scale 4×4、8×8、16×16）
   - 三条线：RGB、NI、TI 三个模态的权重分布
   - 直观展示不同模态对专家的偏好差异

3. **`modality_expert_weights_stats.txt`** - 统计数据
   - 每个模态的平均权重（Mean）
   - 每个模态的标准差（Std）
   - 主导专家信息

---

## 📈 图表解读

### 分组堆叠柱状图

**图表结构**：
```
┌─────────────────────────────────────────┐
│  Expert Weight Distribution by Modality │
│  (验证 Router 的"因材施教"能力)        │
├─────────────────────────────────────────┤
│                                         │
│  RGB    NI    TI                        │
│  ████   ████   ████                     │
│  ████   ████   ████                     │
│  ████   ████   ████                     │
│                                         │
│  标注：主导专家 + 百分比                 │
│                                         │
└─────────────────────────────────────────┘
```

**解读方法**：
- **如果三个模态的权重分布不同** → 证明 Router 根据模态特性动态选择（"因材施教"）✅
- **如果三个模态的权重分布相同** → 说明 Router 可能是随机的或未学习到模态差异 ❌

### 预期结果

**理想情况**（证明"因材施教"）：
- **RGB 模态**：可能更倾向于 Scale 4×4（细粒度特征，适合颜色和纹理细节）
- **NI 模态**：可能更倾向于 Scale 8×8（中等粒度，平衡细节和全局）
- **TI 模态**：可能更倾向于 Scale 16×16（粗粒度特征，适合温度分布的整体模式）

---

## 🔍 技术实现

### 核心代码逻辑

1. **分别处理每个模态**：
   ```python
   for modality_name in ['RGB', 'NI', 'TI']:
       # 调用 BACKBONE.forward()，仅传入当前模态
       _ = backbone(img_dict[modality_name], ..., modality=modality_label)
       # 获取专家权重
       weights = backbone.current_expert_weights
   ```

2. **统计计算**：
   ```python
   # 平均值
   mean_weights = torch.mean(all_weights, dim=0)
   # 标准差
   std_weights = torch.std(all_weights, dim=0)
   ```

3. **可视化**：
   - 堆叠柱状图：使用 `ax.bar()` 的 `bottom` 参数实现堆叠
   - 雷达图：使用 `projection='polar'` 实现极坐标图

---

## 📝 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--weight_path` | 模型权重文件路径（必需） | - |
| `--config_file` | 配置文件路径 | `configs/RGBNT201/yzy_best_Mambapro_moe.yml` |
| `--num_samples` | 收集的样本数量（None表示全部） | `None` |
| `--output_dir` | 输出目录 | `outputs/modality_expert_analysis` |

---

## ✅ 验证动态性

### 判断标准

1. **权重分布差异**：
   - 如果三个模态的权重分布显著不同 → 证明 Router 具备"因材施教"能力
   - 如果三个模态的权重分布相似 → 说明 Router 可能未学习到模态差异

2. **标准差分析**：
   - 标准差大 → 样本间选择多样性高，Router 能够根据具体样本调整
   - 标准差小 → 样本间选择一致性高，Router 对模态有稳定的偏好

3. **主导专家**：
   - 不同模态的主导专家不同 → 证明 Router 能够识别模态特性并做出相应选择

---

## 🔗 相关文档

- **性能图表模态说明**: `性能图表模态说明.md`
- **消融实验方案**: `消融实验_MOE替代方案.md`
- **多尺度配置说明**: `Readme合集/8-多尺度配置说明.md`

---

**最后更新**: 2025-12-21

