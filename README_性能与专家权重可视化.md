# 性能与专家权重演化可视化工具

## 📋 功能概述

从训练日志文件中提取性能指标（mAP、Rank-1）和专家权重数据，绘制双Y轴图，展示性能与专家权重演化的协同关系。

## 🎯 图表设计

### 双Y轴图结构

- **横轴 (X-axis)**: 训练进度（Epoch），标记为 0, 5, 10, ..., 60
- **左侧 Y 轴 (Primary Y-axis)**: 性能百分比（%），用于绘制 mAP 和 Rank-1
- **右侧 Y 轴 (Secondary Y-axis)**: 权重占比（0.0 - 1.0），用于绘制三个专家的比例

### 可视化元素

1. **折线图（上层）**: 
   - mAP（蓝色实线，圆形标记）
   - Rank-1（紫红色实线，方形标记）

2. **堆叠面积图（背景）**:
   - Scale 4×4（红色区域）
   - Scale 8×8（蓝色区域）
   - Scale 16×16（浅蓝色区域）
   - 虚线显示各专家权重的趋势

3. **完整黑色边框**: 图表被完整的黑色矩形边框包裹

## 🚀 使用方法

### 单个日志文件

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid
eval "$(conda shell.bash hook)"
conda activate MambaPro

python visualize_performance_expert_weights.py \
    --log_path /path/to/train_*.log \
    --title_suffix " - Model Name" \
    --output_path /path/to/output.png
```

### 参数说明

- `--log_path`: 训练日志文件路径（必需）
- `--output_path`: 输出图片路径（可选，默认：日志文件同目录下的 `performance_expert_weights.png`）
- `--title_suffix`: 标题后缀（可选，用于区分不同模型）

### 批量处理

```bash
python batch_visualize_performance_expert_weights.py
```

会自动查找所有训练日志文件并生成对应的可视化图。

## 📊 数据提取

脚本会从日志文件中提取以下信息：

1. **Epoch编号**: 从 `Validation Results - Epoch: XX` 提取
2. **mAP值**: 从 `Current mAP: XX%` 提取
3. **Rank-1值**: 从 `CMC curve, Rank-1  :XX%` 提取
4. **专家权重**: 从 `📊 专家权重分布(Val): [w1, w2, w3]` 提取

## 📈 输出示例

### 数据统计
```
✅ 提取到 12 个epoch的验证结果
   Epoch范围: 5 - 60
   mAP范围: 45.8% - 79.4%
   Rank-1范围: 50.1% - 81.9%
   专家权重范围:
     Scale 4: 0.35 - 0.95
     Scale 8: 0.03 - 0.34
     Scale 16: 0.02 - 0.31
```

### 图表特点

- ✅ 清晰的性能指标趋势（mAP 和 Rank-1）
- ✅ 直观的专家权重演化（堆叠面积图）
- ✅ 完整的黑色边框（符合论文要求）
- ✅ 双Y轴设计，便于对比分析

## 🔍 图表解读

### 关键观察点

1. **性能提升与权重分配的关系**:
   - 当 Scale 4×4 的权重从 0.35 增加到 0.95 时
   - mAP 从 45.8% 提升到 79.4%
   - 说明细粒度专家（Scale 4）对性能提升至关重要

2. **权重演化趋势**:
   - 早期：三个专家权重相对均衡（~0.33）
   - 中期：Scale 4 权重开始增加
   - 后期：Scale 4 占据主导地位（~0.95）

3. **性能与决策的协同演化**:
   - 图表直观证明：随着模型学会把权重分配给细粒度专家，性能显著提升

## 📁 文件位置

- **主脚本**: `visualize_performance_expert_weights.py`
- **批量处理脚本**: `batch_visualize_performance_expert_weights.py`
- **输出目录**: `outputs/performance_analysis/`

## 🎨 图表样式

- **图表尺寸**: 14×8 英寸
- **分辨率**: 300 DPI
- **边框**: 完整黑色矩形边框（线宽 2.0）
- **字体**: 支持中文显示
- **图例**: 合并显示所有元素，位置在左上角

## ✅ 使用示例

### 示例 1: 单个模型

```bash
python visualize_performance_expert_weights.py \
    --log_path /home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/logs/train_20251212_112223.log \
    --title_suffix " - 79.4mAP Model" \
    --output_path outputs/performance_analysis/79.4mAP_performance_expert_weights.png
```

### 示例 2: 批量处理所有日志

```bash
python batch_visualize_performance_expert_weights.py
```

## 📝 注意事项

1. 确保日志文件包含验证结果数据
2. 日志格式必须符合预期（包含 `Validation Results`、`Current mAP`、`Rank-1`、`专家权重分布` 等关键词）
3. 需要激活 `MambaPro` conda 环境
4. 输出目录会自动创建


