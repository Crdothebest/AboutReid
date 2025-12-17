# 热力图测试脚本 - 快速使用指南

## 🎯 功能概述

`test_heatmap_from_weight.py` 是一个完整的测试脚本，每次运行可以：

1. ✅ 基于指定的 .pth 权重文件加载模型
2. ✅ **从测试集（test 目录）随机选择 10 张图像**（按人员ID）
3. ✅ 为每个样本生成多模态热力图（RGB/NI/TI）
4. ✅ 进行量化评估（背景抑制、跨模态对齐等）
5. ✅ 自动诊断模型问题并提供改进建议
6. ✅ 生成详细的评估报告

**重要说明**：
- ✅ **所有图像均来自测试集（test 目录），不涉及训练集**
- ✅ 数据集结构: `{dataset_root}/test/RGB/`, `{dataset_root}/test/NI/`, `{dataset_root}/test/TI/`
- ✅ 确保热力图测试使用的是测试集数据，保证评估的客观性

---

## 🚀 快速开始

### 最简单的方式

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

python test_heatmap_from_weight.py \
    --weight_path /path/to/your_model.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml
```

### 实际使用示例

```bash
# 测试第一个模型
python test_heatmap_from_weight.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --num_images 10

# 测试第二个模型（多尺度）
python test_heatmap_from_weight.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale/77.76_4x4+16x16_20251217_160700/MambaProbest.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --num_images 10
```

---

## 📊 输出结果

运行后会在输出目录生成：

1. **可视化图像** (10 张)
   - `heatmap_000123.png` - 多模态热力图（RGB/NIR/TIR 三行三列）

2. **评估报告**
   - `evaluation_report.md` - 详细的评估报告

3. **热力图数据**
   - `heatmap_data.npz` - 原始热力图数据（用于后续分析）

---

## 📈 评估指标

### 关键指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **跨模态对齐度** | > 0.8 | ⭐ 最重要，评估三种模态热力图位置一致性 |
| **背景响应** | < 0.3 | 评估背景抑制能力 |
| **人体响应** | > 0.6 | 评估特征学习有效性 |
| **聚焦分数** | > 0.5 | 评估高响应区域集中度 |

### 判断标准

- **对齐度 > 0.8**: ✅ 优秀，模型学到了模态不变性特征
- **对齐度 0.5-0.8**: ⚠️ 一般，需要进一步优化
- **对齐度 < 0.5**: ❌ 较差，跨模态对齐失败（P0 优先级改进）

---

## 💡 使用技巧

### 1. 固定随机种子（可复现）

```bash
python test_heatmap_from_weight.py \
    --weight_path model.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --seed 42
```

### 2. 指定输出目录

```bash
python test_heatmap_from_weight.py \
    --weight_path model.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --output_dir outputs/Grad_CAM/my_test_results
```

### 3. 调整热力图透明度

```bash
python test_heatmap_from_weight.py \
    --weight_path model.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --alpha 0.6  # 更明显的热力图
```

---

## 📋 完整参数列表

```bash
python test_heatmap_from_weight.py \
    --weight_path <必需> 模型权重文件路径（.pth） \
    --config_file <必需> 配置文件路径（YAML） \
    --dataset_root <可选> 数据集根目录 \
    --num_images <可选> 测试图像数量（默认10） \
    --output_dir <可选> 输出目录（默认自动生成） \
    --target_layer <可选> 目标层路径（默认自动检测） \
    --alpha <可选> 热力图透明度（默认0.4） \
    --seed <可选> 随机种子（用于可复现性）
```

---

## 🔍 结果解读示例

### 优秀结果

```
平均跨模态对齐度: 0.92
  - 优秀 (>0.8): 8
  - 一般 (0.5-0.8): 2
  - 较差 (<0.5): 0
```

**解读**: ✅ 模型性能优秀，可用于论文展示

### 需要改进的结果

```
平均跨模态对齐度: 0.45
  - 优秀 (>0.8): 1
  - 一般 (0.5-0.8): 3
  - 较差 (<0.5): 6
```

**解读**: ❌ 跨模态对齐失败，需要立即改进（P0 优先级）

**改进建议**（报告会自动生成）:
1. 添加跨模态对比学习损失
2. 增加模态不变性约束
3. 检查 MoE 融合策略

---

## 📁 文件结构

```
outputs/Grad_CAM/test_<model_name>_<timestamp>/
├── heatmap_000123.png          # 可视化图像
├── heatmap_000456.png
├── ...
├── evaluation_report.md        # 评估报告
└── heatmap_data.npz            # 热力图数据
```

---

## 🐛 常见问题

### Q1: 找不到目标层
**解决**: 
```bash
# 先列出可用层
python visualize_gradcam.py --list_layers --config_file ... --weight_path ...

# 然后手动指定
python test_heatmap_from_weight.py ... --target_layer BACKBONE.image_encoder.transformer.resblocks.11
```

### Q2: GPU 内存不足
**解决**: 减少测试图像数量
```bash
python test_heatmap_from_weight.py ... --num_images 5
```

### Q3: 某些模态图像缺失
**解决**: 脚本会自动跳过，确保测试集中三种模态图像都存在

---

## 📚 相关文档

- **热力图介绍.md** - 详细的理论说明和解读方法
- **热力图实现说明.md** - 实现细节和使用方法
- **测试脚本使用说明.md** - 更详细的使用说明

---

**快速命令**:
```bash
# 测试模型（最简单）
python test_heatmap_from_weight.py \
    --weight_path <your_model.pth> \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml
```

---

**最后更新**: 2025年12月17日
