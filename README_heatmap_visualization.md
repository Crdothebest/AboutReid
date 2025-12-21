# 多模态热力图可视化生成脚本

## 📋 概述

`generate_heatmap_visualization.py` 是一个专门用于生成多模态热力图可视化的脚本，可以生成类似 `heatmap_000274.png` 的效果。

## 🎨 生成效果

生成的图像采用 **3行×2列** 布局：
- **左列**：RGB、NIR、TIR 三种模态的原始图像
- **右列**：对应模态叠加了 Grad-CAM 热力图的图像

## 🚀 使用方法

### 基本使用

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

python generate_heatmap_visualization.py \
    --weight_path /path/to/model.pth \
    --config_file /path/to/config.yml \
    --query_id 000274 \
    --dataset_root /path/to/RGBNT201 \
    --output_path heatmap_000274.png
```

### 完整示例

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

python generate_heatmap_visualization.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/MambaPro/outputs/baseline/RGBNT201/77.0mAP_20251218_164722.pth \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --query_id 000274 \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --output_path outputs/heatmap_000274.png \
    --alpha 0.4
```

### 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--weight_path` | ✅ | 模型权重文件路径（.pth 文件） | - |
| `--config_file` | ✅ | 配置文件路径（YAML 格式） | - |
| `--query_id` | ✅ | 查询人员ID（如 "000274"） | - |
| `--dataset_root` | ❌ | 数据集根目录 | `/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201` |
| `--output_path` | ❌ | 输出图像路径 | `heatmap_{query_id}.png` |
| `--target_layer` | ❌ | 目标层路径（用于 Grad-CAM） | 自动检测 |
| `--alpha` | ❌ | 热力图透明度（0.0-1.0） | `0.4` |

## 📊 输出说明

- **输出格式**：PNG 图像，300 DPI
- **图像尺寸**：12×18 英寸（3行×2列布局）
- **文件命名**：如果未指定 `--output_path`，默认命名为 `heatmap_{query_id}.png`

## 🔍 常见问题

### 1. 找不到图像文件

**问题**：提示 "未找到任何模态的图像"

**解决**：
- 检查 `--dataset_root` 路径是否正确
- 确认数据集目录结构为：`{dataset_root}/test/RGB/`, `{dataset_root}/test/NI/`, `{dataset_root}/test/TI/`
- 确认图像文件命名格式为：`{query_id}_*.jpg`

### 2. 目标层检测失败

**问题**：提示 "无法找到合适的目标层"

**解决**：
- 手动指定 `--target_layer` 参数
- 常见的目标层：
  - CLIP ViT: `BACKBONE.image_encoder.transformer.resblocks.11`
  - ViT: `BACKBONE.base.transformer.resblocks.11`
  - ResNet: `BACKBONE.base.layer4`

### 3. 内存不足

**问题**：GPU 内存不足

**解决**：
- 使用 CPU 模式（如果 CUDA 不可用，会自动使用 CPU）
- 减少批处理大小（本脚本每次只处理一张图像，应该不会有问题）

## 💡 提示

1. **透明度调整**：使用 `--alpha` 参数调整热力图的透明度
   - `0.0`：完全透明（只显示原始图像）
   - `0.4`：默认值，平衡效果
   - `1.0`：完全不透明（热力图完全覆盖）

2. **批量生成**：如果需要为多个人员ID生成热力图，可以使用循环：
   ```bash
   for id in 000274 000275 000276; do
       python generate_heatmap_visualization.py \
           --weight_path model.pth \
           --config_file config.yml \
           --query_id $id \
           --dataset_root /path/to/RGBNT201 \
           --output_path outputs/heatmap_${id}.png
   done
   ```

3. **与现有脚本的区别**：
   - `test_heatmap_from_weight.py`：批量生成多个样本的热力图，并包含评估报告
   - `generate_heatmap_visualization.py`：专门用于生成单个样本的多模态热力图可视化（本脚本）

## 📝 示例输出

成功运行后会看到类似输出：

```
============================================================
生成多模态热力图可视化
============================================================
🔧 使用设备: cuda

📦 加载模型配置和权重...
✅ 模型加载完成
🔍 自动检测目标层...
   使用目标层: BACKBONE.base.transformer.resblocks.11

🖼️  生成热力图可视化: Query ID = 000274
✅ 已保存多模态热力图可视化: outputs/heatmap_000274.png

🎉 完成！
📁 结果保存在: outputs/heatmap_000274.png
```
