# EigenCAM 热力图生成使用说明

## ✅ 功能已实现

脚本 `generate_heatmap_visualization.py` 现在支持 **EigenCAM** 和 **GradCAM** 两种方法。

## 🚀 使用方法

### EigenCAM 热力图生成（推荐用于 Transformer/Mamba 架构）

```bash
eval "$(conda shell.bash hook)" && conda activate MambaPro && cd /home/zhanghaoyang/Desktop/yzy/AboutReid && python generate_heatmap_visualization.py --weight_path /home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml --query_id 000274 --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 --method eigencam --alpha 0.5
```

### GradCAM 热力图生成（默认方法）

```bash
eval "$(conda shell.bash hook)" && conda activate MambaPro && cd /home/zhanghaoyang/Desktop/yzy/AboutReid && python generate_heatmap_visualization.py --weight_path /home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml --query_id 000274 --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 --method gradcam --alpha 0.4
```

## 📝 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 热力图方法：`gradcam` 或 `eigencam` | `gradcam` |
| `--alpha` | 热力图透明度（0.0-1.0） | `0.4` |
| `--target_layer` | 目标层路径（可选，会自动检测） | 自动检测 |
| `--output_path` | 输出路径（可选，会自动生成） | `outputs/{method}/{weight_name}/{method}_{query_id}.png` |

## 📁 输出路径

- **EigenCAM**: `outputs/EigenCAM/{weight_name}/eigencam_{query_id}.png`
- **GradCAM**: `outputs/Grad_CAM/{weight_name}/heatmap_{query_id}.png`

## 🎯 EigenCAM vs GradCAM

### EigenCAM 特点
- ✅ 不需要梯度计算（更快）
- ✅ 对 Transformer/Mamba 架构效果更好
- ✅ 能精准分离物体和背景
- ✅ 推荐目标层：`BACKBONE.base.ln_post`

### GradCAM 特点
- ✅ 基于梯度，更直观
- ✅ 适用于各种架构
- ✅ 推荐目标层：`BACKBONE.base.transformer.resblocks.11`

## 📊 输出效果

生成的图像采用 **3行×2列** 布局：
- **左列**：RGB、NIR、TIR 三种模态的原始图像
- **右列**：对应模态叠加了热力图的图像（使用 EigenCAM 或 GradCAM）

## 💡 完整命令示例

### 使用 EigenCAM（推荐）

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python generate_heatmap_visualization.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --query_id 000274 \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --method eigencam \
    --alpha 0.5 \
    --target_layer BACKBONE.base.ln_post
```

### 使用 GradCAM

```bash
python generate_heatmap_visualization.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --query_id 000274 \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --method gradcam \
    --alpha 0.4
```

## ✅ 已修复的问题

1. ✅ 添加了 EigenCAM 类到 `grad_cam.py`
2. ✅ 支持 `--method` 参数选择热力图方法
3. ✅ 自动检测并推荐 EigenCAM 目标层
4. ✅ 输出文件自动保存到对应的 `outputs` 目录
5. ✅ 支持 TorchScript 模型自动查找替代权重文件
