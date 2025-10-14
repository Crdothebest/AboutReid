# 🎨 多尺度滑动窗口+MoE特征可视化工具使用说明

## 🎯 **工具功能**

这是一个专门用于**多尺度滑动窗口+MoE特征可视化**的工具，基于Grad-CAM技术，帮助分析模型在多尺度特征提取和专家网络中的注意力分布。

---

## 🚀 **主要功能**

### **1. 多尺度特征可视化**
- 分析4×4、8×8、16×16滑动窗口的特征提取
- 生成各尺度的注意力热力图
- 对比不同尺度的特征关注区域

### **2. MoE专家权重可视化**
- 显示各专家网络的权重分布
- 分析专家网络的选择偏好
- 可视化专家权重的动态变化

### **3. Grad-CAM热力图**
- 生成MoE模块的注意力热力图
- 展示模型关注的关键区域
- 验证多尺度特征的有效性

---

## 📋 **使用方法**

### **基本使用**
```bash
python cam_multiscale_moe_visualize.py \
  --cfg configs/RGBNT201/MambaPro_moe.yml \
  --img-path data/RGBNT201/test/RGB/000001_cam1_0_01.jpg \
  --output-dir visualization_results/
```

### **自定义参数**
```bash
python cam_multiscale_moe_visualize.py \
  --cfg configs/RGBNT201/MambaPro_moe.yml \
  --img-path data/RGBNT201/test/RGB/000001_cam1_0_01.jpg \
  --target-layer clip_multi_scale_moe.moe_fusion \
  --output-dir my_visualization/ \
  --scales 4 8 16
```

---

## 🔧 **参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg` | 必需 | 配置文件路径 |
| `--img-path` | 必需 | 输入图像路径 |
| `--target-layer` | `clip_multi_scale_moe.moe_fusion` | 目标层名称 |
| `--output-dir` | `multiscale_moe_visualization` | 输出目录 |
| `--scales` | `[4, 8, 16]` | 滑动窗口尺度 |

---

## 📊 **输出结果**

### **1. 多尺度特征可视化**
```
multiscale_features.png
├── 原始图像
├── 4×4尺度特征热力图
├── 8×8尺度特征热力图
└── 16×16尺度特征热力图
```

### **2. MoE专家权重可视化**
```
moe_expert_weights.png
├── 专家权重柱状图
└── 专家权重饼图
```

### **3. Grad-CAM热力图**
```
gradcam_heatmap.jpg
└── MoE模块注意力热力图
```

---

## 🎯 **分析结果解读**

### **1. 多尺度特征分析**
- **4×4尺度**：关注局部细节特征
- **8×8尺度**：关注中等结构特征
- **16×16尺度**：关注全局上下文特征

### **2. MoE专家权重分析**
- **权重分布**：显示各专家的使用频率
- **专家偏好**：分析模型对不同尺度的偏好
- **动态变化**：观察权重随输入的变化

### **3. 注意力热力图**
- **红色区域**：模型高度关注的区域
- **蓝色区域**：模型较少关注的区域
- **绿色区域**：中等关注度的区域

---

## 💡 **使用技巧**

### **1. 选择合适的图像**
```bash
# 选择具有明显特征的图像
--img-path data/RGBNT201/test/RGB/000001_cam1_0_01.jpg
```

### **2. 调整分析尺度**
```bash
# 分析更多尺度
--scales 2 4 8 16 32
```

### **3. 指定目标层**
```bash
# 分析不同的MoE组件
--target-layer clip_multi_scale_moe.moe_fusion
--target-layer clip_multi_scale_moe.multi_scale_moe
```

---

## 🔍 **常见问题**

### **1. 模型层不存在**
```
错误：Layer 'clip_multi_scale_moe.moe_fusion' not found
解决：检查模型结构，使用正确的层名称
```

### **2. 无法获取专家权重**
```
警告：无法获取MoE专家权重信息
解决：确保模型包含MoE模块且已正确训练
```

### **3. 可视化结果为空**
```
问题：生成的可视化图像为空
解决：检查输入图像路径和模型权重文件
```

---

## 🚀 **高级用法**

### **1. 批量分析**
```bash
#!/bin/bash
# 批量分析多张图像
for img in data/RGBNT201/test/RGB/*.jpg; do
    python cam_multiscale_moe_visualize.py \
      --cfg configs/RGBNT201/MambaPro_moe.yml \
      --img-path "$img" \
      --output-dir "batch_visualization/$(basename "$img" .jpg)/"
done
```

### **2. 对比分析**
```bash
# 对比不同模型的MoE表现
python cam_multiscale_moe_visualize.py \
  --cfg configs/RGBNT201/MambaPro_moe.yml \
  --img-path test_image.jpg \
  --output-dir moe_model_analysis/

python cam_multiscale_moe_visualize.py \
  --cfg configs/RGBNT201/MambaPro_baseline.yml \
  --img-path test_image.jpg \
  --output-dir baseline_model_analysis/
```

### **3. 论文展示**
```bash
# 生成高质量的可视化结果
python cam_multiscale_moe_visualize.py \
  --cfg configs/RGBNT201/MambaPro_moe.yml \
  --img-path paper_images/representative_samples/ \
  --output-dir paper_visualizations/ \
  --scales 4 8 16
```

---

## 📝 **注意事项**

1. **模型要求**：确保模型包含多尺度MoE模块
2. **GPU内存**：可视化过程需要GPU内存
3. **图像质量**：选择清晰、特征明显的图像
4. **输出目录**：确保有足够的磁盘空间

---

## 🎉 **总结**

这个工具帮助您：
- ✅ **可视化多尺度特征提取**
- ✅ **分析MoE专家网络行为**
- ✅ **验证模型注意力分布**
- ✅ **生成论文展示材料**

**让您的多尺度MoE创新点变得可见！** 🚀
