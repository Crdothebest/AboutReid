# 📊 visualize_reid_retrieval.py 使用说明

## 🎯 **脚本功能**

这是一个用于**对比分析两个ReID模型性能**的可视化工具，主要用于发现和分析旧模型优于新模型的案例，为模型改进提供参考。

---

## 🚀 **使用方法**

### **1. 基本使用（使用默认路径）**

```bash
python visualize_reid_retrieval.py
```

### **2. 自定义路径使用**

```bash
python visualize_reid_retrieval.py \
  --dataset_root data/RGBNT201 \
  --config_path configs/RGBNT201/MambaPro_moe.yml \
  --old_model_path pths/baseline_MambaProbest.pth \
  --new_model_path pths/moe_innovation_MambaProbest.pth \
  --top_k 9
```

### **3. 参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_root` | `data/RGBNT201` | 数据集根目录路径 |
| `--config_path` | `configs/RGBNT201/MambaPro_moe.yml` | 配置文件路径 |
| `--old_model_path` | `pths/baseline_MambaProbest.pth` | 旧模型（基线模型）权重路径 |
| `--new_model_path` | `pths/moe_innovation_MambaProbest.pth` | 新模型（创新模型）权重路径 |
| `--top_k` | `9` | Top-K检索的K值 |

---

## 📁 **项目结构要求**

### **数据集结构**
```
data/
└── RGBNT201/
    └── test/
        ├── RGB/          # RGB模态图像
        ├── NI/           # 近红外模态图像
        └── TI/           # 热红外模态图像
```

### **模型权重文件**
```
pths/
├── baseline_MambaProbest.pth      # 基线模型（旧模型）
└── moe_innovation_MambaProbest.pth # 创新模型（新模型）

# 或者
outputs/
├── baseline_experiment/models/MambaProbest.pth
└── moe_innovation_experiment/models/MambaProbest.pth
```

### **配置文件**
```
configs/
└── RGBNT201/
    └── MambaPro_moe.yml  # 模型配置文件（包含MoE配置）
```

---

## 🔍 **自动路径检测**

脚本会自动检测以下备选路径：

### **旧模型备选路径**
1. `outputs/baseline_experiment/models/MambaProbest.pth`
2. `outputs/baseline_thesis/models/MambaProbest.pth`
3. `pths/MambaProbest.pth`

### **新模型备选路径**
1. `outputs/moe_innovation_experiment/models/MambaProbest.pth`
2. `outputs/moe_innovation_experiment/MambaProbest.pth`
3. `pths/moe_MambaProbest.pth`

---

## 📊 **输出示例**

```
🔍 检查模型权重文件...
✅ 找到旧模型: outputs/baseline_experiment/models/MambaProbest.pth
✅ 找到新模型: outputs/moe_innovation_experiment/models/MambaProbest.pth
📁 旧模型路径: outputs/baseline_experiment/models/MambaProbest.pth
📁 新模型路径: outputs/moe_innovation_experiment/models/MambaProbest.pth

🔧 使用设备: cuda
📦 加载模型配置和权重...
📊 检测到相机数量: 4, 类别数量: 171
✅ 模型加载完成

🔍 步骤1: 筛选RGB模态下旧模型表现良好的Query图像...
📊 RGB模态 - Gallery: 1234张, Query: 1234张
✅ RGB模态下旧模型匹配正确 ≥9 的图像数: 145

🔍 步骤2&3: 对比新旧模型在三模态下的性能...
对比新旧模型中三模态的匹配结果: 100%|██████████| 145/145 [02:15<00:00,  1.07it/s]

📊 满足三模态旧模型准确数均高于新模型的图像数量：23

[1] 图像分析结果:
    📁 图像ID: 000045
    📂 图像路径: /data/RGBNT201/test/RGB/000045_cam1_0_01.jpg
    📈 匹配统计（Top-9正确匹配数）:
       ▸ RGB | 旧模型:  9/9   新模型:  7/9   优势: +2
       ▸ NI  | 旧模型:  8/9   新模型:  6/9   优势: +2
       ▸ TI  | 旧模型:  9/9   新模型:  7/9   优势: +2
```

---

## 🎯 **分析结果解读**

### **1. 筛选标准**
- **P集合**：在RGB模态下，旧模型Top-K检索**全部正确**的Query图像
- **S集合**：在**所有三个模态**下，旧模型的Top-K正确数都**严格大于**新模型的图像

### **2. 结果含义**
- **S集合为空**：新模型在所有测试图像上都表现优于或等于旧模型 ✅
- **S集合非空**：新模型在某些困难样本上存在不足，需要进一步优化 ⚠️

### **3. 优化建议**
- 如果S集合较大，说明新模型在某些场景下表现不佳
- 可以分析S集合中图像的共同特征，指导模型改进
- 考虑调整MoE参数或增加训练数据

---

## 🔧 **故障排除**

### **1. 路径不存在**
```
❌ 未找到旧模型权重文件，请检查路径
```
**解决方案**：检查模型权重文件是否存在于指定路径，或使用命令行参数指定正确路径。

### **2. 数据集路径错误**
```
❌ 数据集路径不存在: data/RGBNT201
```
**解决方案**：确保RGBNT201数据集已正确放置在data/目录下。

### **3. 配置文件错误**
```
❌ 配置文件不存在: configs/RGBNT201/MambaPro.yml
```
**解决方案**：检查配置文件路径是否正确。

---

## 💡 **使用技巧**

### **1. 快速测试**
```bash
# 使用较小的Top-K值进行快速测试
python visualize_reid_retrieval.py --top_k 5
```

### **2. 详细分析**
```bash
# 使用较大的Top-K值进行详细分析
python visualize_reid_retrieval.py --top_k 20
```

### **3. 不同数据集**
```bash
# 分析其他数据集
python visualize_reid_retrieval.py \
  --dataset_root data/RGBNT100 \
  --config_path configs/RGBNT100/MambaPro_moe.yml
```

---

## 📝 **注意事项**

1. **内存使用**：脚本会加载两个完整模型，确保有足够GPU内存
2. **计算时间**：特征提取过程较耗时，建议在GPU上运行
3. **结果保存**：当前版本只输出到终端，如需保存结果请重定向输出
4. **模型兼容性**：确保新旧模型使用相同的配置文件和数据集

---

## 🚀 **扩展功能**

### **保存结果到文件**
```bash
python visualize_reid_retrieval.py > analysis_results.txt 2>&1
```

### **批量分析**
```bash
# 分析多个模型对比
for old_model in pths/baseline_*.pth; do
  for new_model in pths/moe_*.pth; do
    python visualize_reid_retrieval.py \
      --old_model_path "$old_model" \
      --new_model_path "$new_model" \
      > "analysis_$(basename $old_model)_$(basename $new_model).txt"
  done
done
```

---

## 📞 **技术支持**

如果遇到问题，请检查：
1. 路径是否正确
2. 模型权重文件是否完整
3. 数据集结构是否符合要求
4. 配置文件是否正确

**祝您使用愉快！** 🎉
