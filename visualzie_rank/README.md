# ReID模型Top-K Ranked List可视化工具

## 📋 概述

这个工具专门用于生成ReID模型的Top-K检索结果可视化，展示Query图像与Gallery图像的相似度排序，并通过颜色框标注Ground Truth的正确性。

## 🎯 主要功能

1. **模型加载**: 加载训练好的ReID模型（.pth文件）
2. **特征提取**: 对Query和Gallery图像进行特征提取
3. **相似度计算**: 计算Query与Gallery之间的相似度
4. **Top-K排序**: 按相似度排序获取Top-K结果
5. **可视化生成**: 生成包含颜色框标注的可视化结果
6. **结果保存**: 保存可视化图像和汇总报告

## 📁 文件说明

- `visualize_ranked_list.py`: 主要的可视化工具脚本
- `visualize_reid_rank.py`: 原有的模型性能对比工具（不同用途）
- `example_usage.py`: 使用示例脚本
- `README.md`: 本说明文档

## 🚀 使用方法

### 基本用法

```bash
python visualize_ranked_list.py --help
```

### 常用参数

- `--dataset_root`: 数据集根目录 (默认: data/RGBNT201)
- `--config_path`: 配置文件路径 (默认: configs/RGBNT201/MambaPro_moe.yml)
- `--model_path`: 模型权重路径 (默认: pths/MambaProbest.pth)
- `--modality`: 模态类型 (RGB/NI/TI, 默认: RGB)
- `--top_k`: Top-K检索的K值 (默认: 9)
- `--num_queries`: 要可视化的Query数量 (默认: 10)
- `--output_dir`: 输出目录 (默认: ranked_list_results)

### 使用示例

#### 示例1: RGB模态Top-9检索可视化
```bash
python visualize_ranked_list.py \
    --dataset_root data/RGBNT201 \
    --config_path configs/RGBNT201/MambaPro_moe.yml \
    --model_path pths/MambaProbest.pth \
    --modality RGB \
    --top_k 9 \
    --num_queries 5 \
    --output_dir ranked_list_results_rgb
```

#### 示例2: NI模态Top-5检索可视化
```bash
python visualize_ranked_list.py \
    --modality NI \
    --top_k 5 \
    --num_queries 3 \
    --output_dir ranked_list_results_ni
```

#### 示例3: TI模态Top-10检索可视化
```bash
python visualize_ranked_list.py \
    --modality TI \
    --top_k 10 \
    --num_queries 3 \
    --output_dir ranked_list_results_ti
```

## 📊 输出结果

### 可视化图像
- 文件名格式: `ranked_list_XXXXXX_模态_topK.png`
- 包含Query图像和Top-K Gallery图像
- 绿色框标注正确匹配，红色框标注错误匹配
- 显示相似度分数和排名

### 汇总报告
- 文件名: `summary_report.txt`
- 包含所有Query的详细统计信息
- 显示每个Query的Top-K正确匹配数
- 提供总体性能统计

## 🎨 可视化特性

### 颜色标注
- **绿色框**: 正确匹配（Ground Truth正确）
- **红色框**: 错误匹配（Ground Truth错误）

### 显示信息
- Query图像和人员ID
- Top-K Gallery图像按相似度排序
- 相似度分数
- Ground Truth状态
- 排名信息

### 布局设计
- 2行布局：上排显示图像，下排显示相似度条形图
- 清晰的标题和标签
- 图例说明颜色含义

## 📈 性能统计

工具会自动计算并显示：
- 处理的Query数量
- 总正确匹配数
- Top-K准确率
- 每个Query的详细结果

## 🔧 技术细节

### 特征提取
- 支持RGB、NI、TI三种模态
- 使用预训练的ReID模型
- 自动处理多模态输入

### 相似度计算
- 使用余弦相似度
- 支持批量处理
- 高效的特征匹配

### 可视化生成
- 使用matplotlib生成高质量图像
- 支持自定义Top-K值
- 自动调整图像布局

## 🛠️ 依赖要求

- Python 3.7+
- PyTorch
- OpenCV
- Matplotlib
- PIL
- NumPy
- tqdm

## 📝 注意事项

1. **数据集格式**: 确保数据集按照RGBNT201格式组织
2. **模型兼容性**: 确保模型权重文件与配置文件匹配
3. **内存使用**: 大量Query可能需要较多内存
4. **输出目录**: 确保有足够的磁盘空间保存结果

## 🆚 与原有工具的区别

| 特性 | visualize_reid_rank.py | visualize_ranked_list.py |
|------|------------------------|--------------------------|
| **用途** | 模型性能对比分析 | Top-K检索可视化 |
| **输出** | 文本统计报告 | 可视化图像 |
| **功能** | 比较两个模型优劣 | 展示检索排序结果 |
| **标注** | 无 | 颜色框标注Ground Truth |
| **可视化** | 无 | 完整的图像展示 |

## 🎉 使用建议

1. **首次使用**: 建议先用少量Query测试
2. **参数调优**: 根据需求调整Top-K值和Query数量
3. **结果分析**: 查看可视化结果分析模型性能
4. **批量处理**: 可以编写脚本批量处理多个模态

## 📞 技术支持

如有问题，请检查：
1. 数据集路径是否正确
2. 模型权重文件是否存在
3. 配置文件是否匹配
4. 依赖包是否安装完整
