# Multiscale 热力图可视化指南

## 📋 概述

为 multiscale 文件夹下的所有尺度组合权重生成热力图可视化，展示不同尺度组合下模型的特征关注区域。

## 🎯 尺度组合列表

根据 multiscale 文件夹结构，包含以下尺度组合：

1. **73.58_8x8+16x16** (8×8 + 16×16)
2. **75.22_16x16** (16×16)
3. **75.28_8x8** (8×8)
4. **76.13_4x4** (4×4)
5. **76.79_4x4+8x8** (4×4 + 8×8)
6. **77.76_4x4+16x16** (4×4 + 16×16)

## 🚀 使用方法

### 方法1: 使用批量生成脚本（推荐）

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

python generate_multiscale_heatmaps.py
```

这个脚本会：
- 自动扫描所有 multiscale 文件夹
- 为每个 `MambaProbest.pth` 权重生成热力图
- 使用 RGBNT201 数据集
- 为每个尺度组合生成 10 个样本的热力图

### 方法2: 手动为单个尺度组合生成

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

python test_heatmap_from_weight.py \
    --weight_path outputs/multiscale/77.76_4x4+16x16_20251217_160700/MambaProbest.pth \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --num_images 10 \
    --output_dir outputs/multiscale_heatmaps/77.76_4x4+16x16_20251217_160700
```

### 方法3: 使用 EigenCAM（推荐用于 Transformer/Mamba）

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

python test_eigencam.py \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --weight_path outputs/multiscale/77.76_4x4+16x16_20251217_160700/MambaProbest.pth \
    --query_id 000258 \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --output_dir outputs/multiscale_heatmaps_eigencam/77.76_4x4+16x16 \
    --target_layer "BACKBONE.base.ln_post" \
    --alpha 0.4
```

## 📊 输出结果

### 输出目录结构

```
outputs/multiscale_heatmaps/
├── 73.58_8x8+16x16_20251217_175523/
│   ├── heatmap_000269.png
│   ├── heatmap_000272.png
│   ├── ...
│   ├── evaluation_report.md
│   └── heatmap_data.npz
├── 75.22_16x16_20251217_141827/
│   └── ...
├── 75.28_8x8_20251217_131802/
│   └── ...
├── 76.13_4x4_20251217_121738/
│   └── ...
├── 76.79_4x4+8x8_20251217_151621/
│   └── ...
└── 77.76_4x4+16x16_20251217_160700/
    └── ...
```

### 输出文件说明

- **heatmap_XXXXXX.png**: 每个人员ID的多模态热力图（RGB、NI、TI）
- **evaluation_report.md**: 评估报告，包含对齐度等指标
- **heatmap_data.npz**: 热力图数据（numpy 格式）

## 🔍 查看结果

### 查看所有生成的热力图

```bash
# 查看所有热力图文件
find /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps -name "*.png" | wc -l

# 查看特定尺度组合的热力图
ls -lh /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps/77.76_4x4+16x16_20251217_160700/*.png

# 查看评估报告
cat /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps/77.76_4x4+16x16_20251217_160700/evaluation_report.md
```

### 对比不同尺度组合

```bash
# 查看所有尺度组合的评估报告摘要
for dir in /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps/*/; do
    echo "=== $(basename $dir) ==="
    grep "平均跨模态对齐度" "$dir/evaluation_report.md" || echo "未找到评估报告"
    echo
done
```

## 📈 批量生成所有尺度组合的 EigenCAM 热力图

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

# 定义尺度组合列表
SCALES=(
    "73.58_8x8+16x16_20251217_175523"
    "75.22_16x16_20251217_141827"
    "75.28_8x8_20251217_131802"
    "76.13_4x4_20251217_121738"
    "76.79_4x4+8x8_20251217_151621"
    "77.76_4x4+16x16_20251217_160700"
)

OUTPUT_BASE="/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps_eigencam"

# 为每个尺度组合生成 EigenCAM 热力图
for SCALE in "${SCALES[@]}"; do
    echo "处理: $SCALE"
    python test_eigencam.py \
        --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
        --weight_path "outputs/multiscale/${SCALE}/MambaProbest.pth" \
        --query_id 000258 \
        --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
        --output_dir "${OUTPUT_BASE}/${SCALE}" \
        --target_layer "BACKBONE.base.ln_post" \
        --alpha 0.4
done
```

## 🎨 可视化说明

### 热力图颜色含义

- **红色/暖色**: 高响应区域（模型关注度高）
- **蓝色/冷色**: 低响应区域（模型关注度低）

### 多模态热力图布局

每个热力图包含 3 行（RGB、NI、TI），每行 3 列：
- **左列**: 原始图像
- **中列**: 热力图
- **右列**: 叠加图（热力图叠加在原始图像上）

## 📝 自定义参数

### 修改可视化图像数量

编辑 `generate_multiscale_heatmaps.py` 中的 `NUM_IMAGES` 变量：

```python
NUM_IMAGES = 20  # 改为 20 个样本
```

### 修改热力图透明度

在命令行中添加 `--alpha` 参数：

```bash
python test_heatmap_from_weight.py ... --alpha 0.6
```

### 修改目标层

对于 EigenCAM，可以指定不同的目标层：

```bash
python test_eigencam.py ... --target_layer "BACKBONE.base.transformer.resblocks.11.norm1"
```

## 🔧 故障排除

### 权重文件不存在

如果某个尺度组合的权重文件不存在，脚本会跳过并继续处理下一个。

### 配置文件路径错误

确保配置文件路径正确：
```bash
ls /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml
```

### 数据集路径错误

确保数据集路径正确：
```bash
ls /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201/test/RGB
```

## 📊 结果分析

### 评估指标

- **跨模态对齐度**: 衡量不同模态（RGB、NI、TI）热力图的一致性
  - 优秀 (>0.8): 模态间高度一致
  - 一般 (0.5-0.8): 模态间基本一致
  - 较差 (<0.5): 模态间差异较大

### 对比不同尺度组合

通过对比不同尺度组合的热力图，可以：
1. 观察不同尺度对特征关注区域的影响
2. 分析多尺度融合的效果
3. 验证模型的可解释性

---
**最后更新**: 2025-12-20
