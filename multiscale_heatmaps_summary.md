# Multiscale 热力图可视化总结

## ✅ 生成完成状态

所有 6 个尺度组合的热力图可视化已成功生成！

## 📊 各尺度组合结果

| 尺度组合 | mAP | 文件夹名称 | 热力图数量 | 输出目录 |
|---------|-----|-----------|-----------|---------|
| 8×8+16×16 | 73.58 | 73.58_8x8+16x16_20251217_175523 | 10+ | `outputs/multiscale_heatmaps/73.58_8x8+16x16_20251217_175523/` |
| 16×16 | 75.22 | 75.22_16x16_20251217_141827 | 10+ | `outputs/multiscale_heatmaps/75.22_16x16_20251217_141827/` |
| 8×8 | 75.28 | 75.28_8x8_20251217_131802 | 10 | `outputs/multiscale_heatmaps/75.28_8x8_20251217_131802/` |
| 4×4 | 76.13 | 76.13_4x4_20251217_121738 | 10 | `outputs/multiscale_heatmaps/76.13_4x4_20251217_121738/` |
| 4×4+8×8 | 76.79 | 76.79_4x4+8x8_20251217_151621 | 10 | `outputs/multiscale_heatmaps/76.79_4x4+8x8_20251217_151621/` |
| 4×4+16×16 | 77.76 | 77.76_4x4+16x16_20251217_160700 | 10 | `outputs/multiscale_heatmaps/77.76_4x4+16x16_20251217_160700/` |

## 📁 输出文件结构

每个尺度组合的输出目录包含：

```
{scale_folder}/
├── heatmap_000269.png      # 人员ID 000269 的热力图
├── heatmap_000270.png      # 人员ID 000270 的热力图
├── ...
├── evaluation_report.md    # 评估报告
└── heatmap_data.npz        # 热力图数据（numpy格式）
```

## 🔍 快速查看命令

### 查看所有热力图文件

```bash
# 统计所有热力图文件
find /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps -name "heatmap_*.png" | wc -l

# 查看特定尺度组合的热力图
ls -lh /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps/77.76_4x4+16x16_20251217_160700/*.png
```

### 查看评估报告

```bash
# 查看所有尺度组合的评估报告摘要
for dir in /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps/*/; do
    echo "=== $(basename $dir) ==="
    if [ -f "$dir/evaluation_report.md" ]; then
        grep -A 5 "结果摘要" "$dir/evaluation_report.md" | head -10
    else
        echo "未找到评估报告"
    fi
    echo
done
```

### 对比不同尺度组合的对齐度

```bash
for dir in /home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps/*/; do
    scale=$(basename "$dir" | cut -d'_' -f2)
    if [ -f "$dir/evaluation_report.md" ]; then
        avg=$(grep "平均跨模态对齐度" "$dir/evaluation_report.md" | grep -oE "[0-9]+\.[0-9]+")
        echo "$scale: $avg"
    fi
done | sort -t: -k2 -n
```

## 📈 分析建议

### 1. 对比单尺度 vs 多尺度

- **单尺度**: 4×4, 8×8, 16×16
- **多尺度**: 4×4+8×8, 4×4+16×16, 8×8+16×16

观察多尺度组合是否在热力图上展现出更丰富的特征关注区域。

### 2. 性能与可视化关系

- **最高 mAP**: 77.76 (4×4+16×16)
- **最低 mAP**: 73.58 (8×8+16×16)

对比最高和最低性能模型的热力图，分析特征关注区域的差异。

### 3. 跨模态对齐度分析

查看每个尺度组合的 `evaluation_report.md`，对比：
- 跨模态对齐度
- 热力图质量评估
- 不同模态（RGB、NI、TI）的关注区域一致性

## 🎨 可视化说明

### 热力图布局

每个热力图包含 3 行（RGB、NI、TI），每行 3 列：
- **左列**: 原始图像
- **中列**: 热力图（颜色映射）
- **右列**: 叠加图（热力图叠加在原始图像上）

### 颜色含义

- **红色/暖色**: 高响应区域（模型关注度高）
- **蓝色/冷色**: 低响应区域（模型关注度低）

## 📝 后续操作

### 生成更多样本

如果需要为某个尺度组合生成更多样本：

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

python test_heatmap_from_weight.py \
    --weight_path outputs/multiscale/77.76_4x4+16x16_20251217_160700/MambaProbest.pth \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --num_images 20 \
    --output_dir outputs/multiscale_heatmaps/77.76_4x4+16x16_20251217_160700
```

### 使用 EigenCAM 生成热力图

对于 Transformer/Mamba 架构，EigenCAM 效果更好：

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

## 🔗 相关文档

- [Multiscale 热力图可视化指南](./multiscale_heatmaps_guide.md)
- [Baseline 模型热力图可视化命令](../MambaPro/generate_heatmaps_commands.md)

---
**生成时间**: 2025-12-20  
**数据集**: RGBNT201  
**方法**: GradCAM  
**样本数量**: 每个尺度组合 10 个样本
