# EigenCAM 注意力可视化

> 方法：EigenCAM（基于特征图主成分的类激活映射）
>
> 数据集：RGBNT201（三模态行人重识别）

## 说明

EigenCAM 通过对模型最后一层特征图进行 PCA 分解，提取第一主成分作为注意力热力图，
可视化模型在三种模态（RGB/NIR/TIR）下关注的空间区域。

每张图通常包含：原始图像 + 热力图叠加，展示模型对行人不同身体部位的关注程度。

## 目录内容

| 子目录 | 对应实验 | mAP | 图片数 |
|--------|---------|-----|--------|
| `RGBNT201_79.4mAP_MoE/` | MambaPro + 多尺度MoE | 79.4% | 30 张 |
| `RGBNT201_73.6mAP_scales-8+16_MoE2/` | 双尺度[8,16] + 2专家MoE | 73.6% | 1 张 |

## 文件命名

`eigencam_{query_id}.png` — query_id 为 RGBNT201 测试集中的行人查询编号（6 位数字）。

## 生成方式

```bash
python visualize_Cam/generate_heatmap_visualization.py \
    --method eigencam \
    --config configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --weight <模型权重路径> \
    --query_id <查询ID>
```
