# t-SNE 特征分布可视化

> 方法：t-SNE（t-分布随机邻域嵌入）
>
> 数据集：RGBNT201（三模态行人重识别）

## 说明

t-SNE 将模型提取的高维特征（512/1536 维）投影到二维平面，用于直观评估特征的判别性：
- **同一颜色的点**代表同一行人身份
- **类内紧凑**（同色点聚集）= 模型能很好地识别同一个人
- **类间分离**（不同色点远离）= 模型能有效区分不同人

## 文件列表

| 文件名 | 对应实验 | mAP |
|--------|---------|-----|
| `RGBNT201_79.4mAP_MoE.png` | MambaPro + 多尺度MoE（基线） | 79.4% |
| `RGBNT201_77.8mAP_scales-4+16_MoE2.png` | 双尺度[4,16] + 2专家 | 77.8% |
| `RGBNT201_76.8mAP_scales-4+8_MoE2.png` | 双尺度[4,8] + 2专家 | 76.8% |
| `RGBNT201_76.1mAP_scales-4_single.png` | 单尺度[4] | 76.1% |
| `RGBNT201_75.3mAP_scales-8_single.png` | 单尺度[8] | 75.3% |
| `RGBNT201_75.2mAP_scales-16_single.png` | 单尺度[16] | 75.2% |
| `RGBNT201_73.6mAP_scales-8+16_MoE2.png` | 双尺度[8,16] + 2专家 | 73.6% |

## 辅助文件

| 文件 | 说明 |
|------|------|
| `tsne_points.csv` | 某次 t-SNE 的原始坐标数据（x, y, label），可用于自定义绘图 |
| `t-SNE_可视化分析报告.md` | t-SNE 图的通用解读方法与论文写作模板 |

## 生成方式

```bash
python utils_visualize/generate_tsne_for_weights.py \
    --config configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --weight <模型权重路径> \
    --output_dir outputs/visualization/tsne/
```
