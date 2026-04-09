# 专家权重 × 性能对比分析

> 数据集：RGBNT201
>
> 内容：各多尺度/MoE 实验的专家门控权重分布与性能指标的联合可视化

## 说明

该目录下每张图对应一个 RGBNT201 上的训练实验，展示该实验中 MoE 专家网络的
门控权重分配情况，以及对应的性能指标（mAP/Rank-1），用于分析"专家选择策略"
与"最终性能"之间的关系。

## 文件列表

| 文件名 | 方法 | mAP | 专家配置 |
|--------|------|-----|---------|
| `RGBNT201_79.4mAP_MoE.png` | MoE（基线） | 79.4% | 3 专家 |
| `RGBNT201_78.3mAP_scales-4+8+16_MoE3.png` | 三尺度[4,8,16] | 78.3% | 3 专家 |
| `RGBNT201_77.8mAP_scales-4+16_MoE2.png` | 双尺度[4,16] | 77.8% | 2 专家 |
| `RGBNT201_76.8mAP_scales-4+8_MoE2.png` | 双尺度[4,8] | 76.8% | 2 专家 |
| `RGBNT201_76.1mAP_scales-4_single.png` | 单尺度[4] | 76.1% | 1 专家 |
| `RGBNT201_75.3mAP_scales-8_single.png` | 单尺度[8] | 75.3% | 1 专家 |
| `RGBNT201_73.6mAP_scales-8+16_MoE2.png` | 双尺度[8,16] | 73.6% | 2 专家 |

## 生成方式

```bash
python utils_visualize/batch_visualize_multiscale_performance.py
```
