#!/bin/bash
# 运行热力图可视化脚本

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate MambaPro

# 切换到脚本目录
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

# 运行脚本
python generate_heatmap_visualization.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --query_id 000274 \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --output_path heatmap_000274.png
