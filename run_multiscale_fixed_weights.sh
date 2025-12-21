#!/bin/bash

# 多尺度（4、8、16）+ 固定权重实验启动脚本
# 功能：使用多尺度滑动窗口提取特征，MOE专家网络处理，但使用固定权重融合

cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

# 默认权重：均等权重
WEIGHTS="${1:-[0.33,0.33,0.34]}"

echo "🚀 开始训练：多尺度（4、8、16）+ 固定权重实验"
echo "配置："
echo "  ✅ USE_CLIP_MULTI_SCALE: True"
echo "  ✅ CLIP_MULTI_SCALE_SCALES: [4, 8, 16]"
echo "  ✅ USE_MULTI_SCALE_MOE: True"
echo "  ✅ MOE_USE_FIXED_WEIGHTS: True"
echo "  ✅ MOE_FIXED_WEIGHTS: $WEIGHTS"
echo ""

python train_net.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --use_moe \
    MODEL.USE_CLIP_MULTI_SCALE True \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]" \
    MODEL.MOE_USE_FIXED_WEIGHTS True \
    MODEL.MOE_FIXED_WEIGHTS "$WEIGHTS" \
    OUTPUT_DIR "outputs/ablation/multiscale_fixed_weights_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "✅ 训练完成！"

