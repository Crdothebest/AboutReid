# Category: train_utils (训练与实验控制)
# Description: 负责模型训练启动、自动化实验管理及消融实验运行

#!/bin/bash

# 固定权重实验启动脚本
# 功能：使用多尺度滑动窗口（4×4、8×8、16×16）+ MOE专家网络 + 固定权重融合

cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

# 配置参数
CONFIG_FILE="${1:-configs/RGBNT201/20251013_experiment_config.yml}"
FIXED_WEIGHTS="${2:-[0.33,0.33,0.34]}"

echo "🚀 开始训练：多尺度（4、8、16）+ 固定权重实验"
echo "============================================================"
echo "配置："
echo "  配置文件: $CONFIG_FILE"
echo "  ✅ USE_CLIP_MULTI_SCALE: True"
echo "  ✅ CLIP_MULTI_SCALE_SCALES: [4, 8, 16]"
echo "  ✅ USE_MULTI_SCALE_MOE: True"
echo "  ✅ MOE_USE_FIXED_WEIGHTS: True"
echo "  ✅ MOE_FIXED_WEIGHTS: $FIXED_WEIGHTS"
echo "============================================================"
echo ""

# 生成输出目录（带时间戳）
OUTPUT_DIR="outputs/ablation/multiscale_fixed_weights_$(date +%Y%m%d_%H%M%S)"

# 执行训练命令
python train_net.py \
    --config_file "$CONFIG_FILE" \
    --use_moe \
    MODEL.USE_CLIP_MULTI_SCALE True \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]" \
    MODEL.MOE_USE_FIXED_WEIGHTS True \
    MODEL.MOE_FIXED_WEIGHTS "$FIXED_WEIGHTS" \
    OUTPUT_DIR "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "✅ 训练完成！"
echo "📁 输出目录: $OUTPUT_DIR"
echo "============================================================"
