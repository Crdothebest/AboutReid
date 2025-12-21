#!/bin/bash

# 模态专家权重分析启动脚本
# 功能：分析不同模态（RGB、NI、TI）对三个专家（4x4, 8x8, 16x16）的选择权重

cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

# 默认参数
WEIGHT_PATH="${1:-/home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth}"
CONFIG_FILE="${2:-configs/RGBNT201/yzy_best_Mambapro_moe.yml}"
NUM_SAMPLES="${3:-}"
OUTPUT_DIR="${4:-outputs/modality_expert_analysis/79.4mAP_model}"

echo "🚀 开始分析：不同模态对专家的选择权重"
echo "配置："
echo "  权重文件: $WEIGHT_PATH"
echo "  配置文件: $CONFIG_FILE"
echo "  样本数量: ${NUM_SAMPLES:-全部}"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 构建命令
CMD="python analyze_modality_expert_weights.py \
    --weight_path \"$WEIGHT_PATH\" \
    --config_file \"$CONFIG_FILE\" \
    --output_dir \"$OUTPUT_DIR\""

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# 执行命令
eval $CMD

echo ""
echo "✅ 分析完成！"
echo "📁 输出目录: $OUTPUT_DIR"

