#!/bin/bash
# 快速保存优秀结果脚本
# 功能：一键保存满意的训练结果

# 创建结果保存目录
mkdir -p results/excellent_results
mkdir -p results/backup_results

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 获取实验名称（从命令行参数或默认值）
EXPERIMENT_NAME=${1:-"excellent_result"}
DESCRIPTION=${2:-"优秀训练结果"}

echo "🏆 保存优秀结果: ${EXPERIMENT_NAME}_${TIMESTAMP}"
echo "📁 保存目录: results/excellent_results/${EXPERIMENT_NAME}_${TIMESTAMP}"

# 创建结果目录
RESULT_DIR="results/excellent_results/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# 保存训练日志
if [ -f "outputs/moe_optimized_experiment/train_log.txt" ]; then
    cp outputs/moe_optimized_experiment/train_log.txt "$RESULT_DIR/train_log.txt"
    echo "✅ 训练日志已保存"
else
    echo "⚠️  训练日志不存在"
fi

# 保存模型权重
if [ -f "pths/MambaProbest.pth" ]; then
    cp pths/MambaProbest.pth "$RESULT_DIR/MambaProbest.pth"
    echo "✅ 模型权重已保存"
else
    echo "⚠️  模型权重不存在"
fi

# 保存配置文件
if [ -f "configs/RGBNT201/MambaPro_moe.yml" ]; then
    cp configs/RGBNT201/MambaPro_moe.yml "$RESULT_DIR/config.yml"
    echo "✅ 配置文件已保存"
else
    echo "⚠️  配置文件不存在"
fi

# 创建结果记录
cat > "$RESULT_DIR/result_info.txt" << EOF
实验名称: ${EXPERIMENT_NAME}
时间戳: ${TIMESTAMP}
描述: ${DESCRIPTION}
保存时间: $(date)
状态: 已保存

文件列表:
- train_log.txt (训练日志)
- MambaProbest.pth (模型权重)
- config.yml (配置文件)
- result_info.txt (结果信息)
EOF

echo "📝 结果信息已保存"
echo "🎯 结果描述: ${DESCRIPTION}"
echo "📁 保存位置: ${RESULT_DIR}"
echo "✅ 优秀结果保存完成！"
