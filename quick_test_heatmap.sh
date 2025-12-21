#!/bin/bash
"""
快速测试热力图效果脚本

功能说明：
这是一个便捷的 shell 脚本，用于快速测试指定权重文件的热力图效果。

使用方法：
./quick_test_heatmap.sh <weight_path> [config_file] [num_images]

示例：
  # 使用默认配置
  ./quick_test_heatmap.sh outputs/best_model.pth

  # 指定配置文件和图像数量
  ./quick_test_heatmap.sh outputs/best_model.pth configs/RGBNT201/yzy_best_Mambapro_moe.yml 10
"""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 参数解析
WEIGHT_PATH="$1"
CONFIG_FILE="${2:-configs/RGBNT201/yzy_best_Mambapro_moe.yml}"
NUM_IMAGES="${3:-10}"

# 检查参数
if [ -z "$WEIGHT_PATH" ]; then
    echo "❌ 错误: 请提供权重文件路径"
    echo ""
    echo "使用方法:"
    echo "  ./quick_test_heatmap.sh <weight_path> [config_file] [num_images]"
    echo ""
    echo "示例:"
    echo "  ./quick_test_heatmap.sh outputs/best_model.pth"
    echo "  ./quick_test_heatmap.sh outputs/best_model.pth configs/RGBNT201/yzy_best_Mambapro_moe.yml 10"
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$WEIGHT_PATH" ]; then
    echo "❌ 错误: 权重文件不存在: $WEIGHT_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 显示配置信息
echo "="*60
echo "快速测试热力图效果"
echo "="*60
echo "权重文件: $WEIGHT_PATH"
echo "配置文件: $CONFIG_FILE"
echo "测试图像数: $NUM_IMAGES"
echo "="*60
echo ""

# 运行测试脚本
python test_heatmap_from_weight.py \
    --weight_path "$WEIGHT_PATH" \
    --config_file "$CONFIG_FILE" \
    --num_images "$NUM_IMAGES"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试完成！"
    echo "📁 结果保存在输出目录中"
else
    echo ""
    echo "❌ 测试失败，请检查错误信息"
    exit 1
fi






