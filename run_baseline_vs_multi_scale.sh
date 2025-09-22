#!/bin/bash

# =============================================================================
# 快速对比测试脚本：原始CLIP vs 多尺度滑动窗口
# 功能：一键运行两种模式进行对比
# =============================================================================

echo "=========================================="
echo "🔥 原始CLIP vs 多尺度滑动窗口对比测试"
echo "=========================================="

# 设置配置文件路径
CONFIG_FILE="configs/RGBNT201/MambaPro.yml"

echo ""
echo "配置文件: $CONFIG_FILE"
echo ""

# 测试1: 原始CLIP (不使用多尺度滑动窗口)
echo "🔥 步骤1: 测试原始CLIP (不使用多尺度滑动窗口)"
echo "命令: python train_net.py --config_file $CONFIG_FILE --no_multi_scale"
echo ""

python train_net.py --config_file $CONFIG_FILE --no_multi_scale

echo ""
echo "✅ 原始CLIP测试完成！"
echo ""

# 测试2: 多尺度滑动窗口
echo "🔥 步骤2: 测试多尺度滑动窗口"
echo "命令: python train_net.py --config_file $CONFIG_FILE --use_multi_scale"
echo ""

python train_net.py --config_file $CONFIG_FILE --use_multi_scale

echo ""
echo "✅ 多尺度滑动窗口测试完成！"
echo ""

echo "=========================================="
echo "🎉 对比测试完成！"
echo "=========================================="
echo ""
echo "结果对比："
echo "1. 原始CLIP: 查看训练日志和性能指标"
echo "2. 多尺度滑动窗口: 查看训练日志和性能指标"
echo ""
echo "建议："
echo "- 对比两种模式的训练损失"
echo "- 对比两种模式的验证性能"
echo "- 分析多尺度滑动窗口的改进效果"
echo ""
