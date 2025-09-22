#!/bin/bash

# =============================================================================
# 快速测试脚本：多尺度滑动窗口
# 功能：快速测试多尺度滑动窗口创新点
# =============================================================================

echo "🔥 快速测试多尺度滑动窗口"
echo "配置文件: configs/RGBNT201/MambaPro_multi_scale.yml"
echo "输出目录: outputs/multi_scale_experiment"
echo ""

# 运行多尺度滑动窗口实验
python train_net.py --config_file configs/RGBNT201/MambaPro_multi_scale.yml

echo ""
echo "✅ 多尺度滑动窗口测试完成！"
echo "结果保存在: outputs/multi_scale_experiment"
