#!/bin/bash

# 🔥 消融实验启动脚本：4×4小尺度滑动窗口实验
# 实验目的：验证4×4小尺度滑动窗口的独立效果
# 功能：启用基于CLIP的4×4小尺度特征提取机制

echo "🔥 启动消融实验：4×4小尺度滑动窗口实验"
echo "📊 实验配置："
echo "   - 滑动窗口尺度：仅4×4小尺度"
echo "   - MoE融合：禁用"
echo "   - 特征类型：局部细节特征"
echo "   - 预期效果：捕获局部细节和纹理信息"
echo "   - 输出目录：ablation_scale4_only"
echo ""

# 激活环境
source activate base

# 切换到项目目录
cd /home/zubuntu/workspace/yzy/MambaPro

# 运行训练
python train_net.py --config_file configs/RGBNT201/ablation_scale4_only.yml

echo ""
echo "🔥 4×4小尺度滑动窗口实验完成！"

