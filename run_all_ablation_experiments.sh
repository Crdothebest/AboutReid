#!/bin/bash

# 🔥 消融实验综合启动脚本
# 功能：依次运行所有单尺度消融实验
# 实验顺序：4×4 → 8×8 → 16×16

echo "🔥 开始运行所有单尺度消融实验"
echo "=" * 80
echo ""

# 激活环境
source activate base

# 切换到项目目录
cd /home/zubuntu/workspace/yzy/MambaPro

# 实验1：4×4小尺度滑动窗口实验
echo "🔥 实验1/3：4×4小尺度滑动窗口实验"
echo "=" * 50
python train_net.py --config_file configs/RGBNT201/ablation_scale4_only.yml
echo "✅ 4×4小尺度实验完成"
echo ""

# 实验2：8×8中尺度滑动窗口实验
echo "🔥 实验2/3：8×8中尺度滑动窗口实验"
echo "=" * 50
python train_net.py --config_file configs/RGBNT201/ablation_scale8_only.yml
echo "✅ 8×8中尺度实验完成"
echo ""

# 实验3：16×16大尺度滑动窗口实验
echo "🔥 实验3/3：16×16大尺度滑动窗口实验"
echo "=" * 50
python train_net.py --config_file configs/RGBNT201/ablation_scale16_only.yml
echo "✅ 16×16大尺度实验完成"
echo ""

echo "🎉 所有单尺度消融实验完成！"
echo "📊 实验结果保存在对应的输出目录中："
echo "   - ablation_scale4_only/"
echo "   - ablation_scale8_only/"
echo "   - ablation_scale16_only/"

