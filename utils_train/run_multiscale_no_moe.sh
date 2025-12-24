# Category: train_utils (训练与实验控制)
# Description: 负责模型训练启动、自动化实验管理及消融实验运行

#!/bin/bash

# 多尺度（4、8、16）无 MOE 实验启动脚本
# 功能：使用多尺度滑动窗口提取特征，但使用简单 MLP 融合（不使用 MOE）

cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

echo "🚀 开始训练：多尺度（4、8、16）无 MOE 实验"
echo "配置："
echo "  ✅ USE_CLIP_MULTI_SCALE: True"
echo "  ✅ CLIP_MULTI_SCALE_SCALES: [4, 8, 16]"
echo "  ❌ USE_MULTI_SCALE_MOE: False"
echo ""

python train_net.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --use_multi_scale \
    --disable_moe \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]" \
    OUTPUT_DIR "outputs/ablation/multiscale_no_moe_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "✅ 训练完成！"


