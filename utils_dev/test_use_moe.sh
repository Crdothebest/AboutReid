# Category: dev_utils (开发调试)
# Description: 开发辅助工具，包括进程清理、层输出调试、环境诊断及后端 API

#!/bin/bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python train_net.py \
  --config_file configs/RGBNT100/jzb_baseline_optimize.yml \
  --use_moe \
  MODEL.USE_CLIP_MULTI_SCALE True \
  MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]" \
  MODEL.MOE_USE_FIXED_WEIGHTS True \
  MODEL.MOE_FIXED_WEIGHTS "[0.33,0.33,0.34]" \
  OUTPUT_DIR "outputs/ablation/multiscale_fixed_weights_$(date +%Y%m%d_%H%M%S)"

