# Category: train_utils (训练与实验控制)
# Description: 负责模型训练启动、自动化实验管理及消融实验运行

#!/bin/bash
# 连续执行训练，每次完成后休息 5 分钟再继续
# 用法：bash run_continuous_train.sh

CMD="python train_net.py \
  --config_file mybest_model/experiment_20251013_110028/configs/experiment_config.yml \
  OUTPUT_DIR ./outputs/replay_best_tml"

COUNTER=1

while true; do
  echo "========================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 第 ${COUNTER} 次训练开始..."
  echo "========================================"
  
  eval "$CMD"
  
  EXIT_CODE=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 第 ${COUNTER} 次训练完成 (退出码: ${EXIT_CODE})"
  
  if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠️  训练异常退出，退出码: ${EXIT_CODE}"
    echo "继续等待 5 分钟后重试..."
  fi
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 休息 5 分钟后继续下一轮..."
  sleep 300   # 5 分钟 = 300 秒
  
  COUNTER=$((COUNTER + 1))
  echo ""
done
