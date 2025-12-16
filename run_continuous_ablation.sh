#!/bin/bash
# Continuous ablation runs with different window scales.
# Usage: bash run_continuous_ablation.sh
#
# 功能：
# 1. 一直循环，每轮循环之间休息5分钟
# 2. 每轮循环依次执行6种窗口组合（每种组合之间不休息）：
#    - [4] (4x4窗口)
#    - [8] (8x8窗口)
#    - [16] (16x16窗口)
#    - [4,8] (4x4+8x8组合)
#    - [4,16] (4x4+16x16组合)
#    - [8,16] (8x8+16x16组合)
# 3. 训练脚本会自动根据mAP重命名输出目录为: mAP_组合方式_时间戳

CFG="mybest_model/experiment_20251013_110028/configs/experiment_config.yml"
BASE_OUT="./outputs/multiscale"

# 定义所有窗口组合（按顺序执行）
SCALE_LIST=(
  "[4]"        # 4x4窗口
  "[8]"        # 8x8窗口
  "[16]"       # 16x16窗口
  "[4,8]"      # 4x4+8x8组合
  "[4,16]"     # 4x4+16x16组合
  "[8,16]"     # 8x8+16x16组合
)

# 对应的组合方式标签（用于目录命名）
SCALE_LABELS=(
  "4x4"
  "8x8"
  "16x16"
  "4x4+8x8"
  "4x4+16x16"
  "8x8+16x16"
)

# 无限循环
while true; do
  echo "========================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 开始新一轮循环"
  echo "========================================"
  
  # 遍历所有窗口组合
  for i in "${!SCALE_LIST[@]}"; do
    SCALES="${SCALE_LIST[$i]}"
    SCALE_LABEL="${SCALE_LABELS[$i]}"
    
    TS=$(date '+%Y%m%d_%H%M%S')
    # 初始目录名包含组合方式，训练脚本重命名时会保留
    # 格式：组合方式_时间戳，例如：4x4+8x8_20251216_220218
    OUT_DIR="${BASE_OUT}/${SCALE_LABEL}_${TS}"

    # 计算专家数量（等于窗口尺度数量）
    NUM_EXPERTS=$(echo "${SCALES}" | tr -d '[] ' | tr ',' '\n' | wc -l)
    
    # 构建训练命令
    # 注意：同时设置 CLIP_MULTI_SCALE_SCALES 和 MOE_SCALES，确保两者一致
    # MOE_NUM_EXPERTS 会自动匹配 MOE_SCALES 的长度，但这里显式设置以确保一致性
    CMD="python train_net.py \
      --use_multi_scale \
      --use_moe \
      --config_file ${CFG} \
      OUTPUT_DIR ${OUT_DIR} \
      MODEL.CLIP_MULTI_SCALE_SCALES \"${SCALES}\" \
      MODEL.MOE_SCALES \"${SCALES}\" \
      MODEL.MOE_NUM_EXPERTS ${NUM_EXPERTS}"

    echo ""
    echo "========================================"
    echo "🔥 训练任务信息"
    echo "========================================"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  窗口尺度: ${SCALES} → ${SCALE_LABEL}"
    echo "  MoE专家数量: ${NUM_EXPERTS} (自动匹配窗口数量)"
    echo "  输出目录: ${OUT_DIR}"
    echo "========================================"
    echo ""

    # 执行训练
    eval ${CMD}
    EXIT_CODE=$?
    
    echo ""
    echo "========================================"
    echo "✅ 训练完成"
    echo "========================================"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  窗口尺度: ${SCALES} (${SCALE_LABEL})"
    echo "  退出码: ${EXIT_CODE}"
    
    if [ ${EXIT_CODE} -ne 0 ]; then
      echo "  ⚠️  警告: 训练异常退出，继续下一个组合"
    fi

    # 检查输出目录是否被重命名（训练脚本会根据mAP自动重命名）
    # 训练脚本会将目录重命名为: mAP_组合方式_时间戳
    # 例如: 85.23_4x4+8x8_20251216_220218
    if [ -d "${OUT_DIR}" ]; then
      echo "  📁 输出目录: ${OUT_DIR} (未重命名，训练可能未完成)"
    else
      # 查找可能被重命名的目录（格式: mAP_组合方式_时间戳）
      # 查找最近创建的包含组合方式标签的目录
      RENAMED_DIR=$(find "${BASE_OUT}" -maxdepth 1 -type d -name "*${SCALE_LABEL}*" 2>/dev/null | sort -r | head -1)
      if [ -n "${RENAMED_DIR}" ] && [ -d "${RENAMED_DIR}" ]; then
        echo "  📁 输出目录已重命名: ${RENAMED_DIR}"
      fi
    fi
    echo "========================================"
    echo ""
  done
  
  echo "========================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 一轮循环完成，5分钟后开始下一轮"
  echo "========================================"
  sleep 300
  echo ""
done
