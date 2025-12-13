#!/bin/bash
# Continuous ablation runs with different window scales.
# Usage: bash run_continuous_ablation.sh

CFG="mybest_model/experiment_20251013_110028/configs/experiment_config.yml"
BASE_OUT="./outputs/replay_best_tml"

# Customize the scales you want to sweep.
SCALE_LIST=(
  "[8]"        # single window 8
  "[4,8]"      # 4 & 8
  "[4,8,16]"   # 4 & 8 & 16
)

while true; do
  for SCALES in "${SCALE_LIST[@]}"; do
    TS=$(date '+%Y%m%d_%H%M%S')
    CLEAN=$(echo "${SCALES}" | tr -d '[] ')
    SCALE_TAG="s${CLEAN//,/s}"  # e.g., [4,8,16] -> s4s8s16
    OUT_DIR="${BASE_OUT}/run_${TS}_${SCALE_TAG}"

    CMD="python train_net.py \
      --use_multi_scale \
      --config_file ${CFG} \
      OUTPUT_DIR ${OUT_DIR} \
      MODEL.CLIP_MULTI_SCALE_SCALES \"${SCALES}\""
    # 如果希望 MoE 的尺度与 CLIP 多尺度一致，追加：
    #   MODEL.CLIP_MULTI_SCALE_SCALES \"${SCALES}\" MODEL.MOE_SCALES \"${SCALES}\"

    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] window=${SCALES} start..."
    echo "========================================"

    eval ${CMD}
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] window=${SCALES} finished (exit=${EXIT_CODE})"
    if [ ${EXIT_CODE} -ne 0 ]; then
      echo "Warning: training exited abnormally; continue after 5 minutes."
    fi

    echo "Sleeping 5 minutes before next run..."
    sleep 300
    echo ""
  done
done
