#!/usr/bin/env bash
# 一键导出验证集特征并运行 t-SNE 可视化
# 用法（推荐传入权重路径作为第1个参数）：
#   bash run_tsne_from_weight.sh /workspace/yzy/MambaPro/outputs/better_pth/your_model.pth
# 可选环境变量：
#   CONFIG_PATH  默认 configs/RGBNT201/yzy_best_Mambapro_moe.yml
#   FEAT_PATH    默认 data/features.npy
#   LABEL_PATH   默认 data/labels.npy
#   TSNE_PERP    默认 40
#   TSNE_LR      默认 400
#   TSNE_PCA     默认 50
#   TSNE_SAMPLE  默认 5000（>0 时随机下采样，加速）
#   OUTPUT_DIR   默认 outputs/tsne
#   PNG_PATH     默认 $OUTPUT_DIR/tsne_时间戳.png
#   CSV_PATH     默认 $OUTPUT_DIR/tsne_points.csv

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/RGBNT201/yzy_best_Mambapro_moe.yml}"
WEIGHT_PATH="${1:-${WEIGHT_PATH:-}}"
FEAT_PATH="${FEAT_PATH:-data/features.npy}"
LABEL_PATH="${LABEL_PATH:-data/labels.npy}"
TSNE_PERP="${TSNE_PERP:-40}"
TSNE_LR="${TSNE_LR:-400}"
TSNE_PCA="${TSNE_PCA:-50}"
TSNE_SAMPLE="${TSNE_SAMPLE:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tsne}"
PNG_PATH="${PNG_PATH:-${OUTPUT_DIR}/tsne_$(date +%Y%m%d_%H%M%S).png}"
CSV_PATH="${CSV_PATH:-${OUTPUT_DIR}/tsne_points.csv}"

if [[ -z "${WEIGHT_PATH}" ]]; then
  echo "用法: bash run_tsne_from_weight.sh /path/to/weight.pth"
  echo "或提前导出 WEIGHT_PATH 环境变量后再运行。"
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "❌ 配置文件不存在: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -f "${WEIGHT_PATH}" ]]; then
  echo "❌ 权重文件不存在: ${WEIGHT_PATH}"
  exit 1
fi

mkdir -p "$(dirname "${FEAT_PATH}")" "$(dirname "${LABEL_PATH}")" "${OUTPUT_DIR}"

echo "🔹 使用配置: ${CONFIG_PATH}"
echo "🔹 使用权重: ${WEIGHT_PATH}"
echo "🔹 导出特征到: ${FEAT_PATH}"
echo "🔹 导出标签到: ${LABEL_PATH}"

python - <<'PY'
import os
import numpy as np
import torch
from config import cfg
from data import make_dataloader
from modeling import make_model

config_file = os.environ["CONFIG_PATH"]
weight_path = os.environ["WEIGHT_PATH"]
feat_path = os.environ["FEAT_PATH"]
label_path = os.environ["LABEL_PATH"]

cfg.merge_from_file(config_file)
cfg.freeze()

device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
model.load_param(weight_path)
model.to(device)
model.eval()

feats, labels = [], []
with torch.no_grad():
    for img, pid, camid, camids, target_view, imgpath in val_loader:
        img = {k: v.to(device) for k, v in img.items()}  # RGB/NI/TI
        camids = camids.to(device)
        target_view = target_view.to(device)
        feat = model(img, cam_label=camids, view_label=target_view)
        feats.append(feat.cpu().numpy())
        labels.append(pid.numpy())

feats = np.vstack(feats)
labels = np.concatenate(labels)
np.save(feat_path, feats)
np.save(label_path, labels)
print(f"✅ 导出特征: {feats.shape} -> {feat_path}")
print(f"✅ 导出标签: {labels.shape} -> {label_path}")
PY

echo "🔹 运行 t-SNE ..."
python tsne_visualize.py \
  --features "${FEAT_PATH}" \
  --labels   "${LABEL_PATH}" \
  --perplexity "${TSNE_PERP}" \
  --learning-rate "${TSNE_LR}" \
  --pca-dims "${TSNE_PCA}" \
  --sample "${TSNE_SAMPLE}" \
  --output-png "${PNG_PATH}" \
  --output-csv "${CSV_PATH}"

echo "✅ 完成，PNG: ${PNG_PATH}"
echo "✅ 完成，CSV: ${CSV_PATH}"
