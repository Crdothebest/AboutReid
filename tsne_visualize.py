#!/usr/bin/env python
"""
轻量级 t-SNE 可视化脚本：
- 支持 npy / csv 特征输入；可选标签文件（npy/csv）。
- 可选择先 PCA 再 t-SNE；可随机下采样避免 O(n^2) 过慢。
- 输出 PNG，可选导出二维坐标 CSV。
使用示例：
  python tsne_visualize.py \
    --features path/to/features.npy \
    --labels path/to/labels.npy \
    --perplexity 40 --learning-rate 400 --pca-dims 50 \
    --sample 5000 \
    --output-png outputs/tsne/tsne.png \
    --output-csv outputs/tsne/tsne_points.csv
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_matrix(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        return np.load(path)
    # 默认以逗号分隔读取 CSV/TSV
    return pd.read_csv(path).values


def normalize_l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def maybe_sample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    x_sub = x[idx]
    y_sub = y[idx] if y is not None else None
    return x_sub, y_sub


def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization helper")
    parser.add_argument("--features", required=True, help="特征文件路径，支持 npy/csv")
    parser.add_argument("--labels", default=None, help="可选标签文件路径，支持 npy/csv")
    parser.add_argument("--perplexity", type=float, default=40.0, help="t-SNE perplexity")
    parser.add_argument("--learning-rate", type=float, default=400.0, help="t-SNE learning rate")
    parser.add_argument("--n-iter", type=int, default=1000, help="t-SNE 迭代步数")
    parser.add_argument("--pca-dims", type=int, default=50, help="先用 PCA 降到多少维，再做 t-SNE；<=0 则跳过 PCA")
    parser.add_argument("--sample", type=int, default=0, help="最大样本数，>0 时随机下采样以加速")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，保证可复现")
    parser.add_argument("--output-png", default=None, help="输出可视化 PNG 路径")
    parser.add_argument("--output-csv", default=None, help="可选：导出二维坐标 CSV")
    args = parser.parse_args()

    print("🔹 加载特征:", args.features)
    feats = load_matrix(args.features)
    print(f"   形状: {feats.shape}")

    labels = None
    if args.labels:
        print("🔹 加载标签:", args.labels)
        labels = load_matrix(args.labels).squeeze()
        if labels.shape[0] != feats.shape[0]:
            raise ValueError(f"标签数量 {labels.shape[0]} 与特征数量 {feats.shape[0]} 不匹配")

    # 可选下采样
    feats, labels = maybe_sample(feats, labels, args.sample, args.seed)
    if args.sample > 0:
        print(f"🔹 下采样后: {feats.shape[0]} 条样本")

    # 归一化
    feats = normalize_l2(feats)

    # 可选 PCA
    if args.pca_dims > 0 and feats.shape[1] > args.pca_dims:
        print(f"🔹 PCA -> {args.pca_dims} 维")
        feats = PCA(n_components=args.pca_dims, random_state=args.seed).fit_transform(feats)

    # t-SNE
    print("🔹 运行 t-SNE ...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        init="pca",
        random_state=args.seed,
        verbose=1,
    )
    feats_2d = tsne.fit_transform(feats)
    print("✅ t-SNE 完成")

    # 输出文件路径
    if args.output_png is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_png = f"outputs/tsne/tsne_{stamp}.png"
    ensure_dir(args.output_png)

    if args.output_csv:
        ensure_dir(args.output_csv)
        df_out = pd.DataFrame(feats_2d, columns=["x", "y"])
        if labels is not None:
            df_out["label"] = labels
        df_out.to_csv(args.output_csv, index=False)
        print(f"💾 已导出坐标: {args.output_csv}")

    # 画图
    plt.figure(figsize=(8, 6))
    if labels is None:
        plt.scatter(feats_2d[:, 0], feats_2d[:, 1], s=5, alpha=0.7, c="#1f77b4")
    else:
        # 将标签因子化，保证颜色映射稳定
        labels_flat = pd.Series(labels).astype("category")
        colors = labels_flat.cat.codes
        scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], s=5, alpha=0.7, c=colors, cmap="tab20")
        # 仅在类别数不多时绘制图例
        if labels_flat.nunique() <= 20:
            handles, _ = scatter.legend_elements(num=labels_flat.nunique())
            plt.legend(handles, labels_flat.cat.categories, title="label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"t-SNE (perp={args.perplexity}, lr={args.learning_rate})")
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300)
    plt.close()
    print(f"💾 已保存可视化: {args.output_png}")


if __name__ == "__main__":
    main()
