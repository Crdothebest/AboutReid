# Category: vis_utils (可视化工具)
# Description: 提供热力图 (CAM)、t-SNE 降维、检索结果展示等模型可视化功能

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
    """
    加载特征或标签矩阵文件（支持 .npy 和 .csv 格式）
    
    功能说明：
    - 自动识别文件格式（.npy 或 .csv）
    - .npy 格式：直接使用 numpy.load() 加载，速度快，适合大型数组
    - .csv 格式：使用 pandas.read_csv() 加载，适合表格数据
    
    Args:
        path (str): 文件路径，支持 .npy 或 .csv 格式
        
    Returns:
        np.ndarray: 加载的矩阵数据，形状为 (n_samples, n_features) 或 (n_samples,)
        
    示例:
        >>> features = load_matrix('data/features.npy')  # 加载特征文件
        >>> labels = load_matrix('data/labels.csv')      # 加载标签文件
    """
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        # NumPy 二进制格式：直接加载，速度快，保持数据类型
        return np.load(path)
    # CSV/TSV 格式：使用 pandas 读取，自动处理分隔符和表头
    return pd.read_csv(path).values


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """
    L2 归一化特征向量（将每个样本的特征向量归一化为单位向量）
    
    功能说明：
    - 计算每个样本的 L2 范数（欧几里得距离）
    - 将特征向量除以其 L2 范数，得到单位向量
    - 归一化后的特征向量长度为 1，便于计算余弦相似度
    
    数学公式：
        x_normalized = x / (||x||_2 + ε)
        其中 ||x||_2 = sqrt(sum(x_i^2))，ε=1e-12 防止除零
    
    Args:
        x (np.ndarray): 输入特征矩阵，形状为 (n_samples, n_features)
        
    Returns:
        np.ndarray: L2 归一化后的特征矩阵，形状与输入相同
        
    示例:
        >>> features = np.array([[3, 4], [1, 1]])
        >>> normalized = normalize_l2(features)
        >>> # 结果：[[0.6, 0.8], [0.707, 0.707]]（单位向量）
    """
    # 计算每个样本的 L2 范数（沿特征维度 axis=1）
    # keepdims=True 保持维度，便于广播除法
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12  # 1e-12 防止除零
    return x / norm  # 广播除法：每个特征除以对应的 L2 范数


def maybe_sample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    """
    随机下采样数据（用于加速 t-SNE 计算）
    
    功能说明：
    - t-SNE 的时间复杂度为 O(n^2)，样本数过多时计算非常慢
    - 通过随机下采样减少样本数，在保持数据分布的同时加速计算
    - 使用固定随机种子保证结果可复现
    
    Args:
        x (np.ndarray): 特征矩阵，形状为 (n_samples, n_features)
        y (np.ndarray): 标签数组，形状为 (n_samples,)，可为 None
        max_samples (int): 最大样本数，>0 时进行下采样，<=0 或 >=n_samples 时不采样
        seed (int): 随机种子，保证可复现性
        
    Returns:
        tuple: (x_sub, y_sub) - 下采样后的特征和标签
               - 如果不需要下采样，返回原始数据
               - y_sub 可能为 None（如果输入 y 为 None）
        
    示例:
        >>> features = np.random.randn(10000, 512)
        >>> labels = np.random.randint(0, 10, 10000)
        >>> feat_sub, label_sub = maybe_sample(features, labels, max_samples=5000, seed=42)
        >>> # 结果：随机选择 5000 个样本
    """
    # 如果不需要下采样（max_samples <= 0 或样本数 <= max_samples），直接返回
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y
    
    # 使用固定随机种子生成随机数生成器（保证可复现）
    rng = np.random.default_rng(seed)
    # 随机选择 max_samples 个索引（不重复）
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    
    # 根据索引提取对应的特征和标签
    x_sub = x[idx]
    y_sub = y[idx] if y is not None else None  # 如果标签为 None，返回 None
    return x_sub, y_sub


def ensure_dir(path: str):
    """
    确保文件路径的目录存在（如果不存在则创建）
    
    功能说明：
    - 自动创建输出文件所需的目录结构
    - 支持多级目录创建（如 outputs/tsne/2024/12/）
    - 如果目录已存在，则不进行任何操作
    
    Args:
        path (str): 文件路径（可以是文件或目录路径）
        
    示例:
        >>> ensure_dir('outputs/tsne/result.png')  # 创建 outputs/tsne/ 目录
        >>> ensure_dir('data/features.npy')       # 创建 data/ 目录
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    """
    主函数：执行 t-SNE 降维可视化
    
    功能流程：
    1. 解析命令行参数
    2. 加载特征和标签文件
    3. 可选：随机下采样（加速计算）
    4. L2 归一化特征
    5. 可选：PCA 降维（进一步加速）
    6. 执行 t-SNE 降维到 2D
    7. 可视化并保存结果（PNG + 可选的 CSV）
    
    输出：
    - PNG 图像：2D 散点图，不同颜色代表不同类别
    - CSV 文件（可选）：包含每个样本的 2D 坐标和标签
    """
    parser = argparse.ArgumentParser(
        description="t-SNE 可视化工具：将高维特征降维到 2D 空间进行可视化分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 基础用法：使用默认参数
  python tsne_visualize.py --features data/features.npy --labels data/labels.npy
  
  # 完整参数：自定义所有设置
  python tsne_visualize.py \\
    --features data/features.npy \\
    --labels data/labels.npy \\
    --perplexity 40 \\
    --learning-rate 400 \\
    --pca-dims 50 \\
    --sample 5000 \\
    --output-png outputs/tsne/result.png \\
    --output-csv outputs/tsne/coordinates.csv
  
  # 仅可视化（无标签）
  python tsne_visualize.py --features data/features.npy --output-png outputs/tsne/result.png
        """
    )
    
    # ========== 必需参数 ==========
    parser.add_argument(
        "--features", 
        required=True, 
        help="特征文件路径（必需），支持 .npy 或 .csv 格式。形状应为 (n_samples, n_features)"
    )
    
    # ========== 可选参数 ==========
    parser.add_argument(
        "--labels", 
        default=None, 
        help="标签文件路径（可选），支持 .npy 或 .csv 格式。形状应为 (n_samples,)。"
              "如果提供，可视化时会用不同颜色表示不同类别"
    )
    
    parser.add_argument(
        "--perplexity", 
        type=float, 
        default=40.0, 
        help="t-SNE 困惑度参数（默认：40.0）。"
              "控制每个点的近邻数量，典型值范围：5-50。"
              "较小的值关注局部结构，较大的值关注全局结构。"
              "建议值：样本数 < 1000 时用 5-30，> 1000 时用 30-50"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=400.0, 
        help="t-SNE 学习率（默认：400.0）。"
              "控制优化步长，典型值范围：10-1000。"
              "如果可视化出现"爆炸"（点聚集在边缘），降低学习率；"
              "如果收敛太慢，提高学习率"
    )
    
    parser.add_argument(
        "--n-iter", 
        type=int, 
        default=1000, 
        help="t-SNE 迭代步数（默认：1000）。"
              "更多迭代通常得到更好的结果，但计算时间更长。"
              "建议值：500-2000，可根据收敛情况调整"
    )
    
    parser.add_argument(
        "--pca-dims", 
        type=int, 
        default=50, 
        help="PCA 预降维维度（默认：50）。"
              "先用 PCA 降到指定维度，再做 t-SNE（加速计算）。"
              "<=0 则跳过 PCA，直接对原始特征做 t-SNE。"
              "建议值：原始特征维度很大（>1000）时使用 50-100"
    )
    
    parser.add_argument(
        "--sample", 
        type=int, 
        default=0, 
        help="最大样本数（默认：0，不采样）。"
              ">0 时随机下采样到指定数量，用于加速 t-SNE 计算。"
              "t-SNE 时间复杂度为 O(n^2)，样本数过多时非常慢。"
              "建议值：5000-10000（在可接受时间内获得合理结果）"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="随机种子（默认：0）。"
              "用于控制随机下采样和 t-SNE 初始化的随机性，"
              "相同种子保证结果可复现"
    )
    
    parser.add_argument(
        "--output-png", 
        default=None, 
        help="输出可视化 PNG 路径（默认：outputs/tsne/tsne_时间戳.png）。"
              "如果未指定，自动生成带时间戳的文件名"
    )
    
    parser.add_argument(
        "--output-csv", 
        default=None, 
        help="输出坐标 CSV 路径（可选）。"
              "如果指定，会导出每个样本的 2D 坐标和标签（如果有），"
              "可用于后续分析或重新可视化"
    )
    
    args = parser.parse_args()

    # ========== 步骤 1: 加载数据 ==========
    print("🔹 加载特征:", args.features)
    feats = load_matrix(args.features)
    print(f"   形状: {feats.shape} (n_samples={feats.shape[0]}, n_features={feats.shape[1]})")

    # 加载标签（如果提供）
    labels = None
    if args.labels:
        print("🔹 加载标签:", args.labels)
        labels = load_matrix(args.labels).squeeze()  # squeeze() 去除单维度
        if labels.shape[0] != feats.shape[0]:
            raise ValueError(
                f"标签数量 {labels.shape[0]} 与特征数量 {feats.shape[0]} 不匹配。"
                f"请确保特征和标签文件对应相同的样本"
            )
        print(f"   标签形状: {labels.shape}, 类别数: {len(np.unique(labels))}")

    # ========== 步骤 2: 可选下采样（加速计算）==========
    # t-SNE 时间复杂度为 O(n^2)，样本数过多时计算非常慢
    # 通过随机下采样减少样本数，在保持数据分布的同时加速计算
    feats, labels = maybe_sample(feats, labels, args.sample, args.seed)
    if args.sample > 0 and feats.shape[0] < args.sample:
        print(f"🔹 下采样后: {feats.shape[0]} 条样本（原始样本数少于指定数量，未采样）")
    elif args.sample > 0:
        print(f"🔹 下采样后: {feats.shape[0]} 条样本（从原始数据中随机选择）")

    # ========== 步骤 3: L2 归一化特征 ==========
    # 归一化特征向量，使每个样本的特征向量长度为 1
    # 这有助于 t-SNE 更好地捕获特征之间的相对关系
    print("🔹 L2 归一化特征...")
    feats = normalize_l2(feats)
    print(f"   归一化后特征范数: {np.linalg.norm(feats, axis=1)[:5]} (前5个样本，应为1.0)")

    # ========== 步骤 4: 可选 PCA 预降维（进一步加速）==========
    # 如果特征维度很大（>1000），先用 PCA 降到较低维度（如 50-100），再做 t-SNE
    # 这样可以显著加速计算，同时保留主要信息
    if args.pca_dims > 0 and feats.shape[1] > args.pca_dims:
        print(f"🔹 PCA 预降维: {feats.shape[1]} -> {args.pca_dims} 维（加速 t-SNE 计算）")
        pca = PCA(n_components=args.pca_dims, random_state=args.seed)
        feats = pca.fit_transform(feats)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"   PCA 保留方差比例: {explained_var:.2%}")

    # ========== 步骤 5: 执行 t-SNE 降维 ==========
    # t-SNE (t-Distributed Stochastic Neighbor Embedding) 是一种非线性降维方法
    # 它将高维数据映射到 2D 或 3D 空间，保持局部邻域结构
    # 适用于可视化高维数据的聚类结构
    print("🔹 运行 t-SNE 降维（这可能需要几分钟，取决于样本数）...")
    print(f"   参数: perplexity={args.perplexity}, learning_rate={args.learning_rate}, n_iter={args.n_iter}")
    tsne = TSNE(
        n_components=2,              # 降维到 2D（用于可视化）
        perplexity=args.perplexity,   # 困惑度：控制每个点的近邻数量
        learning_rate=args.learning_rate,  # 学习率：控制优化步长
        n_iter=args.n_iter,           # 迭代次数：更多迭代通常得到更好结果
        init="pca",                   # 初始化方法：使用 PCA 初始化（比随机初始化更稳定）
        random_state=args.seed,       # 随机种子：保证可复现
        verbose=1,                    # 显示进度信息
    )
    feats_2d = tsne.fit_transform(feats)  # 执行降维，返回 (n_samples, 2) 的 2D 坐标
    print("✅ t-SNE 完成")
    print(f"   输出形状: {feats_2d.shape} (n_samples, 2)")

    # ========== 步骤 6: 准备输出路径 ==========
    # 如果未指定输出路径，自动生成带时间戳的文件名
    if args.output_png is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_png = f"outputs/tsne/tsne_{stamp}.png"
    ensure_dir(args.output_png)  # 确保输出目录存在

    # ========== 步骤 7: 导出坐标 CSV（可选）==========
    # 如果指定了 CSV 输出路径，导出 2D 坐标和标签（如果有）
    # 可用于后续分析、重新可视化或与其他工具集成
    if args.output_csv:
        ensure_dir(args.output_csv)
        df_out = pd.DataFrame(feats_2d, columns=["x", "y"])  # 创建包含 x, y 坐标的 DataFrame
        if labels is not None:
            df_out["label"] = labels  # 如果有标签，添加到 DataFrame
        df_out.to_csv(args.output_csv, index=False)  # 保存为 CSV（不包含行索引）
        print(f"💾 已导出坐标 CSV: {args.output_csv}")
        print(f"   包含列: {list(df_out.columns)}")

    # ========== 步骤 8: 可视化 ==========
    # 创建 2D 散点图，展示 t-SNE 降维结果
    # 如果有标签，用不同颜色表示不同类别；如果没有标签，用单一颜色
    plt.figure(figsize=(10, 8))  # 设置图像大小（宽×高，单位：英寸）
    
    if labels is None:
        # 无标签模式：所有点用相同颜色
        plt.scatter(
            feats_2d[:, 0],      # x 坐标（t-SNE 第一维）
            feats_2d[:, 1],      # y 坐标（t-SNE 第二维）
            s=5,                 # 点的大小（像素）
            alpha=0.7,           # 透明度（0-1，0.7 表示 70% 不透明）
            c="#1f77b4"          # 点的颜色（蓝色）
        )
        print("🔹 可视化模式: 无标签（所有点用相同颜色）")
    else:
        # 有标签模式：不同类别用不同颜色
        # 将标签转换为分类类型，保证颜色映射稳定（相同标签总是相同颜色）
        labels_flat = pd.Series(labels).astype("category")
        colors = labels_flat.cat.codes  # 获取类别编码（0, 1, 2, ...）
        
        # 绘制散点图，使用颜色映射（tab20 调色板支持最多 20 种不同颜色）
        scatter = plt.scatter(
            feats_2d[:, 0],      # x 坐标
            feats_2d[:, 1],      # y 坐标
            s=5,                 # 点的大小
            alpha=0.7,           # 透明度
            c=colors,            # 颜色（根据标签编码）
            cmap="tab20"         # 颜色映射：使用 tab20 调色板（20 种不同颜色）
        )
        
        # 如果类别数 <= 20，绘制图例（显示每个颜色对应的类别）
        # 类别数过多时，图例会过于拥挤，因此不显示
        num_classes = labels_flat.nunique()
        if num_classes <= 20:
            handles, _ = scatter.legend_elements(num=num_classes)  # 获取图例句柄
            plt.legend(
                handles, 
                labels_flat.cat.categories,  # 图例标签（类别名称）
                title="Person ID",           # 图例标题
                bbox_to_anchor=(1.05, 1),     # 图例位置（图像右侧）
                loc="upper left",             # 图例对齐方式
                fontsize=8                    # 图例字体大小
            )
            print(f"🔹 可视化模式: 有标签（{num_classes} 个类别，已显示图例）")
        else:
            print(f"🔹 可视化模式: 有标签（{num_classes} 个类别，图例过多未显示）")
    
    # 设置图表标题和坐标轴标签
    plt.title(
        f"t-SNE Visualization\n(perplexity={args.perplexity}, learning_rate={args.learning_rate}, n_iter={args.n_iter})",
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.grid(True, alpha=0.3)  # 添加网格线（透明度 30%，便于读取坐标）
    
    # 调整布局并保存图像
    plt.tight_layout()  # 自动调整子图参数，避免标签被截断
    plt.savefig(args.output_png, dpi=300, bbox_inches='tight')  # 保存为 PNG（300 DPI 高分辨率）
    plt.close()  # 关闭图像，释放内存
    print(f"💾 已保存可视化图像: {args.output_png}")
    print(f"   图像尺寸: 10×8 英寸，分辨率: 300 DPI")


if __name__ == "__main__":
    main()
