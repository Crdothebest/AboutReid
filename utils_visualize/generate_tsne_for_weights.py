# Category: vis_utils (可视化工具)
# Description: 提供热力图 (CAM)、t-SNE 降维、检索结果展示等模型可视化功能

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为指定权重文件提取特征并生成 t-SNE 可视化
支持多个权重文件的对比可视化
"""
import os
import sys
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import cfg
from data import make_dataloader
from modeling import make_model


def extract_features_from_weight(weight_path, config_file, device='cuda'):
    """
    从权重文件提取验证集特征
    
    Args:
        weight_path: 权重文件路径
        config_file: 配置文件路径
        device: 计算设备
        
    Returns:
        feats: 特征矩阵 (n_samples, feature_dim)
        labels: 标签数组 (n_samples,)
    """
    print(f"\n{'='*60}")
    print(f"提取特征: {os.path.basename(weight_path)}")
    print(f"{'='*60}")
    
    # 加载配置
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    # 创建数据加载器
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    # 创建模型
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    # 加载权重
    print(f"📦 加载权重: {weight_path}")
    model.load_param(weight_path)
    model.to(device)
    model.eval()
    
    # 提取特征
    print("🔍 提取特征中...")
    feats, labels = [], []
    with torch.no_grad():
        for batch_idx, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
            if batch_idx % 50 == 0:
                print(f"  处理批次: {batch_idx}/{len(val_loader)}")
            
            img = {k: v.to(device) for k, v in img.items()}  # RGB/NI/TI
            camids = camids.to(device)
            target_view = target_view.to(device)
            
            feat = model(img, cam_label=camids, view_label=target_view)
            feats.append(feat.cpu().numpy())
            
            # 处理 pid
            if isinstance(pid, tuple):
                pid = torch.tensor(pid, dtype=torch.int64)
            labels.append(pid.cpu().numpy())
    
    feats = np.vstack(feats)
    labels = np.concatenate(labels)
    
    print(f"✅ 特征提取完成: {feats.shape}, 标签: {labels.shape}")
    return feats, labels


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2 归一化特征向量"""
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def generate_tsne(feats, labels, output_path, perplexity=40, learning_rate=400, 
                  pca_dims=50, max_samples=5000, title_suffix=""):
    """
    生成 t-SNE 可视化
    
    Args:
        feats: 特征矩阵 (n_samples, feature_dim)
        labels: 标签数组 (n_samples,)
        output_path: 输出图片路径
        perplexity: t-SNE 困惑度
        learning_rate: t-SNE 学习率
        pca_dims: PCA 降维维度（0 表示不使用 PCA）
        max_samples: 最大采样数（0 表示不采样）
        title_suffix: 标题后缀
    """
    print(f"\n{'='*60}")
    print(f"生成 t-SNE 可视化")
    print(f"{'='*60}")
    
    # L2 归一化
    feats = normalize_l2(feats)
    print(f"📊 特征形状: {feats.shape}")
    
    # 随机采样（如果样本数过多）
    if max_samples > 0 and len(feats) > max_samples:
        print(f"📉 随机采样: {len(feats)} -> {max_samples}")
        indices = np.random.choice(len(feats), max_samples, replace=False)
        feats = feats[indices]
        labels = labels[indices]
    
    # PCA 降维（可选）
    if pca_dims > 0 and pca_dims < feats.shape[1]:
        print(f"🔧 PCA 降维: {feats.shape[1]} -> {pca_dims}")
        pca = PCA(n_components=pca_dims, random_state=42)
        feats = pca.fit_transform(feats)
        print(f"   PCA 解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    
    # t-SNE 降维
    print(f"🎨 运行 t-SNE (perplexity={perplexity}, lr={learning_rate})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    feats_2d = tsne.fit_transform(feats)
    
    # 可视化
    print("📊 绘制可视化图...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 获取唯一标签和颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            c=[colors[i]],
            label=f'ID {label}',
            alpha=0.6,
            s=20
        )
    
    ax.set_title(f't-SNE Visualization{title_suffix}\n(perp={perplexity}, lr={learning_rate})', 
              fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # 如果类别太多，不显示图例
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    else:
        ax.text(0.02, 0.98, f'{len(unique_labels)} classes', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 添加完整的黑色边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)  # 设置边框宽度为 2.0
        spine.set_visible(True)
    
    # 确保所有边框都可见
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # 设置整个图形的边框
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2.0)
    
    plt.tight_layout()
    
    # 保存图片，使用 edgecolor='black' 确保边框完整
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black', pad_inches=0.1)
    print(f"✅ 已保存: {output_path}")
    plt.close()


def find_multiscale_weights(multiscale_dir):
    """
    查找 multiscale 文件夹下所有尺度的最佳权重文件
    
    Args:
        multiscale_dir: multiscale 文件夹路径
        
    Returns:
        list: 权重文件信息列表，每个元素包含 'path' 和 'name'
    """
    weight_files = []
    
    if not os.path.exists(multiscale_dir):
        print(f"⚠️  multiscale 目录不存在: {multiscale_dir}")
        return weight_files
    
    # 查找所有 MambaProbest.pth 文件
    for root, dirs, files in os.walk(multiscale_dir):
        if 'MambaProbest.pth' in files:
            weight_path = os.path.join(root, 'MambaProbest.pth')
            # 从路径中提取模型名称（文件夹名）
            folder_name = os.path.basename(root)
            weight_files.append({
                'path': weight_path,
                'name': folder_name
            })
    
    return sorted(weight_files, key=lambda x: x['name'])


def main():
    # 配置参数
    config_file = "configs/RGBNT201/yzy_best_Mambapro_moe.yml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 自动查找 multiscale 文件夹下的所有最佳权重文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    multiscale_dir = os.path.join(script_dir, "outputs", "multiscale")
    
    print(f"🔍 查找 multiscale 文件夹下的权重文件: {multiscale_dir}")
    weight_files = find_multiscale_weights(multiscale_dir)
    
    if not weight_files:
        print("❌ 未找到任何权重文件")
        return
    
    print(f"✅ 找到 {len(weight_files)} 个权重文件:")
    for wf in weight_files:
        print(f"   - {wf['name']}: {wf['path']}")
    
    # t-SNE 参数
    perplexity = 40
    learning_rate = 400
    pca_dims = 50
    max_samples = 5000
    
    # 输出目录
    output_dir = "outputs/tsne"
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个权重文件生成 t-SNE
    for weight_info in weight_files:
        weight_path = weight_info['path']
        weight_name = weight_info['name']
        
        if not os.path.exists(weight_path):
            print(f"⚠️  权重文件不存在: {weight_path}")
            continue
        
        try:
            # 提取特征
            feats, labels = extract_features_from_weight(weight_path, config_file, device)
            
            # 生成 t-SNE 可视化
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"tsne_{weight_name}_{timestamp}.png")
            
            generate_tsne(
                feats, labels, output_path,
                perplexity=perplexity,
                learning_rate=learning_rate,
                pca_dims=pca_dims,
                max_samples=max_samples,
                title_suffix=f" - {weight_name}"
            )
            
        except Exception as e:
            print(f"❌ 处理失败 {weight_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ 所有 t-SNE 可视化生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
