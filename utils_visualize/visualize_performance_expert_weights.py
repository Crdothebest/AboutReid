# Category: vis_utils (可视化工具)
# Description: 提供热力图 (CAM)、t-SNE 降维、检索结果展示等模型可视化功能

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从训练日志中提取性能指标和专家权重，绘制双Y轴图
展示性能指标（mAP/Rank-1）与专家权重演化的协同关系
"""
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    """
    解析训练日志文件，提取性能指标和专家权重
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        dict: 包含epochs, maps, rank1s, expert_weights的字典
    """
    epochs = []
    maps = []
    rank1s = []
    expert_weights = []  # 每个元素是 [w1, w2, w3] 的列表
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_epoch = None
    current_map = None
    current_rank1 = None
    current_expert_weights = None
    
    for line in lines:
        # 提取 Epoch 编号
        epoch_match = re.search(r'Validation Results - Epoch: (\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # 提取 mAP
        map_match = re.search(r'Current mAP: ([\d.]+)%', line)
        if map_match:
            current_map = float(map_match.group(1))
        
        # 提取 Rank-1
        rank1_match = re.search(r'CMC curve, Rank-1\s*:([\d.]+)%', line)
        if rank1_match:
            current_rank1 = float(rank1_match.group(1))
        
        # 提取专家权重分布（支持2个或3个专家的格式）
        # 格式1: 📊 专家权重分布(Val): [w1, w2, w3]
        # 格式2: 📊 专家权重分布(Val, 2个专家): [w1, w2]
        expert_match = re.search(r'📊 专家权重分布\(Val(?:,\s*\d+个专家)?\): \[([\d.\s,]+)\]', line)
        if expert_match:
            weights_str = expert_match.group(1)
            # 解析权重列表，例如 "0.35 , 0.34 , 0.31" 或 "0.98 , 0.02"
            weights = [float(w.strip()) for w in weights_str.split(',')]
            current_expert_weights = weights
        
        # 如果所有信息都收集到了，添加到列表
        if current_epoch is not None and current_map is not None and \
           current_rank1 is not None and current_expert_weights is not None:
            epochs.append(current_epoch)
            maps.append(current_map)
            rank1s.append(current_rank1)
            expert_weights.append(current_expert_weights)
            
            # 重置，准备下一个epoch
            current_epoch = None
            current_map = None
            current_rank1 = None
            current_expert_weights = None
    
    return {
        'epochs': np.array(epochs),
        'maps': np.array(maps),
        'rank1s': np.array(rank1s),
        'expert_weights': np.array(expert_weights)  # shape: [n_epochs, 3]
    }


def plot_performance_expert_weights(data, output_path, title_suffix=""):
    """
    绘制双Y轴图：性能指标与专家权重演化
    
    Args:
        data: 从parse_log_file返回的字典
        output_path: 输出图片路径
        title_suffix: 标题后缀
    """
    epochs = data['epochs']
    maps = data['maps']
    rank1s = data['rank1s']
    expert_weights = data['expert_weights']  # [n_epochs, 3]
    
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ========== 左侧Y轴：性能指标（折线图）==========
    color_map = '#1E88E5'  # 更鲜艳的蓝色
    color_rank1 = '#D81B60'  # 更鲜艳的紫红色
    
    line1 = ax1.plot(epochs, maps, 'o-', color=color_map, linewidth=3.0, 
                     markersize=8, label='mAP', zorder=5, markerfacecolor='white', 
                     markeredgewidth=2, markeredgecolor=color_map)
    line2 = ax1.plot(epochs, rank1s, 's-', color=color_rank1, linewidth=3.0, 
                     markersize=8, label='Rank-1', zorder=5, markerfacecolor='white',
                     markeredgewidth=2, markeredgecolor=color_rank1)
    
    # 设置坐标轴标签（字体放大）
    ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=16, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    
    # 设置Y轴范围，从更标准的起始值开始（如40或50）
    min_perf = min(maps.min(), rank1s.min())
    perf_start = max(0, int(min_perf // 10) * 10 - 10)  # 向下取整到10的倍数，再减10
    perf_end = int(max(maps.max(), rank1s.max()) // 10) * 10 + 10  # 向上取整到10的倍数，再加10
    ax1.set_ylim(perf_start, perf_end)
    
    # 设置网格线，确保与右轴对齐
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax1.set_xlim(left=0)
    
    # 设置X轴刻度（每5个epoch一个标记）
    max_epoch = int(epochs.max())
    x_ticks = np.arange(0, max_epoch + 1, 5)
    ax1.set_xticks(x_ticks)
    
    # ========== 右侧Y轴：专家权重（堆叠面积图）==========
    ax2 = ax1.twinx()
    
    # 检测专家数量（支持2个或3个专家）
    n_experts = expert_weights.shape[1]
    
    if n_experts == 2:
        # 2个专家的配置（例如：8×8+16×16）
        colors_experts = ['#6C757D', '#B0C4DE']  # 深灰色、浅灰蓝色
        labels_experts = ['Scale 8×8', 'Scale 16×16']
        line_colors = ['#2F2F2F', '#4169E1']  # 深灰色、深蓝色
    else:
        # 3个专家的配置（默认：4×4+8×8+16×16）
        colors_experts = ['#FF6B6B', '#6C757D', '#B0C4DE']  # 珊瑚红、深灰色、浅灰蓝色
        labels_experts = ['Scale 4×4', 'Scale 8×8', 'Scale 16×16']
        line_colors = ['#8B0000', '#2F2F2F', '#4169E1']  # 深红色、深灰色、深蓝色
    
    # 计算堆叠位置
    bottoms = [np.zeros_like(epochs)]
    for i in range(n_experts - 1):
        bottoms.append(bottoms[-1] + expert_weights[:, i])
    
    # 绘制堆叠面积图（从下往上堆叠）
    for i in range(n_experts):
        bottom_start = bottoms[i]
        bottom_end = bottoms[i] + expert_weights[:, i] if i < n_experts - 1 else 1.0
        ax2.fill_between(epochs, bottom_start, bottom_end,
                          color=colors_experts[i], alpha=0.4, label=labels_experts[i], zorder=1)
    
    # 添加专家权重折线（使用对比色，加粗，确保清晰可见）
    for i in range(n_experts):
        ax2.plot(epochs, expert_weights[:, i], '--', color=line_colors[i],
                 linewidth=2.5, alpha=1.0, zorder=4)
    
    ax2.set_ylabel('Expert Weight Ratio', fontsize=16, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=14)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks(np.arange(0, 1.1, 0.2))  # 设置Y轴刻度
    
    # 确保右轴网格线与左轴对齐（通过设置相同的网格间隔）
    # 这里我们让右轴不显示网格，避免视觉干扰
    ax2.grid(False)
    
    # ========== 关键转折点标注 ==========
    # 找到性能跃升的关键区间（通常在第15-25个epoch之间）
    # 查找mAP增长最快的区间
    if len(epochs) > 1:
        map_diffs = np.diff(maps)
        rapid_growth_start_idx = np.argmax(map_diffs)
        rapid_growth_end_idx = min(rapid_growth_start_idx + 3, len(epochs) - 1)
        
        if rapid_growth_start_idx < len(epochs) - 1:
            rapid_start_epoch = epochs[rapid_growth_start_idx]
            rapid_end_epoch = epochs[rapid_growth_end_idx]
            
            # 添加垂直阴影区域标注性能跃升区间
            ax1.axvspan(rapid_start_epoch, rapid_end_epoch, alpha=0.15, 
                       color='orange', zorder=0, label='_nolegend_')
            
            # 添加箭头标注
            mid_epoch = (rapid_start_epoch + rapid_end_epoch) / 2
            mid_map = (maps[rapid_growth_start_idx] + maps[rapid_growth_end_idx]) / 2
            ax1.annotate('Phase of Rapid\nAdaptation', 
                        xy=(mid_epoch, mid_map),
                        xytext=(mid_epoch, mid_map + 10),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                        fontsize=11, fontweight='bold', color='orange',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='orange', alpha=0.8),
                        zorder=6)
    
    # ========== 最终性能标注 ==========
    # 在曲线终点标注具体数值
    if len(epochs) > 0:
        final_epoch = epochs[-1]
        final_map = maps[-1]
        final_rank1 = rank1s[-1]
        
        # 标注mAP
        ax1.annotate(f'{final_map:.1f}%', 
                    xy=(final_epoch, final_map),
                    xytext=(final_epoch + 2, final_map),
                    fontsize=12, fontweight='bold', color=color_map,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color_map, alpha=0.9),
                    zorder=6)
        
        # 标注Rank-1
        ax1.annotate(f'{final_rank1:.1f}%', 
                    xy=(final_epoch, final_rank1),
                    xytext=(final_epoch + 2, final_rank1),
                    fontsize=12, fontweight='bold', color=color_rank1,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color_rank1, alpha=0.9),
                    zorder=6)
    
    # ========== 图例和标题 ==========
    # 分离图例：性能指标放在左上角，专家权重放在右上角
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # 性能指标图例（左上角）
    legend1 = ax1.legend(lines1, labels1, 
                        loc='upper left', fontsize=12, framealpha=0.95,
                        edgecolor='black', fancybox=False, frameon=True)
    legend1.get_frame().set_linewidth(1.5)
    
    # 专家权重图例（右上角）
    legend2 = ax2.legend(lines2, labels2, 
                        loc='upper right', fontsize=12, framealpha=0.95,
                        edgecolor='black', fancybox=False, frameon=True)
    legend2.get_frame().set_linewidth(1.5)
    
    # 标题（字体放大）
    title = f'Performance Metrics and Expert Weight Evolution{title_suffix}'
    plt.title(title, fontsize=18, fontweight='bold', pad=25)
    
    # 添加黑色边框
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)
        spine.set_visible(True)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)
        spine.set_visible(True)
    
    # 设置整个图形的边框
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2.0)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black', pad_inches=0.1)
    print(f"✅ 已保存: {output_path}")
    plt.close()


def main():
    """主函数：批量处理日志文件"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从训练日志绘制性能与专家权重演化图')
    parser.add_argument('--log_path', type=str, required=True,
                        help='训练日志文件路径')
    parser.add_argument('--output_path', type=str, default=None,
                        help='输出图片路径（默认：日志文件同目录下的performance_expert_weights.png）')
    parser.add_argument('--title_suffix', type=str, default='',
                        help='标题后缀')
    
    args = parser.parse_args()
    
    # 解析日志文件
    print(f"📖 解析日志文件: {args.log_path}")
    data = parse_log_file(args.log_path)
    
    if len(data['epochs']) == 0:
        print("❌ 未找到任何验证结果数据")
        return
    
    n_experts = data['expert_weights'].shape[1]
    expert_labels = ['Scale 4×4', 'Scale 8×8', 'Scale 16×16']
    
    print(f"✅ 提取到 {len(data['epochs'])} 个epoch的验证结果")
    print(f"   Epoch范围: {data['epochs'].min()} - {data['epochs'].max()}")
    print(f"   mAP范围: {data['maps'].min():.1f}% - {data['maps'].max():.1f}%")
    print(f"   Rank-1范围: {data['rank1s'].min():.1f}% - {data['rank1s'].max():.1f}%")
    print(f"   专家权重范围 ({n_experts}个专家):")
    for i in range(n_experts):
        if n_experts == 1:
            label = 'Scale 4×4'
        elif n_experts == 2:
            label = expert_labels[i + 1]  # Scale 8×8, Scale 16×16
        else:
            label = expert_labels[i]
        print(f"     {label}: {data['expert_weights'][:, i].min():.2f} - {data['expert_weights'][:, i].max():.2f}")
    
    # 确定输出路径
    if args.output_path is None:
        log_dir = os.path.dirname(args.log_path)
        output_path = os.path.join(log_dir, 'performance_expert_weights.png')
    else:
        output_path = args.output_path
    
    # 绘制图表
    print(f"\n📊 绘制双Y轴图...")
    plot_performance_expert_weights(data, output_path, args.title_suffix)
    
    print(f"\n🎉 完成！")


if __name__ == '__main__':
    main()



