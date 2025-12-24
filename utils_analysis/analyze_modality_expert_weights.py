# Category: analysis_utils (模型深度分析)
# Description: 深入分析 MoE 专家权重分配、模态梯度及模型层级选择逻辑

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析测试集中不同模态（RGB、NI、TI）对三个专家（4x4, 8x8, 16x16）的选择权重

实验目的：
通过定量分析，验证模型在面对物理特性迥异的模态（RGB、NI、TI）时，
是否具备"因材施教"的尺度选择能力。

数据采集方法：
1. 加载模型：使用验证集表现最好的权重文件（Best .pth）
2. 固定输入测试：
   - 步骤一：仅激活测试集中的 RGB 模态数据，运行推理，记录所有样本在 MoE 层输出的 Router 权重
   - 步骤二：对 NI 模态重复上述过程
   - 步骤三：对 TI 模态重复上述过程
3. 统计维度：计算平均值（Mean）和标准差（Standard Deviation）

图表可视化：
- 分组堆叠柱状图 (Grouped Stacked Bar Chart)
- X 轴：三个模态（RGB、NI、TI）
- Y 轴：权重占比 (0% - 100%)
- 颜色分层：每个柱子内部由三种颜色组成，分别对应 Scale 4、Scale 8、Scale 16
- 标注：在柱子上方标注该模态下主导专家的百分比数值
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import cfg
from data import make_dataloader
from modeling import make_model


def collect_expert_weights_by_modality(model, val_loader, device, num_samples=None):
    """
    收集测试集中不同模态的专家权重
    
    Args:
        model: 训练好的模型
        val_loader: 验证数据加载器
        device: 计算设备
        num_samples: 收集的样本数量（None表示全部）
    
    Returns:
        dict: 包含每个模态的专家权重列表
            {
                'RGB': [weights_list],  # 每个元素是 [B, num_experts]
                'NI': [weights_list],
                'TI': [weights_list]
            }
    """
    model.eval()
    
    # 获取 BACKBONE 模块
    backbone = model.BACKBONE if hasattr(model, 'BACKBONE') else model.module.BACKBONE
    
    if not hasattr(backbone, 'clip_multi_scale_moe'):
        print("❌ 模型未启用 MoE 模块，无法收集专家权重")
        return None
    
    modality_weights = {
        'RGB': [],
        'NI': [],
        'TI': []
    }
    
    sample_count = 0
    
    with torch.no_grad():
        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(tqdm(val_loader, desc="收集专家权重")):
            if num_samples is not None and sample_count >= num_samples:
                break
            
            img_dict = {
                'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            camids = camids.to(device)
            target_view = target_view.to(device)
            
            batch_size = img_dict['RGB'].shape[0]
            
            # 分别处理每个模态
            for modality_name in ['RGB', 'NI', 'TI']:
                # 获取模态标签
                if modality_name == 'RGB':
                    modality_label = 'rgb'
                elif modality_name == 'NI':
                    modality_label = 'nir'
                else:  # TI
                    modality_label = 'tir'
                
                try:
                    # 直接调用 BACKBONE 的 forward
                    # 这会触发 clip_multi_scale_moe，并设置 current_expert_weights
                    _ = backbone(
                        img_dict[modality_name],
                        cam_label=camids,
                        view_label=target_view,
                        modality=modality_label
                    )
                    
                    # 从 BACKBONE 获取专家权重
                    if hasattr(backbone, 'current_expert_weights'):
                        weights = backbone.current_expert_weights.detach().cpu()
                        modality_weights[modality_name].append(weights)
                    else:
                        print(f"⚠️  {modality_name} 模态：无法获取专家权重（current_expert_weights 不存在）")
                    
                except Exception as e:
                    print(f"⚠️  处理 {modality_name} 模态时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            sample_count += batch_size
    
    return modality_weights


def compute_statistics(modality_weights):
    """
    计算每个模态对三个专家的统计信息（平均值和标准差）
    
    Args:
        modality_weights: dict，包含每个模态的权重列表
    
    Returns:
        dict: 每个模态的统计信息
            {
                'RGB': {
                    'mean': [w_4x4, w_8x8, w_16x16],
                    'std': [std_4x4, std_8x8, std_16x16],
                    'samples': N
                },
                ...
            }
    """
    stats = {}
    
    for modality, weights_list in modality_weights.items():
        if len(weights_list) == 0:
            print(f"⚠️  {modality} 模态没有收集到权重")
            continue
        
        # 拼接所有batch的权重
        all_weights = torch.cat(weights_list, dim=0)  # [N, num_experts]
        
        # 计算平均值和标准差
        mean_weights = torch.mean(all_weights, dim=0).numpy()  # [num_experts]
        std_weights = torch.std(all_weights, dim=0).numpy()  # [num_experts]
        
        stats[modality] = {
            'mean': mean_weights,
            'std': std_weights,
            'samples': all_weights.shape[0]
        }
        
        print(f"\n{modality} 模态:")
        print(f"  样本数量: {all_weights.shape[0]}")
        print(f"  专家数量: {all_weights.shape[1]}")
        print(f"  平均权重: {mean_weights}")
        print(f"  标准差: {std_weights}")
    
    return stats


def plot_grouped_stacked_bar_chart(stats, output_path):
    """
    绘制分组堆叠柱状图
    
    Args:
        stats: dict，每个模态的统计信息
        output_path: 输出图片路径
    """
    modalities = list(stats.keys())
    if len(modalities) == 0:
        print("❌ 没有统计数据，无法绘制图表")
        return
    
    num_experts = len(stats[modalities[0]]['mean'])
    
    # 准备数据
    expert_labels = ['Scale 4×4', 'Scale 8×8', 'Scale 16×16']
    if num_experts == 2:
        expert_labels = ['Scale 8×8', 'Scale 16×16']
    elif num_experts == 1:
        expert_labels = ['Scale 4×4']
    
    # 提取每个专家的平均权重（转换为百分比）
    expert_data = {label: [] for label in expert_labels}
    expert_stds = {label: [] for label in expert_labels}
    dominant_expert = {}  # 记录每个模态的主导专家
    
    for modality in modalities:
        mean_weights = stats[modality]['mean']
        std_weights = stats[modality]['std']
        
        # 找到主导专家（权重最大的）
        dominant_idx = np.argmax(mean_weights)
        dominant_expert[modality] = {
            'expert': expert_labels[dominant_idx],
            'weight': mean_weights[dominant_idx] * 100
        }
        
        for i, label in enumerate(expert_labels):
            expert_data[label].append(mean_weights[i] * 100)  # 转换为百分比
            expert_stds[label].append(std_weights[i] * 100)  # 转换为百分比
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 设置颜色（学术风格）
    colors = ['#FF6B6B', '#6C757D', '#B0C4DE']  # 珊瑚红、深灰色、浅灰蓝色
    if num_experts == 2:
        colors = ['#6C757D', '#B0C4DE']
    elif num_experts == 1:
        colors = ['#FF6B6B']
    
    # 绘制堆叠柱状图
    x = np.arange(len(modalities))
    width = 0.6
    bottom = np.zeros(len(modalities))
    
    bars = []
    for i, (label, color) in enumerate(zip(expert_labels, colors)):
        values = expert_data[label]
        bars.append(ax.bar(x, values, width, bottom=bottom, label=label, color=color, alpha=0.8))
        bottom += values
    
    # 设置标签和标题
    ax.set_xlabel('Modality', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Expert Weight (%)', fontsize=16, fontweight='bold')
    ax.set_title('Expert Weight Distribution by Modality\n(Validating Router\'s Adaptive Selection)', 
                fontsize=18, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # 添加数值标签（主导专家的百分比）
    for i, modality in enumerate(modalities):
        dominant_info = dominant_expert[modality]
        total_height = 100
        
        # 在柱子顶部标注主导专家
        ax.text(i, total_height + 2, f"{dominant_info['expert']}\n{dominant_info['weight']:.1f}%", 
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='black', alpha=0.8))
        
        # 在每个专家区域添加数值标签（如果大于5%）
        current_bottom = 0
        for j, label in enumerate(expert_labels):
            value = expert_data[label][i]
            if value > 5:  # 只显示大于5%的标签
                ax.text(i, current_bottom + value/2, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            current_bottom += value
    
    # 添加黑色边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)
        spine.set_visible(True)
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black', pad_inches=0.1)
    print(f"\n✅ 分组堆叠柱状图已保存: {output_path}")
    plt.close()


def plot_radar_chart(stats, output_path):
    """
    绘制雷达图
    
    Args:
        stats: dict，每个模态的统计信息
        output_path: 输出图片路径
    """
    modalities = list(stats.keys())
    if len(modalities) == 0:
        return
    
    num_experts = len(stats[modalities[0]]['mean'])
    
    # 准备数据
    expert_labels = ['Scale 4×4', 'Scale 8×8', 'Scale 16×16']
    if num_experts == 2:
        expert_labels = ['Scale 8×8', 'Scale 16×16']
    elif num_experts == 1:
        expert_labels = ['Scale 4×4']
    
    # 设置角度
    angles = np.linspace(0, 2 * np.pi, num_experts, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 设置颜色
    colors = ['#1E88E5', '#D81B60', '#FFC107']  # 蓝色、紫红色、黄色
    
    # 绘制每个模态的雷达图
    for i, modality in enumerate(modalities):
        mean_weights = stats[modality]['mean'] * 100  # 转换为百分比
        values = mean_weights.tolist()
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=modality, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(expert_labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置标题和图例
    ax.set_title('Expert Weight Distribution by Modality (Radar Chart)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black', pad_inches=0.1)
    print(f"✅ 雷达图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='分析不同模态对专家的选择权重')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--config_file', type=str, 
                        default='configs/RGBNT201/yzy_best_Mambapro_moe.yml',
                        help='配置文件路径')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='收集的样本数量（None表示全部）')
    parser.add_argument('--output_dir', type=str, 
                        default='outputs/modality_expert_analysis',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("\n📊 加载测试数据...")
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print(f"测试集查询数量: {num_query}")
    
    # 加载模型
    print("\n🤖 加载模型...")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    # 加载权重
    print(f"📥 加载权重: {args.weight_path}")
    model.load_param(args.weight_path)
    model.to(device)
    model.eval()
    
    # 检查模型是否启用 MoE
    backbone = model.BACKBONE if hasattr(model, 'BACKBONE') else model.module.BACKBONE
    if not hasattr(backbone, 'clip_multi_scale_moe'):
        print("❌ 模型未启用 MoE 模块，无法分析专家权重")
        return
    
    print("✅ 模型已启用 MoE 模块")
    
    # 收集专家权重
    print("\n🔍 收集专家权重...")
    print("=" * 60)
    print("实验设计：")
    print("  步骤一：仅激活 RGB 模态，记录 Router 权重")
    print("  步骤二：仅激活 NI 模态，记录 Router 权重")
    print("  步骤三：仅激活 TI 模态，记录 Router 权重")
    print("=" * 60)
    
    modality_weights = collect_expert_weights_by_modality(
        model, val_loader, device, num_samples=args.num_samples
    )
    
    if modality_weights is None:
        print("❌ 未能收集到专家权重")
        return
    
    # 计算统计信息
    print("\n📈 计算统计信息（平均值和标准差）...")
    stats = compute_statistics(modality_weights)
    
    if len(stats) == 0:
        print("❌ 没有有效的统计数据")
        return
    
    # 绘制图表
    print("\n📊 绘制图表...")
    
    # 分组堆叠柱状图
    stacked_bar_path = os.path.join(args.output_dir, 'modality_expert_weights_stacked_bar.png')
    plot_grouped_stacked_bar_chart(stats, stacked_bar_path)
    
    # 雷达图
    radar_path = os.path.join(args.output_dir, 'modality_expert_weights_radar.png')
    plot_radar_chart(stats, radar_path)
    
    # 保存统计数据
    stats_path = os.path.join(args.output_dir, 'modality_expert_weights_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("不同模态对专家的选择权重统计\n")
        f.write("=" * 60 + "\n\n")
        f.write("实验目的：验证模型在面对物理特性迥异的模态时，是否具备\"因材施教\"的尺度选择能力\n\n")
        
        for modality, stat in stats.items():
            f.write(f"{modality} 模态:\n")
            f.write(f"  样本数量: {stat['samples']}\n")
            num_experts = len(stat['mean'])
            expert_labels = ['Scale 4×4', 'Scale 8×8', 'Scale 16×16']
            if num_experts == 2:
                expert_labels = ['Scale 8×8', 'Scale 16×16']
            elif num_experts == 1:
                expert_labels = ['Scale 4×4']
            
            f.write(f"  平均权重 (Mean):\n")
            for i, label in enumerate(expert_labels):
                f.write(f"    {label}: {stat['mean'][i]*100:.2f}%\n")
            
            f.write(f"  标准差 (Std):\n")
            for i, label in enumerate(expert_labels):
                f.write(f"    {label}: {stat['std'][i]*100:.2f}%\n")
            
            # 主导专家
            dominant_idx = np.argmax(stat['mean'])
            f.write(f"  主导专家: {expert_labels[dominant_idx]} ({stat['mean'][dominant_idx]*100:.2f}%)\n")
            f.write("\n")
    
    print(f"\n✅ 统计数据已保存: {stats_path}")
    print(f"\n🎉 分析完成！")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"\n📊 结果解读：")
    print("  - 如果三个模态的权重分布不同 → 证明 Router 根据模态特性动态选择（\"因材施教\"）")
    print("  - 如果三个模态的权重分布相同 → 说明 Router 可能是随机的或未学习到模态差异")


if __name__ == '__main__':
    main()
