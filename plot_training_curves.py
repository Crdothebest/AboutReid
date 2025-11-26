#!/usr/bin/env python3
"""
训练曲线可视化脚本
用于从训练日志中提取指标并绘制epoch折线图

功能：
- 解析训练日志文件（train_log.txt）
- 提取训练指标：Loss, Accuracy, Learning Rate
- 提取验证指标：mAP, Rank-1, Rank-5, Rank-10
- 绘制多子图折线图
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_log_file(log_file_path):
    """
    解析训练日志文件，提取所有指标
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        dict: 包含所有指标的字典
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"日志文件不存在: {log_file_path}")
    
    metrics = {
        'train_epochs': [],
        'train_losses': [],
        'train_accuracies': [],
        'train_learning_rates': [],
        'val_epochs': [],
        'val_maps': [],
        'val_rank1': [],
        'val_rank5': [],
        'val_rank10': [],
        'dataset_name': 'Unknown',  # 数据集名称
    }
    
    # 尝试多种编码方式读取文件
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    lines = None
    for encoding in encodings:
        try:
            with open(log_file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    
    if lines is None:
        raise ValueError(f"无法读取日志文件: {log_file_path}，尝试了多种编码方式都失败")
    
    # 解析数据集名称
    for line in lines:
        # 匹配数据集名称：DATASETS: NAMES: ('RGBNT201') 或 DATASETS: NAMES: ('MSVR310')
        if 'DATASETS:' in line or 'NAMES:' in line:
            # 尝试提取数据集名称
            dataset_match = re.search(r"NAMES:\s*\(['\"]?(\w+)['\"]?\)", line)
            if dataset_match:
                metrics['dataset_name'] = dataset_match.group(1)
            # 也尝试匹配单独的数据集名称行
            elif 'RGBNT201' in line or 'RGBNT100' in line or 'MSVR310' in line:
                if 'RGBNT201' in line:
                    metrics['dataset_name'] = 'RGBNT201'
                elif 'RGBNT100' in line:
                    metrics['dataset_name'] = 'RGBNT100'
                elif 'MSVR310' in line:
                    metrics['dataset_name'] = 'MSVR310'
    
    # 解析训练指标
    for line in lines:
        # 匹配训练日志：Epoch[1] Iteration[10/100] Loss: 5.234, Acc: 0.123, Base Lr: 5.00e-04
        if 'Epoch[' in line and 'Loss:' in line and 'Acc:' in line:
            epoch_match = re.search(r'Epoch\[(\d+)\]', line)
            loss_match = re.search(r'Loss:\s+([\d.]+)', line)
            acc_match = re.search(r'Acc:\s+([\d.]+)', line)
            lr_match = re.search(r'Base Lr:\s+([\d.e-]+)', line)
            
            if epoch_match and loss_match and acc_match and lr_match:
                epoch = int(epoch_match.group(1))
                loss = float(loss_match.group(1))
                acc = float(acc_match.group(1))
                lr = float(lr_match.group(1))
                
                # 只记录每个epoch最后一次迭代的数据（通常是最后一个log_period）
                # 或者记录所有迭代，后面取平均值
                metrics['train_epochs'].append(epoch)
                metrics['train_losses'].append(loss)
                metrics['train_accuracies'].append(acc)
                metrics['train_learning_rates'].append(lr)
    
    # 解析验证指标 - 改进版本：按epoch分组解析
    current_val_epoch = None
    current_map = None
    current_rank1 = None
    current_rank5 = None
    current_rank10 = None
    
    for i, line in enumerate(lines):
        # 匹配验证结果开始：Validation Results - Epoch: 10
        if 'Validation Results - Epoch:' in line:
            epoch_match = re.search(r'Epoch:\s+(\d+)', line)
            if epoch_match:
                # 如果之前有未保存的验证结果，先保存
                if current_val_epoch is not None:
                    if current_map is not None:
                        metrics['val_epochs'].append(current_val_epoch)
                        metrics['val_maps'].append(current_map)
                        if current_rank1 is not None:
                            metrics['val_rank1'].append(current_rank1)
                        if current_rank5 is not None:
                            metrics['val_rank5'].append(current_rank5)
                        if current_rank10 is not None:
                            metrics['val_rank10'].append(current_rank10)
                
                # 开始新的验证epoch
                current_val_epoch = int(epoch_match.group(1))
                current_map = None
                current_rank1 = None
                current_rank5 = None
                current_rank10 = None
        
        # 匹配mAP：mAP: 45.2% (必须在Validation Results之后)
        elif current_val_epoch is not None and 'mAP:' in line and '%' in line and 'Best mAP' not in line:
            map_match = re.search(r'mAP:\s+([\d.]+)%', line)
            if map_match:
                current_map = float(map_match.group(1)) / 100.0  # 转换为小数
        
        # 匹配Rank-k：CMC curve, Rank-1  :45.2%
        elif current_val_epoch is not None and 'CMC curve' in line:
            rank_match = re.search(r'Rank-(\d+)\s*:([\d.]+)%', line)
            if rank_match:
                rank_k = int(rank_match.group(1))
                rank_val = float(rank_match.group(2)) / 100.0  # 转换为小数
                
                if rank_k == 1:
                    current_rank1 = rank_val
                elif rank_k == 5:
                    current_rank5 = rank_val
                elif rank_k == 10:
                    current_rank10 = rank_val
    
    # 保存最后一个验证结果
    if current_val_epoch is not None and current_map is not None:
        metrics['val_epochs'].append(current_val_epoch)
        metrics['val_maps'].append(current_map)
        if current_rank1 is not None:
            metrics['val_rank1'].append(current_rank1)
        if current_rank5 is not None:
            metrics['val_rank5'].append(current_rank5)
        if current_rank10 is not None:
            metrics['val_rank10'].append(current_rank10)
    
    # 处理训练指标：每个epoch可能有多个迭代记录，取平均值
    if metrics['train_epochs']:
        unique_epochs = sorted(set(metrics['train_epochs']))
        processed_train = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'learning_rates': []
        }
        
        for epoch in unique_epochs:
            # 找到该epoch的所有记录
            indices = [i for i, e in enumerate(metrics['train_epochs']) if e == epoch]
            if indices:
                # 取该epoch最后一次迭代的值（通常是最后一个）
                last_idx = indices[-1]
                processed_train['epochs'].append(epoch)
                processed_train['losses'].append(metrics['train_losses'][last_idx])
                processed_train['accuracies'].append(metrics['train_accuracies'][last_idx])
                processed_train['learning_rates'].append(metrics['train_learning_rates'][last_idx])
        
        metrics['train_epochs'] = processed_train['epochs']
        metrics['train_losses'] = processed_train['losses']
        metrics['train_accuracies'] = processed_train['accuracies']
        metrics['train_learning_rates'] = processed_train['learning_rates']
    
    return metrics


def plot_training_curves(metrics, save_path=None, show_plot=True):
    """
    绘制训练曲线
    
    Args:
        metrics: 从日志中提取的指标字典
        save_path: 保存图片的路径（可选）
        show_plot: 是否显示图片
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 构建标题，包含数据集信息
    dataset_name = metrics.get('dataset_name', 'Unknown')
    title = f'训练曲线可视化 - 数据集: {dataset_name}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. 训练损失
    ax1 = axes[0, 0]
    if metrics['train_epochs'] and metrics['train_losses']:
        ax1.plot(metrics['train_epochs'], metrics['train_losses'], 'b-', linewidth=2, label='训练损失')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('训练损失曲线', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. 训练准确率
    ax2 = axes[0, 1]
    if metrics['train_epochs'] and metrics['train_accuracies']:
        ax2.plot(metrics['train_epochs'], metrics['train_accuracies'], 'g-', linewidth=2, label='训练准确率')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('训练准确率曲线', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. 学习率
    ax3 = axes[0, 2]
    if metrics['train_epochs'] and metrics['train_learning_rates']:
        ax3.plot(metrics['train_epochs'], metrics['train_learning_rates'], 'r-', linewidth=2, label='学习率')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('学习率变化曲线', fontsize=13, fontweight='bold')
        ax3.set_yscale('log')  # 使用对数刻度
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. mAP
    ax4 = axes[1, 0]
    if metrics['val_epochs'] and metrics['val_maps']:
        ax4.plot(metrics['val_epochs'], [m * 100 for m in metrics['val_maps']], 'm-', 
                linewidth=2, marker='o', markersize=6, label='mAP')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('mAP (%)', fontsize=12)
        ax4.set_title('验证集 mAP 曲线', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # 5. Rank-k 准确率
    ax5 = axes[1, 1]
    if metrics['val_epochs']:
        if metrics['val_rank1']:
            ax5.plot(metrics['val_epochs'], [r * 100 for r in metrics['val_rank1']], 
                    'b-', linewidth=2, marker='o', markersize=6, label='Rank-1')
        if metrics['val_rank5']:
            ax5.plot(metrics['val_epochs'], [r * 100 for r in metrics['val_rank5']], 
                    'g-', linewidth=2, marker='s', markersize=6, label='Rank-5')
        if metrics['val_rank10']:
            ax5.plot(metrics['val_epochs'], [r * 100 for r in metrics['val_rank10']], 
                    'r-', linewidth=2, marker='^', markersize=6, label='Rank-10')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Accuracy (%)', fontsize=12)
        ax5.set_title('验证集 Rank-k 准确率曲线', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # 6. 综合对比图（损失和准确率）
    ax6 = axes[1, 2]
    if metrics['train_epochs']:
        ax6_twin = ax6.twinx()
        
        line1 = None
        line2 = None
        
        if metrics['train_losses']:
            line1 = ax6.plot(metrics['train_epochs'], metrics['train_losses'], 
                           'b-', linewidth=2, label='训练损失', alpha=0.7)
            ax6.set_xlabel('Epoch', fontsize=12)
            ax6.set_ylabel('Loss', fontsize=12, color='b')
            ax6.tick_params(axis='y', labelcolor='b')
        
        if metrics['train_accuracies']:
            line2 = ax6_twin.plot(metrics['train_epochs'], [a * 100 for a in metrics['train_accuracies']], 
                                'g-', linewidth=2, label='训练准确率', alpha=0.7)
            ax6_twin.set_ylabel('Accuracy (%)', fontsize=12, color='g')
            ax6_twin.tick_params(axis='y', labelcolor='g')
        
        # 合并图例
        lines = []
        if line1:
            lines.extend(line1)
        if line2:
            lines.extend(line2)
        if lines:
            labels = [l.get_label() for l in lines]
            ax6.legend(lines, labels, loc='upper left')
        ax6.set_title('训练损失与准确率对比', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图片已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def find_log_from_pth(pth_path):
    """
    根据pth文件路径自动查找对应的训练日志文件
    
    Args:
        pth_path: pth文件路径
        
    Returns:
        str: 训练日志文件路径，如果找不到返回None
    """
    pth_path = os.path.abspath(pth_path)
    
    # 如果pth文件不存在，返回None
    if not os.path.exists(pth_path):
        return None
    
    pth_dir = os.path.dirname(pth_path)
    
    # 策略1: 在同一目录下查找 train_log.txt
    log_path = os.path.join(pth_dir, 'train_log.txt')
    if os.path.exists(log_path):
        return log_path
    
    # 策略2: 在父目录的 logs 子目录下查找
    parent_dir = os.path.dirname(pth_dir)
    log_path = os.path.join(parent_dir, 'logs', 'train_log.txt')
    if os.path.exists(log_path):
        return log_path
    
    # 策略3: 在 models 目录的父目录的 logs 目录下查找
    if 'models' in pth_dir:
        parent_dir = os.path.dirname(pth_dir)
        log_path = os.path.join(parent_dir, 'logs', 'train_log.txt')
        if os.path.exists(log_path):
            return log_path
    
    # 策略4: 查找 experiment_info.txt，从中读取日志路径
    info_path = os.path.join(parent_dir, 'experiment_info.txt')
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 查找训练日志路径
                log_match = re.search(r'训练日志:\s*(.+)', content)
                if log_match:
                    log_path = log_match.group(1).strip()
                    if os.path.exists(log_path):
                        return log_path
        except:
            pass
    
    return None


def main():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--log_file', type=str, default=None,
                       help='训练日志文件路径 (train_log.txt)')
    parser.add_argument('--pth_file', type=str, default=None,
                       help='模型权重文件路径 (.pth)，将自动查找对应的训练日志')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（可选，默认保存在日志文件同目录）')
    parser.add_argument('--no_show', action='store_true',
                       help='不显示图片，仅保存')
    
    args = parser.parse_args()
    
    # 确定日志文件路径
    log_file_path = None
    
    if args.pth_file:
        # 如果提供了pth文件，尝试自动查找对应的训练日志
        print(f"🔍 根据pth文件查找训练日志: {args.pth_file}")
        log_file_path = find_log_from_pth(args.pth_file)
        if log_file_path:
            print(f"✅ 找到训练日志: {log_file_path}")
        else:
            print(f"❌ 无法自动找到对应的训练日志文件")
            print(f"   请手动指定训练日志路径: --log_file <路径>")
            return
    elif args.log_file:
        log_file_path = args.log_file
    else:
        print("❌ 错误: 必须提供 --log_file 或 --pth_file 参数")
        parser.print_help()
        return
    
    # 解析日志文件
    print(f"📊 正在解析日志文件: {log_file_path}")
    try:
        metrics = parse_log_file(log_file_path)
        
        # 打印统计信息
        print("\n📈 提取到的指标统计:")
        if metrics['train_epochs']:
            print(f"  训练数据: {len(metrics['train_epochs'])} 个epoch")
            print(f"  损失范围: {min(metrics['train_losses']):.4f} ~ {max(metrics['train_losses']):.4f}")
            print(f"  准确率范围: {min(metrics['train_accuracies']):.4f} ~ {max(metrics['train_accuracies']):.4f}")
        
        if metrics['val_epochs']:
            print(f"  验证数据: {len(metrics['val_epochs'])} 个epoch")
            if metrics['val_maps']:
                print(f"  mAP范围: {min(metrics['val_maps'])*100:.2f}% ~ {max(metrics['val_maps'])*100:.2f}%")
                print(f"  最佳mAP: {max(metrics['val_maps'])*100:.2f}% (Epoch {metrics['val_epochs'][metrics['val_maps'].index(max(metrics['val_maps']))]})")
            if metrics['val_rank1']:
                print(f"  Rank-1范围: {min(metrics['val_rank1'])*100:.2f}% ~ {max(metrics['val_rank1'])*100:.2f}%")
                print(f"  最佳Rank-1: {max(metrics['val_rank1'])*100:.2f}% (Epoch {metrics['val_epochs'][metrics['val_rank1'].index(max(metrics['val_rank1']))]})")
            if metrics['val_rank5']:
                print(f"  Rank-5范围: {min(metrics['val_rank5'])*100:.2f}% ~ {max(metrics['val_rank5'])*100:.2f}%")
                print(f"  最佳Rank-5: {max(metrics['val_rank5'])*100:.2f}% (Epoch {metrics['val_epochs'][metrics['val_rank5'].index(max(metrics['val_rank5']))]})")
            if metrics['val_rank10']:
                print(f"  Rank-10范围: {min(metrics['val_rank10'])*100:.2f}% ~ {max(metrics['val_rank10'])*100:.2f}%")
                print(f"  最佳Rank-10: {max(metrics['val_rank10'])*100:.2f}% (Epoch {metrics['val_epochs'][metrics['val_rank10'].index(max(metrics['val_rank10']))]})")
        else:
            print("  ⚠️  未找到验证数据，请检查日志文件是否包含验证结果")
        
        # 确定输出路径
        if args.output:
            save_path = args.output
        else:
            log_dir = os.path.dirname(log_file_path)
            save_path = os.path.join(log_dir, 'training_curves.png')
        
        # 绘制曲线
        print(f"\n🎨 正在绘制训练曲线...")
        plot_training_curves(metrics, save_path=save_path, show_plot=not args.no_show)
        
        print("\n✅ 完成！")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

