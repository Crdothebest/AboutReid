#!/usr/bin/env python3
"""
训练监控脚本
用于监控训练进度和提取关键指标

作者修改：创建训练监控脚本，实时跟踪训练进度
功能：监控训练日志，提取损失、准确率等关键指标
撤销方法：删除此文件
"""

import os
import time
import re
import json
from datetime import datetime

def monitor_training_log(log_file_path, experiment_name):
    """
    监控训练日志文件
    
    作者修改：创建训练日志监控函数
    功能：实时读取训练日志，提取关键指标
    撤销方法：删除此函数
    """
    print(f"🔍 监控 {experiment_name} 训练进度...")
    print(f"📁 日志文件: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"⚠️  日志文件不存在: {log_file_path}")
        return None
    
    # 提取关键指标
    metrics = {
        'epochs': [],
        'losses': [],
        'accuracies': [],
        'learning_rates': [],
        'timestamps': []
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            # 提取epoch信息
            if 'Epoch[' in line and 'Loss:' in line:
                # 提取epoch数
                epoch_match = re.search(r'Epoch\[(\d+)\]', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    
                    # 提取损失
                    loss_match = re.search(r'Loss: ([\d.]+)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        
                        # 提取准确率
                        acc_match = re.search(r'Acc: ([\d.]+)', line)
                        if acc_match:
                            acc = float(acc_match.group(1))
                            
                            # 提取学习率
                            lr_match = re.search(r'Base Lr: ([\d.e-]+)', line)
                            if lr_match:
                                lr = float(lr_match.group(1))
                                
                                metrics['epochs'].append(epoch)
                                metrics['losses'].append(loss)
                                metrics['accuracies'].append(acc)
                                metrics['learning_rates'].append(lr)
                                metrics['timestamps'].append(datetime.now().isoformat())
        
        # 计算统计信息
        if metrics['epochs']:
            latest_epoch = max(metrics['epochs'])
            latest_loss = metrics['losses'][-1] if metrics['losses'] else 0
            latest_acc = metrics['accuracies'][-1] if metrics['accuracies'] else 0
            best_acc = max(metrics['accuracies']) if metrics['accuracies'] else 0
            
            print(f"📊 {experiment_name} 当前状态:")
            print(f"   最新Epoch: {latest_epoch}")
            print(f"   最新损失: {latest_loss:.4f}")
            print(f"   最新准确率: {latest_acc:.4f} ({latest_acc*100:.2f}%)")
            print(f"   最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
            
            return {
                'experiment_name': experiment_name,
                'latest_epoch': latest_epoch,
                'latest_loss': latest_loss,
                'latest_accuracy': latest_acc,
                'best_accuracy': best_acc,
                'total_metrics': len(metrics['epochs'])
            }
        else:
            print(f"⚠️  未找到有效的训练指标")
            return None
            
    except Exception as e:
        print(f"❌ 监控训练日志时出错: {str(e)}")
        return None

def main():
    """
    主函数：监控两个实验的训练进度
    
    作者修改：创建训练监控主流程
    功能：同时监控基线实验和创新点实验的进度
    撤销方法：删除此函数
    """
    print("🔍 训练进度监控")
    print("=" * 50)
    
    # 监控基线实验
    baseline_log = "outputs/baseline_experiment/train_log.txt"
    baseline_status = monitor_training_log(baseline_log, "基线实验")
    
    print()
    
    # 监控创新点实验
    moe_log = "outputs/moe_innovation_experiment/train_log.txt"
    moe_status = monitor_training_log(moe_log, "创新点实验")
    
    # 生成对比报告
    if baseline_status and moe_status:
        print("\n📊 训练进度对比:")
        print(f"   基线实验 - Epoch: {baseline_status['latest_epoch']}, 准确率: {baseline_status['latest_accuracy']*100:.2f}%")
        print(f"   创新点实验 - Epoch: {moe_status['latest_epoch']}, 准确率: {moe_status['latest_accuracy']*100:.2f}%")
        
        if moe_status['latest_accuracy'] > baseline_status['latest_accuracy']:
            improvement = (moe_status['latest_accuracy'] - baseline_status['latest_accuracy']) * 100
            print(f"   ✅ 创新点领先: +{improvement:.2f}%")
        else:
            decline = (baseline_status['latest_accuracy'] - moe_status['latest_accuracy']) * 100
            print(f"   ⚠️  创新点落后: -{decline:.2f}%")

if __name__ == "__main__":
    main()
