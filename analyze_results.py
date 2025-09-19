#!/usr/bin/env python3
"""
实验结果分析脚本
用于分析完整的60个epoch训练结果

作者修改：创建实验结果分析脚本
功能：分析基线实验和创新点实验的最终性能指标
撤销方法：删除此文件
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime

def extract_final_metrics(log_file_path):
    """
    从训练日志中提取最终性能指标
    
    作者修改：创建性能指标提取函数
    功能：从训练日志中提取mAP、Rank-1、Rank-5等最终指标
    撤销方法：删除此函数
    """
    if not os.path.exists(log_file_path):
        print(f"⚠️  日志文件不存在: {log_file_path}")
        return None
    
    metrics = {
        'final_loss': 0,
        'final_accuracy': 0,
        'best_accuracy': 0,
        'mAP': 0,
        'Rank-1': 0,
        'Rank-5': 0,
        'total_epochs': 0,
        'training_time': 0
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取最终损失和准确率
        loss_pattern = r'Loss: ([\d.]+)'
        acc_pattern = r'Acc: ([\d.]+)'
        
        losses = re.findall(loss_pattern, content)
        accuracies = re.findall(acc_pattern, content)
        
        if losses:
            metrics['final_loss'] = float(losses[-1])
        if accuracies:
            metrics['final_accuracy'] = float(accuracies[-1])
            metrics['best_accuracy'] = max([float(acc) for acc in accuracies])
        
        # 提取epoch信息
        epoch_pattern = r'Epoch\[(\d+)\]'
        epochs = re.findall(epoch_pattern, content)
        if epochs:
            metrics['total_epochs'] = max([int(epoch) for epoch in epochs])
        
        # 提取mAP和Rank指标（如果存在）
        map_pattern = r'mAP: ([\d.]+)'
        rank1_pattern = r'Rank-1: ([\d.]+)'
        rank5_pattern = r'Rank-5: ([\d.]+)'
        
        map_matches = re.findall(map_pattern, content)
        rank1_matches = re.findall(rank1_pattern, content)
        rank5_matches = re.findall(rank5_pattern, content)
        
        if map_matches:
            metrics['mAP'] = float(map_matches[-1])
        if rank1_matches:
            metrics['Rank-1'] = float(rank1_matches[-1])
        if rank5_matches:
            metrics['Rank-5'] = float(rank5_matches[-1])
        
        return metrics
        
    except Exception as e:
        print(f"❌ 分析日志文件时出错: {str(e)}")
        return None

def generate_comparison_report(baseline_metrics, moe_metrics, output_dir):
    """
    生成对比报告
    
    作者修改：创建实验结果对比报告生成函数
    功能：对比基线实验和创新点实验的最终结果
    撤销方法：删除此函数
    """
    print("\n📊 生成最终对比报告...")
    
    if not baseline_metrics or not moe_metrics:
        print("❌ 无法生成对比报告：缺少必要的指标数据")
        return
    
    # 计算提升幅度
    improvements = {}
    for key in ['final_accuracy', 'best_accuracy', 'mAP', 'Rank-1', 'Rank-5']:
        if baseline_metrics[key] > 0 and moe_metrics[key] > 0:
            improvement = moe_metrics[key] - baseline_metrics[key]
            improvement_percent = (improvement / baseline_metrics[key]) * 100
            improvements[key] = {
                'absolute': improvement,
                'percentage': improvement_percent
            }
    
    # 生成报告
    report = {
        "experiment_summary": {
            "baseline": baseline_metrics,
            "innovation": moe_metrics
        },
        "performance_comparison": improvements,
        "conclusion": {}
    }
    
    # 生成结论
    if improvements.get('final_accuracy', {}).get('absolute', 0) > 0:
        report["conclusion"]["accuracy"] = "✅ 创新点在最终准确率上有提升"
    else:
        report["conclusion"]["accuracy"] = "❌ 创新点在最终准确率上无提升"
    
    if improvements.get('mAP', {}).get('absolute', 0) > 0:
        report["conclusion"]["mAP"] = "✅ 创新点在mAP指标上有提升"
    else:
        report["conclusion"]["mAP"] = "❌ 创新点在mAP指标上无提升"
    
    # 保存报告
    report_file = os.path.join(output_dir, "final_comparison_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    print("\n" + "="*60)
    print("📊 最终实验结果对比报告")
    print("="*60)
    
    print(f"\n🔬 基线实验结果:")
    print(f"   最终损失: {baseline_metrics['final_loss']:.4f}")
    print(f"   最终准确率: {baseline_metrics['final_accuracy']:.4f} ({baseline_metrics['final_accuracy']*100:.2f}%)")
    print(f"   最佳准确率: {baseline_metrics['best_accuracy']:.4f} ({baseline_metrics['best_accuracy']*100:.2f}%)")
    if baseline_metrics['mAP'] > 0:
        print(f"   mAP: {baseline_metrics['mAP']:.2f}%")
    if baseline_metrics['Rank-1'] > 0:
        print(f"   Rank-1: {baseline_metrics['Rank-1']:.2f}%")
    if baseline_metrics['Rank-5'] > 0:
        print(f"   Rank-5: {baseline_metrics['Rank-5']:.2f}%")
    
    print(f"\n🚀 创新点实验结果:")
    print(f"   最终损失: {moe_metrics['final_loss']:.4f}")
    print(f"   最终准确率: {moe_metrics['final_accuracy']:.4f} ({moe_metrics['final_accuracy']*100:.2f}%)")
    print(f"   最佳准确率: {moe_metrics['best_accuracy']:.4f} ({moe_metrics['best_accuracy']*100:.2f}%)")
    if moe_metrics['mAP'] > 0:
        print(f"   mAP: {moe_metrics['mAP']:.2f}%")
    if moe_metrics['Rank-1'] > 0:
        print(f"   Rank-1: {moe_metrics['Rank-1']:.2f}%")
    if moe_metrics['Rank-5'] > 0:
        print(f"   Rank-5: {moe_metrics['Rank-5']:.2f}%")
    
    print(f"\n📈 性能提升对比:")
    for key, improvement in improvements.items():
        if improvement['absolute'] > 0:
            print(f"   {key}: +{improvement['absolute']:.4f} (+{improvement['percentage']:.2f}%)")
        else:
            print(f"   {key}: {improvement['absolute']:.4f} ({improvement['percentage']:.2f}%)")
    
    print(f"\n🎯 结论:")
    for key, conclusion in report["conclusion"].items():
        print(f"   {conclusion}")
    
    print(f"\n📁 详细报告已保存到: {report_file}")

def main():
    """
    主函数：分析实验结果
    
    作者修改：创建结果分析主流程
    功能：分析基线实验和创新点实验的结果并生成对比报告
    撤销方法：删除此函数
    """
    parser = argparse.ArgumentParser(description="分析MambaPro实验结果")
    parser.add_argument("--baseline_log", required=True, help="基线实验日志文件路径")
    parser.add_argument("--moe_log", required=True, help="创新点实验日志文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    args = parser.parse_args()
    
    print("🔍 MambaPro 实验结果分析")
    print("=" * 50)
    print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 分析基线实验结果
    print("\n📊 分析基线实验结果...")
    baseline_metrics = extract_final_metrics(args.baseline_log)
    
    # 分析创新点实验结果
    print("📊 分析创新点实验结果...")
    moe_metrics = extract_final_metrics(args.moe_log)
    
    # 生成对比报告
    if baseline_metrics and moe_metrics:
        generate_comparison_report(baseline_metrics, moe_metrics, args.output_dir)
    else:
        print("❌ 无法生成对比报告：缺少必要的指标数据")

if __name__ == "__main__":
    main()
