#!/usr/bin/env python3
"""
批量绘制所有实验的训练曲线折线图
"""

import os
import subprocess
from pathlib import Path
from collections import defaultdict


def find_all_experiments(base_dir='results/experiments_by_features'):
    """
    查找所有实验目录及其训练日志
    
    Args:
        base_dir: 实验结果根目录
        
    Returns:
        list: [(实验目录路径, 训练日志路径), ...]
    """
    experiments = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 错误: 目录不存在: {base_dir}")
        return experiments
    
    # 遍历所有数据集目录
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        # 遍历所有实验类型目录
        for type_dir in dataset_dir.iterdir():
            if not type_dir.is_dir():
                continue
            
            # 遍历所有实验目录
            for exp_dir in type_dir.iterdir():
                if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment_'):
                    continue
                
                # 查找训练日志文件
                log_file = exp_dir / 'logs' / 'train_log.txt'
                if log_file.exists():
                    experiments.append((exp_dir, log_file))
                else:
                    print(f"⚠️  未找到训练日志: {log_file}")
    
    return experiments


def batch_plot_training_curves(base_dir='results/experiments_by_features', 
                                script_path='plot_training_curves.py',
                                dry_run=False):
    """
    批量绘制所有实验的训练曲线
    
    Args:
        base_dir: 实验结果根目录
        script_path: 绘制脚本路径
        dry_run: 如果为True，只显示将要执行的操作，不实际执行
    """
    experiments = find_all_experiments(base_dir)
    
    if not experiments:
        print("❌ 未找到任何实验")
        return
    
    print(f"📊 找到 {len(experiments)} 个实验")
    print("=" * 80)
    
    # 统计信息
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 按数据集和类型分组统计
    stats = defaultdict(lambda: defaultdict(int))
    
    for exp_dir, log_file in experiments:
        # 提取数据集和实验类型
        parts = exp_dir.parts
        if len(parts) >= 3:
            dataset = parts[-3]
            exp_type = parts[-2]
            exp_name = parts[-1]
            stats[dataset][exp_type] += 1
        
        # 检查是否已经存在训练曲线图
        existing_plot = exp_dir / 'logs' / 'training_curves.png'
        
        if existing_plot.exists() and not dry_run:
            print(f"⏭️  跳过 {exp_dir.name} (已存在训练曲线图)")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"[预览] 将绘制: {exp_dir.name}")
            print(f"       日志: {log_file}")
            print(f"       输出: {exp_dir / 'logs' / 'training_curves.png'}")
        else:
            print(f"🎨 正在绘制: {exp_dir.name}")
            
            try:
                # 执行绘制脚本
                result = subprocess.run(
                    ['python3', script_path, '--log_file', str(log_file), '--no_show'],
                    capture_output=True,
                    text=True,
                    timeout=60  # 60秒超时
                )
                
                if result.returncode == 0:
                    print(f"   ✅ 成功")
                    success_count += 1
                else:
                    print(f"   ❌ 失败: {result.stderr[:100]}")
                    failed_count += 1
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏱️  超时")
                failed_count += 1
            except Exception as e:
                print(f"   ❌ 错误: {str(e)}")
                failed_count += 1
    
    print("\n" + "=" * 80)
    print("📈 统计结果:")
    print("=" * 80)
    
    if not dry_run:
        print(f"✅ 成功: {success_count} 个")
        print(f"❌ 失败: {failed_count} 个")
        print(f"⏭️  跳过: {skipped_count} 个")
        print(f"📊 总计: {len(experiments)} 个")
    
    print("\n按数据集和类型统计:")
    for dataset in sorted(stats.keys()):
        print(f"\n  {dataset}:")
        for exp_type in sorted(stats[dataset].keys()):
            print(f"    {exp_type}: {stats[dataset][exp_type]} 个实验")
    
    print("\n" + "=" * 80)
    if dry_run:
        print("✅ 预览完成！移除 --dry_run 参数来实际执行")
    else:
        print("✅ 批量绘制完成！")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='批量绘制所有实验的训练曲线')
    parser.add_argument('--base_dir', type=str, default='results/experiments_by_features',
                       help='实验结果根目录 (默认: results/experiments_by_features)')
    parser.add_argument('--script', type=str, default='plot_training_curves.py',
                       help='绘制脚本路径 (默认: plot_training_curves.py)')
    parser.add_argument('--dry_run', action='store_true',
                       help='预览模式，不实际执行')
    
    args = parser.parse_args()
    
    batch_plot_training_curves(
        base_dir=args.base_dir,
        script_path=args.script,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()

