#!/usr/bin/env python3
"""
实验结果按数据集分类脚本
将 results/everyExperiments 目录下的实验结果按照数据集名称进行分类整理
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict


def extract_dataset_name(experiment_dir):
    """
    从实验目录中提取数据集名称
    
    策略：
    1. 从训练日志文件中提取
    2. 从配置文件中提取
    3. 从experiment_info.txt中提取原始配置文件路径
    
    Args:
        experiment_dir: 实验目录路径
        
    Returns:
        str: 数据集名称，如果找不到返回 'Unknown'
    """
    dataset_name = None
    
    # 策略1: 从训练日志文件中提取
    log_file = os.path.join(experiment_dir, 'logs', 'train_log.txt')
    if os.path.exists(log_file):
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(log_file, 'r', encoding=encoding, errors='ignore') as f:
                        for line in f:
                            # 匹配数据集名称：DATASETS: NAMES: ('RGBNT201')
                            if 'DATASETS:' in line or 'NAMES:' in line:
                                dataset_match = re.search(r"NAMES:\s*\(['\"]?(\w+)['\"]?\)", line)
                                if dataset_match:
                                    dataset_name = dataset_match.group(1)
                                    break
                            # 也尝试匹配单独的数据集名称
                            elif 'RGBNT201' in line:
                                dataset_name = 'RGBNT201'
                                break
                            elif 'RGBNT100' in line:
                                dataset_name = 'RGBNT100'
                                break
                            elif 'MSVR310' in line:
                                dataset_name = 'MSVR310'
                                break
                    if dataset_name:
                        break
                except:
                    continue
        except:
            pass
    
    # 策略2: 从experiment_info.txt中提取原始配置文件路径
    if not dataset_name:
        info_file = os.path.join(experiment_dir, 'experiment_info.txt')
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # 从原始配置文件路径中提取数据集名称
                    if 'configs/RGBNT201/' in content:
                        dataset_name = 'RGBNT201'
                    elif 'configs/RGBNT100/' in content:
                        dataset_name = 'RGBNT100'
                    elif 'configs/MSVR310/' in content:
                        dataset_name = 'MSVR310'
            except:
                pass
    
    # 策略3: 从配置文件中提取
    if not dataset_name:
        config_file = os.path.join(experiment_dir, 'configs', 'experiment_config.yml')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'RGBNT201' in content:
                        dataset_name = 'RGBNT201'
                    elif 'RGBNT100' in content:
                        dataset_name = 'RGBNT100'
                    elif 'MSVR310' in content:
                        dataset_name = 'MSVR310'
            except:
                pass
    
    return dataset_name if dataset_name else 'Unknown'


def organize_experiments_by_dataset(base_dir='results/everyExperiments', 
                                    output_base='results/experiments_by_dataset',
                                    dry_run=False):
    """
    按数据集对实验结果进行分类整理
    
    Args:
        base_dir: 实验结果根目录
        output_base: 输出分类目录的根目录
        dry_run: 如果为True，只显示将要执行的操作，不实际移动文件
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 错误: 目录不存在: {base_dir}")
        return
    
    # 统计信息
    dataset_experiments = defaultdict(list)
    unknown_experiments = []
    
    print(f"📊 开始扫描实验结果目录: {base_dir}")
    print("=" * 80)
    
    # 扫描所有实验目录
    experiment_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('experiment_')])
    
    print(f"找到 {len(experiment_dirs)} 个实验目录\n")
    
    for exp_dir in experiment_dirs:
        dataset_name = extract_dataset_name(exp_dir)
        
        if dataset_name == 'Unknown':
            unknown_experiments.append(exp_dir.name)
            print(f"⚠️  {exp_dir.name}: 无法识别数据集")
        else:
            dataset_experiments[dataset_name].append(exp_dir.name)
            print(f"✅ {exp_dir.name}: {dataset_name}")
    
    print("\n" + "=" * 80)
    print("📈 统计结果:")
    print("=" * 80)
    
    for dataset_name, experiments in sorted(dataset_experiments.items()):
        print(f"  {dataset_name}: {len(experiments)} 个实验")
    
    if unknown_experiments:
        print(f"  Unknown: {len(unknown_experiments)} 个实验")
    
    print("\n" + "=" * 80)
    
    if dry_run:
        print("🔍 预览模式：以下是将要执行的操作（不会实际移动文件）")
        print("=" * 80)
    else:
        print("🚀 开始整理实验结果...")
        print("=" * 80)
    
    # 创建分类目录并移动实验
    output_path = Path(output_base)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    total_moved = 0
    
    for dataset_name, experiments in sorted(dataset_experiments.items()):
        dataset_dir = output_path / dataset_name
        
        if not dry_run:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 数据集: {dataset_name}")
        print(f"   目标目录: {dataset_dir}")
        
        for exp_name in experiments:
            src = base_path / exp_name
            dst = dataset_dir / exp_name
            
            if dry_run:
                print(f"   [预览] {exp_name} -> {dst}")
            else:
                if dst.exists():
                    print(f"   ⚠️  跳过 {exp_name} (目标已存在)")
                else:
                    try:
                        shutil.move(str(src), str(dst))
                        print(f"   ✅ 移动 {exp_name}")
                        total_moved += 1
                    except Exception as e:
                        print(f"   ❌ 移动失败 {exp_name}: {str(e)}")
    
    # 处理Unknown数据集
    if unknown_experiments:
        unknown_dir = output_path / 'Unknown'
        if not dry_run:
            unknown_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 数据集: Unknown")
        print(f"   目标目录: {unknown_dir}")
        
        for exp_name in unknown_experiments:
            src = base_path / exp_name
            dst = unknown_dir / exp_name
            
            if dry_run:
                print(f"   [预览] {exp_name} -> {dst}")
            else:
                if dst.exists():
                    print(f"   ⚠️  跳过 {exp_name} (目标已存在)")
                else:
                    try:
                        shutil.move(str(src), str(dst))
                        print(f"   ✅ 移动 {exp_name}")
                        total_moved += 1
                    except Exception as e:
                        print(f"   ❌ 移动失败 {exp_name}: {str(e)}")
    
    print("\n" + "=" * 80)
    if dry_run:
        print("✅ 预览完成！使用 --execute 参数来实际执行移动操作")
    else:
        print(f"✅ 整理完成！共移动 {total_moved} 个实验目录")
        print(f"📁 分类结果保存在: {output_base}")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='按数据集对实验结果进行分类整理')
    parser.add_argument('--base_dir', type=str, default='results/everyExperiments',
                       help='实验结果根目录 (默认: results/everyExperiments)')
    parser.add_argument('--output', type=str, default='results/experiments_by_dataset',
                       help='输出分类目录的根目录 (默认: results/experiments_by_dataset)')
    parser.add_argument('--execute', action='store_true',
                       help='实际执行移动操作（默认是预览模式）')
    
    args = parser.parse_args()
    
    organize_experiments_by_dataset(
        base_dir=args.base_dir,
        output_base=args.output,
        dry_run=not args.execute
    )


if __name__ == '__main__':
    main()

