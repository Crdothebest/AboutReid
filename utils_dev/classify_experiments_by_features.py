# Category: dev_utils (开发调试)
# Description: 开发辅助工具，包括进程清理、层输出调试、环境诊断及后端 API

#!/usr/bin/env python3
"""
实验结果按创新点分类脚本
判断哪些是baseline实验，哪些使用了MoE、多尺度滑动窗口或其他创新点
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict


def extract_experiment_features(experiment_dir):
    """
    从实验目录中提取实验特征配置
    
    Args:
        experiment_dir: 实验目录路径
        
    Returns:
        dict: 包含实验特征的字典
    """
    features = {
        'use_multi_scale': False,      # 多尺度滑动窗口
        'use_moe': False,              # MoE模块
        'use_gate_fusion': False,      # 门控融合
        'moe_scales': None,            # MoE尺度
        'multi_scale_scales': None,    # 多尺度滑动窗口尺度
        'experiment_type': 'Unknown',  # 实验类型
    }
    
    # 策略1: 从训练日志中提取
    log_file = os.path.join(experiment_dir, 'logs', 'train_log.txt')
    if os.path.exists(log_file):
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(log_file, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                        
                        # 检查MoE状态
                        if 'MoE模块状态: 启用' in content or 'USE_MULTI_SCALE_MOE = True' in content:
                            features['use_moe'] = True
                        elif 'MoE模块状态: 禁用' in content or 'USE_MULTI_SCALE_MOE = False' in content:
                            features['use_moe'] = False
                        
                        # 检查多尺度滑动窗口状态
                        if '多尺度滑动窗口状态: 启用' in content or 'USE_CLIP_MULTI_SCALE = True' in content:
                            features['use_multi_scale'] = True
                        elif '多尺度滑动窗口状态: 禁用' in content or 'USE_CLIP_MULTI_SCALE = False' in content:
                            features['use_multi_scale'] = False
                        
                        # 检查门控融合状态
                        if 'USE_GATE_FUSION: True' in content or '门控融合机制: 启用' in content:
                            features['use_gate_fusion'] = True
                        elif 'USE_GATE_FUSION: False' in content or '门控融合机制: 禁用' in content:
                            features['use_gate_fusion'] = False
                        
                        # 提取MoE尺度
                        moe_scales_match = re.search(r'MoE滑动窗口尺度:\s*\[([\d,\s]+)\]', content)
                        if moe_scales_match:
                            scales_str = moe_scales_match.group(1)
                            features['moe_scales'] = [int(s.strip()) for s in scales_str.split(',')]
                        
                        # 提取多尺度滑动窗口尺度
                        multi_scale_match = re.search(r'CLIP_MULTI_SCALE_SCALES:\s*\[([\d,\s]+)\]', content)
                        if multi_scale_match:
                            scales_str = multi_scale_match.group(1)
                            features['multi_scale_scales'] = [int(s.strip()) for s in scales_str.split(',')]
                    
                    break
                except:
                    continue
        except:
            pass
    
    # 策略2: 从配置文件中提取
    config_file = os.path.join(experiment_dir, 'configs', 'experiment_config.yml')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # 检查USE_MULTI_SCALE_MOE
                if re.search(r'USE_MULTI_SCALE_MOE:\s*True', content):
                    features['use_moe'] = True
                elif re.search(r'USE_MULTI_SCALE_MOE:\s*False', content):
                    features['use_moe'] = False
                
                # 检查USE_CLIP_MULTI_SCALE
                if re.search(r'USE_CLIP_MULTI_SCALE:\s*True', content):
                    features['use_multi_scale'] = True
                elif re.search(r'USE_CLIP_MULTI_SCALE:\s*False', content):
                    features['use_multi_scale'] = False
                
                # 检查USE_GATE_FUSION
                if re.search(r'USE_GATE_FUSION:\s*True', content):
                    features['use_gate_fusion'] = True
                elif re.search(r'USE_GATE_FUSION:\s*False', content):
                    features['use_gate_fusion'] = False
                
                # 提取MoE尺度
                moe_scales_match = re.search(r'MOE_SCALES:\s*\[([\d,\s]+)\]', content)
                if moe_scales_match:
                    scales_str = moe_scales_match.group(1)
                    features['moe_scales'] = [int(s.strip()) for s in scales_str.split(',')]
                
                # 提取多尺度滑动窗口尺度
                multi_scale_match = re.search(r'CLIP_MULTI_SCALE_SCALES:\s*\[([\d,\s]+)\]', content)
                if multi_scale_match:
                    scales_str = multi_scale_match.group(1)
                    features['multi_scale_scales'] = [int(s.strip()) for s in scales_str.split(',')]
        except:
            pass
    
    # 判断实验类型
    if not features['use_multi_scale'] and not features['use_moe'] and not features['use_gate_fusion']:
        features['experiment_type'] = 'Baseline'
    elif features['use_multi_scale'] and not features['use_moe'] and not features['use_gate_fusion']:
        features['experiment_type'] = 'MultiScale'
    elif features['use_moe'] and not features['use_gate_fusion']:
        if features['use_multi_scale']:
            features['experiment_type'] = 'MoE+MultiScale'
        else:
            features['experiment_type'] = 'MoE'
    elif features['use_gate_fusion']:
        if features['use_moe'] and features['use_multi_scale']:
            features['experiment_type'] = 'Full'  # 全部启用
        elif features['use_moe']:
            features['experiment_type'] = 'MoE+GateFusion'
        else:
            features['experiment_type'] = 'GateFusion'
    else:
        features['experiment_type'] = 'Mixed'
    
    return features


def classify_experiments_by_features(base_dir='results', 
                                     output_base='results/experiments_by_features',
                                     dry_run=False):
    """
    按创新点对实验结果进行分类整理
    
    Args:
        base_dir: 实验结果根目录（包含各数据集文件夹）
        output_base: 输出分类目录的根目录
        dry_run: 如果为True，只显示将要执行的操作，不实际移动文件
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 错误: 目录不存在: {base_dir}")
        return
    
    # 统计信息
    type_experiments = defaultdict(lambda: defaultdict(list))
    unknown_experiments = defaultdict(list)
    
    print(f"📊 开始扫描实验结果目录: {base_dir}")
    print("=" * 80)
    
    # 扫描所有数据集目录
    dataset_dirs = [d for d in base_path.iterdir() 
                   if d.is_dir() and d.name in ['MSVR310', 'RGBNT100', 'RGBNT201']]
    
    if not dataset_dirs:
        print("❌ 未找到数据集目录（MSVR310, RGBNT100, RGBNT201）")
        return
    
    total_experiments = 0
    
    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name
        print(f"\n📁 处理数据集: {dataset_name}")
        
        # 扫描该数据集下的所有实验
        experiment_dirs = sorted([d for d in dataset_dir.iterdir() 
                                 if d.is_dir() and d.name.startswith('experiment_')])
        
        print(f"   找到 {len(experiment_dirs)} 个实验")
        
        for exp_dir in experiment_dirs:
            total_experiments += 1
            features = extract_experiment_features(exp_dir)
            exp_type = features['experiment_type']
            
            # 构建特征描述
            feature_desc = []
            if features['use_multi_scale']:
                scales = features['multi_scale_scales'] or 'Unknown'
                feature_desc.append(f"MultiScale{scales}")
            if features['use_moe']:
                scales = features['moe_scales'] or 'Unknown'
                feature_desc.append(f"MoE{scales}")
            if features['use_gate_fusion']:
                feature_desc.append("GateFusion")
            
            feature_str = "+".join(feature_desc) if feature_desc else "None"
            
            if exp_type == 'Unknown':
                unknown_experiments[dataset_name].append((exp_dir.name, feature_str))
                print(f"   ⚠️  {exp_dir.name}: {exp_type} ({feature_str})")
            else:
                type_experiments[dataset_name][exp_type].append((exp_dir.name, feature_str))
                print(f"   ✅ {exp_dir.name}: {exp_type} ({feature_str})")
    
    print("\n" + "=" * 80)
    print("📈 统计结果:")
    print("=" * 80)
    
    for dataset_name in sorted(type_experiments.keys()):
        print(f"\n  {dataset_name}:")
        for exp_type in sorted(type_experiments[dataset_name].keys()):
            count = len(type_experiments[dataset_name][exp_type])
            print(f"    {exp_type}: {count} 个实验")
        
        if dataset_name in unknown_experiments:
            print(f"    Unknown: {len(unknown_experiments[dataset_name])} 个实验")
    
    print(f"\n总计: {total_experiments} 个实验")
    print("=" * 80)
    
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
    
    # 按数据集和实验类型组织
    for dataset_name in sorted(type_experiments.keys()):
        dataset_output_dir = output_path / dataset_name
        if not dry_run:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        for exp_type in sorted(type_experiments[dataset_name].keys()):
            type_dir = dataset_output_dir / exp_type
            if not dry_run:
                type_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📁 {dataset_name} / {exp_type}")
            print(f"   目标目录: {type_dir}")
            
            for exp_name, feature_str in type_experiments[dataset_name][exp_type]:
                src = base_path / dataset_name / exp_name
                dst = type_dir / exp_name
                
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
    
    # 处理Unknown实验
    for dataset_name in sorted(unknown_experiments.keys()):
        dataset_output_dir = output_path / dataset_name
        unknown_dir = dataset_output_dir / 'Unknown'
        if not dry_run:
            unknown_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 {dataset_name} / Unknown")
        print(f"   目标目录: {unknown_dir}")
        
        for exp_name, feature_str in unknown_experiments[dataset_name]:
            src = base_path / dataset_name / exp_name
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
    
    parser = argparse.ArgumentParser(description='按创新点对实验结果进行分类整理')
    parser.add_argument('--base_dir', type=str, default='results',
                       help='实验结果根目录 (默认: results)')
    parser.add_argument('--output', type=str, default='results/experiments_by_features',
                       help='输出分类目录的根目录 (默认: results/experiments_by_features)')
    parser.add_argument('--execute', action='store_true',
                       help='实际执行移动操作（默认是预览模式）')
    
    args = parser.parse_args()
    
    classify_experiments_by_features(
        base_dir=args.base_dir,
        output_base=args.output,
        dry_run=not args.execute
    )


if __name__ == '__main__':
    main()

