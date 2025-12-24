# Category: dev_utils (开发调试)
# Description: 开发辅助工具，包括进程清理、层输出调试、环境诊断及后端 API

#!/usr/bin/env python3
"""
删除所有状态为"运行中"的实验
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


def find_running_experiments(base_dir='results'):
    """
    查找所有状态为"运行中"的实验
    
    Args:
        base_dir: 实验结果根目录
        
    Returns:
        list: [(实验目录路径, 状态信息), ...]
    """
    running_experiments = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 错误: 目录不存在: {base_dir}")
        return running_experiments
    
    # 递归查找所有experiment_info.txt文件
    for info_file in base_path.rglob('experiment_info.txt'):
        try:
            with open(info_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # 检查状态
                if '状态: 运行中' in content or '状态:运行中' in content:
                    exp_dir = info_file.parent
                    running_experiments.append((exp_dir, info_file))
        except Exception as e:
            print(f"⚠️  读取文件失败: {info_file}, 错误: {str(e)}")
    
    return running_experiments


def cleanup_running_experiments(base_dir='results', dry_run=False):
    """
    删除所有状态为"运行中"的实验
    
    Args:
        base_dir: 实验结果根目录
        dry_run: 如果为True，只显示将要执行的操作，不实际删除
    """
    running_experiments = find_running_experiments(base_dir)
    
    if not running_experiments:
        print("✅ 未找到状态为'运行中'的实验")
        return
    
    print(f"📊 找到 {len(running_experiments)} 个状态为'运行中'的实验")
    print("=" * 80)
    
    # 按数据集和类型分组统计
    stats = defaultdict(int)
    
    for exp_dir, info_file in running_experiments:
        # 尝试提取数据集和实验类型信息
        parts = exp_dir.parts
        dataset = "Unknown"
        exp_type = "Unknown"
        
        # 从路径中提取信息
        for i, part in enumerate(parts):
            if part in ['MSVR310', 'RGBNT100', 'RGBNT201']:
                dataset = part
                if i + 1 < len(parts):
                    exp_type = parts[i + 1]
                break
        
        stats[f"{dataset}/{exp_type}"] += 1
        
        if dry_run:
            print(f"[预览] 将删除: {exp_dir}")
            print(f"       数据集: {dataset}, 类型: {exp_type}")
        else:
            print(f"🗑️  删除: {exp_dir}")
    
    print("\n" + "=" * 80)
    print("📈 统计结果:")
    print("=" * 80)
    
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count} 个实验")
    
    print(f"\n总计: {len(running_experiments)} 个实验")
    print("=" * 80)
    
    if dry_run:
        print("🔍 预览模式：以下是将要删除的实验（不会实际删除）")
        print("=" * 80)
    else:
        print("🚀 开始删除实验...")
        print("=" * 80)
    
    deleted_count = 0
    failed_count = 0
    
    for exp_dir, info_file in running_experiments:
        if dry_run:
            print(f"[预览] 删除: {exp_dir}")
        else:
            try:
                if exp_dir.exists():
                    shutil.rmtree(exp_dir)
                    print(f"   ✅ 已删除: {exp_dir.name}")
                    deleted_count += 1
                else:
                    print(f"   ⚠️  目录不存在: {exp_dir}")
            except Exception as e:
                print(f"   ❌ 删除失败: {exp_dir.name}, 错误: {str(e)}")
                failed_count += 1
    
    print("\n" + "=" * 80)
    if dry_run:
        print("✅ 预览完成！移除 --dry_run 参数来实际执行删除操作")
    else:
        print(f"✅ 清理完成！")
        print(f"   已删除: {deleted_count} 个实验")
        if failed_count > 0:
            print(f"   失败: {failed_count} 个实验")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='删除所有状态为"运行中"的实验')
    parser.add_argument('--base_dir', type=str, default='results',
                       help='实验结果根目录 (默认: results)')
    parser.add_argument('--dry_run', action='store_true',
                       help='预览模式，不实际删除')
    parser.add_argument('--yes', action='store_true',
                       help='自动确认，不询问')
    
    args = parser.parse_args()
    
    # 确认操作
    if not args.dry_run and not args.yes:
        print("⚠️  警告: 此操作将永久删除状态为'运行中'的实验！")
        response = input("确认继续？(yes/no): ")
        if response.lower() != 'yes':
            print("❌ 操作已取消")
            return
    
    cleanup_running_experiments(
        base_dir=args.base_dir,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()

