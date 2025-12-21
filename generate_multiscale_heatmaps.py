#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 multiscale 文件夹下的所有尺度组合权重生成热力图可视化

功能说明：
- 自动扫描 multiscale 文件夹下的所有尺度组合
- 为每个 MambaProbest.pth 权重生成热力图可视化
- 使用对应的数据集（RGBNT201）

使用方法：
python generate_multiscale_heatmaps.py
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ========== 配置参数 ==========

# multiscale 输出目录
MULTISCALE_DIR = '/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale'

# 数据集路径（所有 multiscale 实验都使用 RGBNT201）
DATASET_ROOT = '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201'

# 配置文件路径（所有 multiscale 实验使用相同的配置）
CONFIG_FILE = '/home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml'

# 输出目录
OUTPUT_BASE_DIR = '/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale_heatmaps'

# 要可视化的图像数量（按人员ID）
NUM_IMAGES = 10


def find_multiscale_weights():
    """
    查找所有 multiscale 文件夹下的 MambaProbest.pth 权重文件
    
    Returns:
        list: 权重文件信息列表，每个元素包含 (scale_name, weight_path)
    """
    weights = []
    
    # 查找所有包含 MambaProbest.pth 的文件夹
    pattern = os.path.join(MULTISCALE_DIR, '*', 'MambaProbest.pth')
    weight_files = glob.glob(pattern)
    
    for weight_path in sorted(weight_files):
        # 从路径中提取尺度组合名称
        # 例如: .../77.76_4x4+16x16_20251217_160700/MambaProbest.pth
        folder_name = os.path.basename(os.path.dirname(weight_path))
        scale_name = folder_name.split('_')[1] if '_' in folder_name else folder_name
        
        weights.append({
            'name': folder_name,  # 完整文件夹名称
            'scale': scale_name,  # 尺度组合（如 4x4+16x16）
            'weight_path': weight_path
        })
    
    return weights


def generate_heatmaps_for_weight(weight_info, output_dir):
    """
    为指定权重生成热力图
    
    Args:
        weight_info (dict): 权重信息字典
        output_dir (str): 输出目录
    """
    weight_name = weight_info['name']
    weight_path = weight_info['weight_path']
    scale_name = weight_info['scale']
    
    print(f"\n{'='*80}")
    print(f"📊 处理尺度组合: {scale_name}")
    print(f"   文件夹: {weight_name}")
    print(f"   权重路径: {weight_path}")
    print(f"{'='*80}")
    
    # 检查权重文件是否存在
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        return False
    
    # 检查配置文件是否存在
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 配置文件不存在: {CONFIG_FILE}")
        return False
    
    # 检查数据集路径是否存在
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 数据集路径不存在: {DATASET_ROOT}")
        return False
    
    # 创建输出目录
    scale_output_dir = os.path.join(output_dir, weight_name)
    os.makedirs(scale_output_dir, exist_ok=True)
    
    # 构建命令行
    cmd = [
        'python', 'test_heatmap_from_weight.py',
        '--weight_path', weight_path,
        '--config_file', CONFIG_FILE,
        '--dataset_root', DATASET_ROOT,
        '--num_images', str(NUM_IMAGES),
        '--output_dir', scale_output_dir,
        '--alpha', '0.4'
    ]
    
    print(f"\n🔄 执行命令:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # 执行命令
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            check=True
        )
        print(f"\n✅ 完成: {scale_name}")
        print(f"📁 输出目录: {scale_output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 失败: {scale_name}")
        print(f"   错误代码: {e.returncode}")
        return False


def main():
    """
    主函数
    """
    print("="*80)
    print("🎨 为 Multiscale 所有尺度组合生成热力图可视化")
    print("="*80)
    
    # 查找所有权重文件
    print(f"\n🔍 扫描 multiscale 文件夹...")
    weights = find_multiscale_weights()
    
    if not weights:
        print("❌ 未找到任何权重文件")
        return
    
    print(f"✅ 找到 {len(weights)} 个尺度组合:")
    for i, w in enumerate(weights, 1):
        print(f"   {i}. {w['scale']} ({w['name']})")
    
    # 创建输出目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 为每个权重生成热力图
    success_count = 0
    failed_count = 0
    
    for i, weight_info in enumerate(weights, 1):
        print(f"\n\n[{i}/{len(weights)}] 处理权重...")
        
        if generate_heatmaps_for_weight(weight_info, OUTPUT_BASE_DIR):
            success_count += 1
        else:
            failed_count += 1
    
    # 打印总结
    print("\n" + "="*80)
    print("🎉 批量生成完成！")
    print("="*80)
    print(f"✅ 成功: {success_count}/{len(weights)}")
    print(f"❌ 失败: {failed_count}/{len(weights)}")
    print(f"📁 结果保存在: {OUTPUT_BASE_DIR}")
    print("="*80)
    
    # 打印每个尺度组合的输出目录
    print("\n📋 各尺度组合输出目录:")
    for weight_info in weights:
        scale_output_dir = os.path.join(OUTPUT_BASE_DIR, weight_info['name'])
        if os.path.exists(scale_output_dir):
            png_files = glob.glob(os.path.join(scale_output_dir, '*.png'))
            print(f"   - {weight_info['scale']}: {len(png_files)} 个热力图文件")
            print(f"     路径: {scale_output_dir}")


if __name__ == '__main__':
    main()
