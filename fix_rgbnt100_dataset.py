#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RGBNT100 数据集整理与修复脚本

功能：
1. 检查数据集原始结构（R、N、T 目录）
2. 检查整理后的数据集结构（rgbir 目录）
3. 如果数据集未整理，运行整理脚本
4. 修复配置文件中的路径
5. 验证数据集完整性
"""

import os
import sys
import glob
from pathlib import Path

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def check_source_structure():
    """检查原始数据集结构（R、N、T 目录）"""
    print("=" * 80)
    print("📁 检查原始数据集结构（R、N、T 目录）")
    print("=" * 80)
    
    # 可能的源目录位置
    possible_sources = [
        '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100',
        '/home/zubuntu/workspace/MambaPro/MambaPro/data/datasets/RGBNT100',
    ]
    
    source_dir = None
    for path in possible_sources:
        if os.path.exists(path):
            source_dir = path
            break
    
    if source_dir is None:
        print("❌ 未找到原始数据集目录")
        print("   请检查以下路径:")
        for path in possible_sources:
            print(f"     - {path}")
        return None
    
    print(f"✅ 找到原始数据集: {source_dir}")
    
    # 检查 R、N、T 目录
    rgb_dir = os.path.join(source_dir, 'R')
    nir_dir = os.path.join(source_dir, 'N')
    tir_dir = os.path.join(source_dir, 'T')
    
    dirs_info = {}
    for name, path in [('R', rgb_dir), ('N', nir_dir), ('T', tir_dir)]:
        exists = os.path.exists(path)
        dirs_info[name] = {
            'exists': exists,
            'path': path
        }
        if exists:
            # 统计子目录（车辆ID目录）
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            dirs_info[name]['subdirs'] = len(subdirs)
            print(f"  ✅ {name} 目录: {len(subdirs)} 个车辆ID目录")
        else:
            print(f"  ❌ {name} 目录不存在")
            dirs_info[name]['subdirs'] = 0
    
    return source_dir, dirs_info


def check_organized_structure():
    """检查整理后的数据集结构（rgbir 目录）"""
    print("\n" + "=" * 80)
    print("📁 检查整理后的数据集结构（rgbir 目录）")
    print("=" * 80)
    
    # 可能的整理后目录位置
    possible_outputs = [
        '/home/zhanghaoyang/Desktop/yzy/RGBNT100/rgbir',
        '/home/zubuntu/workspace/MambaPro/MambaPro/data/RGBNT100/rgbir',
        '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/rgbir',
    ]
    
    output_dir = None
    for path in possible_outputs:
        if os.path.exists(path):
            output_dir = path
            break
    
    if output_dir is None:
        print("❌ 未找到整理后的数据集目录")
        print("   需要运行整理脚本")
        return None
    
    print(f"✅ 找到整理后的数据集: {output_dir}")
    
    # 检查各个子目录
    required_dirs = {
        'train': os.path.join(output_dir, 'bounding_box_train'),
        'query': os.path.join(output_dir, 'query'),
        'gallery': os.path.join(output_dir, 'bounding_box_test'),
    }
    
    dirs_info = {}
    for name, path in required_dirs.items():
        exists = os.path.exists(path)
        dirs_info[name] = {
            'exists': exists,
            'path': path
        }
        if exists:
            img_count = len(glob.glob(os.path.join(path, '*.jpg')))
            dirs_info[name]['image_count'] = img_count
            print(f"  ✅ {name:10s}: {img_count:5d} 张图像")
        else:
            print(f"  ❌ {name:10s}: 目录不存在")
            dirs_info[name]['image_count'] = 0
    
    return output_dir, dirs_info


def organize_dataset(source_dir):
    """运行数据集整理脚本"""
    print("\n" + "=" * 80)
    print("🔄 运行数据集整理脚本")
    print("=" * 80)
    
    organize_script = '/home/zhanghaoyang/Desktop/yzy/organize_rgbnt100.py'
    
    if not os.path.exists(organize_script):
        print(f"❌ 整理脚本不存在: {organize_script}")
        return False
    
    print(f"✅ 找到整理脚本: {organize_script}")
    print("   正在运行整理脚本...")
    
    try:
        # 导入并运行整理函数
        sys.path.insert(0, '/home/zhanghaoyang/Desktop/yzy')
        from organize_rgbnt100 import organize_rgbnt100_dataset
        
        # 运行整理（不创建验证集）
        organize_rgbnt100_dataset(validation_split=0)
        
        print("✅ 数据集整理完成！")
        return True
    except Exception as e:
        print(f"❌ 整理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_config_path():
    """修复配置文件中的路径"""
    print("\n" + "=" * 80)
    print("⚙️  修复配置文件路径")
    print("=" * 80)
    
    config_file = 'configs/RGBNT100/jzb_baseline_optimize.yml'
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    # 读取配置文件
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    current_root = config.get('DATASETS', {}).get('ROOT_DIR', '')
    print(f"当前 ROOT_DIR: {current_root}")
    
    # 检查当前路径是否存在
    dataset_path = os.path.join(current_root, 'RGBNT100/rgbir')
    if os.path.exists(dataset_path):
        print(f"✅ 当前路径有效: {dataset_path}")
        return True
    
    # 尝试找到正确的路径
    possible_roots = [
        '/home/zhanghaoyang/Desktop/yzy/MambaPro/data',
        '/home/zubuntu/workspace/MambaPro/MambaPro/data',
    ]
    
    correct_root = None
    for root in possible_roots:
        test_path = os.path.join(root, 'RGBNT100/rgbir')
        if os.path.exists(test_path):
            correct_root = root
            break
    
    if correct_root:
        print(f"💡 建议修改 ROOT_DIR 为: {correct_root}")
        print(f"   或者创建符号链接")
        
        # 询问是否修改（这里只提示，不自动修改）
        print("\n⚠️  如需修改配置文件，请手动编辑:")
        print(f"   {config_file}")
        print(f"   将 DATASETS.ROOT_DIR 改为: {correct_root}")
        return False
    else:
        print("❌ 未找到有效的数据集路径")
        return False


def create_symlink_if_needed():
    """如果需要，创建符号链接"""
    print("\n" + "=" * 80)
    print("🔗 检查符号链接")
    print("=" * 80)
    
    config_file = 'configs/RGBNT100/jzb_baseline_optimize.yml'
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    expected_root = config.get('DATASETS', {}).get('ROOT_DIR', '')
    expected_path = os.path.join(expected_root, 'RGBNT100')
    
    # 查找实际数据集位置
    actual_paths = [
        '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100',
        '/home/zhanghaoyang/Desktop/yzy/RGBNT100',
    ]
    
    actual_path = None
    for path in actual_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path and not os.path.exists(expected_path):
        print(f"💡 可以创建符号链接:")
        print(f"   ln -s {actual_path} {expected_path}")
        print(f"   这样配置文件中的路径就会生效")
    elif os.path.exists(expected_path):
        print(f"✅ 路径已存在: {expected_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("🔧 RGBNT100 数据集整理与修复")
    print("=" * 80)
    print()
    
    # 1. 检查原始数据集结构
    source_info = check_source_structure()
    
    # 2. 检查整理后的数据集结构
    organized_info = check_organized_structure()
    
    # 3. 如果数据集未整理，运行整理脚本
    if organized_info is None and source_info is not None:
        print("\n" + "=" * 80)
        print("⚠️  数据集未整理，需要运行整理脚本")
        print("=" * 80)
        
        source_dir, _ = source_info
        response = input("\n是否现在运行整理脚本？(y/n): ").strip().lower()
        if response == 'y':
            organize_dataset(source_dir)
        else:
            print("跳过整理，请稍后手动运行:")
            print("  python /home/zhanghaoyang/Desktop/yzy/organize_rgbnt100.py")
    
    # 4. 修复配置文件路径
    fix_config_path()
    
    # 5. 检查符号链接
    create_symlink_if_needed()
    
    print("\n" + "=" * 80)
    print("✅ 检查完成！")
    print("=" * 80)
    print("\n下一步:")
    print("1. 如果数据集未整理，运行: python /home/zhanghaoyang/Desktop/yzy/organize_rgbnt100.py")
    print("2. 检查配置文件中的 ROOT_DIR 路径是否正确")
    print("3. 如果路径不匹配，可以创建符号链接或修改配置文件")


if __name__ == '__main__':
    main()
