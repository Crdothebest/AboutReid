#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RGBNT100 数据集分析与调理脚本

功能：
1. 检查数据集目录结构
2. 验证图像文件完整性
3. 统计数据集信息（身份数、图像数、摄像头数等）
4. 检查文件命名格式
5. 验证数据加载逻辑
6. 生成数据集报告
"""

import os
import glob
import re
from pathlib import Path
from PIL import Image
from collections import defaultdict
import sys

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import cfg
from data.datasets.RGBNT100 import RGBNT100


def check_directory_structure(root_dir):
    """检查数据集目录结构"""
    print("=" * 80)
    print("📁 检查数据集目录结构")
    print("=" * 80)
    
    dataset_dir = os.path.join(root_dir, 'RGBNT100/rgbir')
    required_dirs = {
        'train': os.path.join(dataset_dir, 'bounding_box_train'),
        'query': os.path.join(dataset_dir, 'query'),
        'gallery': os.path.join(dataset_dir, 'bounding_box_test'),
    }
    
    results = {}
    for name, path in required_dirs.items():
        exists = os.path.exists(path)
        results[name] = {
            'exists': exists,
            'path': path
        }
        if exists:
            # 统计图像文件数量
            img_count = len(glob.glob(os.path.join(path, '*.jpg')))
            results[name]['image_count'] = img_count
            print(f"✅ {name:10s}: {path}")
            print(f"   图像数量: {img_count}")
        else:
            print(f"❌ {name:10s}: {path} (不存在)")
            results[name]['image_count'] = 0
    
    return results


def analyze_file_naming(dataset_dir):
    """分析文件命名格式"""
    print("\n" + "=" * 80)
    print("📝 分析文件命名格式")
    print("=" * 80)
    
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    
    for split in ['bounding_box_train', 'query', 'bounding_box_test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        img_paths = glob.glob(os.path.join(split_dir, '*.jpg'))
        if len(img_paths) == 0:
            print(f"\n{split}: 无图像文件")
            continue
        
        print(f"\n{split}:")
        valid_count = 0
        invalid_count = 0
        pid_set = set()
        camid_set = set()
        
        for img_path in img_paths[:10]:  # 只检查前10个作为示例
            img_name = os.path.basename(img_path)
            match = pattern.search(img_name)
            if match:
                pid, camid = map(int, match.groups())
                pid_set.add(pid)
                camid_set.add(camid)
                valid_count += 1
            else:
                invalid_count += 1
                print(f"  ⚠️  无效命名: {img_name}")
        
        # 统计所有文件
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            match = pattern.search(img_name)
            if match:
                pid, camid = map(int, match.groups())
                pid_set.add(pid)
                camid_set.add(camid)
                valid_count += 1
            else:
                invalid_count += 1
        
        print(f"  有效文件: {valid_count}")
        print(f"  无效文件: {invalid_count}")
        print(f"  PID 范围: {min(pid_set) if pid_set else 'N/A'} - {max(pid_set) if pid_set else 'N/A'} ({len(pid_set)} 个)")
        print(f"  Camera ID 范围: {min(camid_set) if camid_set else 'N/A'} - {max(camid_set) if camid_set else 'N/A'} ({len(camid_set)} 个)")


def check_image_integrity(dataset_dir, sample_size=10):
    """检查图像完整性"""
    print("\n" + "=" * 80)
    print("🖼️  检查图像完整性（采样检查）")
    print("=" * 80)
    
    for split in ['bounding_box_train', 'query', 'bounding_box_test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        img_paths = glob.glob(os.path.join(split_dir, '*.jpg'))
        if len(img_paths) == 0:
            continue
        
        print(f"\n{split}:")
        sample_paths = img_paths[:sample_size] if len(img_paths) >= sample_size else img_paths
        
        valid_count = 0
        invalid_count = 0
        
        for img_path in sample_paths:
            try:
                img = Image.open(img_path)
                width, height = img.size
                
                # 检查图像尺寸是否符合预期（768×128 或类似）
                if width >= 256 and height >= 128:
                    valid_count += 1
                    if valid_count <= 3:  # 只打印前3个的详细信息
                        print(f"  ✅ {os.path.basename(img_path)}: {width}×{height}")
                else:
                    invalid_count += 1
                    print(f"  ⚠️  {os.path.basename(img_path)}: 尺寸异常 {width}×{height}")
            except Exception as e:
                invalid_count += 1
                print(f"  ❌ {os.path.basename(img_path)}: 无法读取 - {e}")
        
        print(f"  采样检查: {valid_count} 有效, {invalid_count} 无效 (共检查 {len(sample_paths)} 张)")


def load_and_statistics(root_dir):
    """加载数据集并统计信息"""
    print("\n" + "=" * 80)
    print("📊 加载数据集并统计信息")
    print("=" * 80)
    
    try:
        dataset = RGBNT100(root=root_dir, verbose=True)
        
        print("\n数据集统计:")
        print(f"  训练集:")
        print(f"    - 身份数: {dataset.num_train_pids}")
        print(f"    - 图像数: {dataset.num_train_imgs}")
        print(f"    - 摄像头数: {dataset.num_train_cams}")
        print(f"    - 视角数: {dataset.num_train_vids}")
        
        print(f"  查询集:")
        print(f"    - 身份数: {dataset.num_query_pids}")
        print(f"    - 图像数: {dataset.num_query_imgs}")
        print(f"    - 摄像头数: {dataset.num_query_cams}")
        print(f"    - 视角数: {dataset.num_query_vids}")
        
        print(f"  图库集:")
        print(f"    - 身份数: {dataset.num_gallery_pids}")
        print(f"    - 图像数: {dataset.num_gallery_imgs}")
        print(f"    - 摄像头数: {dataset.num_gallery_cams}")
        print(f"    - 视角数: {dataset.num_gallery_vids}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_data_loading(dataset, sample_size=5):
    """检查数据加载逻辑"""
    print("\n" + "=" * 80)
    print("🔄 检查数据加载逻辑")
    print("=" * 80)
    
    if dataset is None:
        print("❌ 数据集未加载，跳过数据加载检查")
        return
    
    from data.datasets.bases import read_image
    
    print("\n测试图像读取:")
    for i, (img_list, pid, camid, trackid) in enumerate(dataset.train[:sample_size]):
        print(f"\n样本 {i+1}:")
        print(f"  PID: {pid}, Camera ID: {camid}, Track ID: {trackid}")
        print(f"  图像路径: {img_list}")
        
        try:
            images = read_image(img_list)
            print(f"  ✅ 成功读取 {len(images)} 个模态")
            for j, img in enumerate(images):
                print(f"    模态 {j}: {img.size if hasattr(img, 'size') else 'N/A'}")
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")


def check_config_paths():
    """检查配置文件中的路径设置"""
    print("\n" + "=" * 80)
    print("⚙️  检查配置文件路径设置")
    print("=" * 80)
    
    config_file = 'configs/RGBNT100/jzb_baseline_optimize.yml'
    if os.path.exists(config_file):
        print(f"✅ 配置文件存在: {config_file}")
        
        # 读取配置文件中的 ROOT_DIR
        import yaml
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            root_dir = config.get('DATASETS', {}).get('ROOT_DIR', '')
            print(f"  配置的 ROOT_DIR: {root_dir}")
            
            if root_dir:
                dataset_path = os.path.join(root_dir, 'RGBNT100/rgbir')
                if os.path.exists(dataset_path):
                    print(f"  ✅ 数据集路径存在: {dataset_path}")
                else:
                    print(f"  ❌ 数据集路径不存在: {dataset_path}")
                    print(f"  💡 建议检查路径配置")
        except Exception as e:
            print(f"  ⚠️  无法解析配置文件: {e}")
    else:
        print(f"❌ 配置文件不存在: {config_file}")


def generate_report(root_dir, dataset):
    """生成数据集报告"""
    print("\n" + "=" * 80)
    print("📋 数据集分析报告")
    print("=" * 80)
    
    report = []
    report.append("RGBNT100 数据集分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 目录结构
    report.append("1. 目录结构:")
    dataset_dir = os.path.join(root_dir, 'RGBNT100/rgbir')
    for split in ['bounding_box_train', 'query', 'bounding_box_test']:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            img_count = len(glob.glob(os.path.join(split_dir, '*.jpg')))
            report.append(f"   ✅ {split}: {img_count} 张图像")
        else:
            report.append(f"   ❌ {split}: 目录不存在")
    
    # 数据集统计
    if dataset:
        report.append("")
        report.append("2. 数据集统计:")
        report.append(f"   训练集: {dataset.num_train_pids} 个身份, {dataset.num_train_imgs} 张图像")
        report.append(f"   查询集: {dataset.num_query_pids} 个身份, {dataset.num_query_imgs} 张图像")
        report.append(f"   图库集: {dataset.num_gallery_pids} 个身份, {dataset.num_gallery_imgs} 张图像")
    
    # 数据特点
    report.append("")
    report.append("3. 数据特点:")
    report.append("   - RGBNT100 是 RGB-IR 双模态数据集")
    report.append("   - 图像格式: 768×128 水平拼接图像")
    report.append("   - RGB 部分: [0:256, 0:128]")
    report.append("   - NI 部分: [256:512, 0:128]")
    report.append("   - TI 部分: [512:768, 0:128] (虚拟，实际使用 NI)")
    report.append("   - 文件命名格式: PID_cCAMID_*.jpg")
    
    # 输出报告
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存报告
    report_file = 'outputs/rgbnt100_dataset_analysis_report.txt'
    os.makedirs('outputs', exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n📄 报告已保存: {report_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("🔍 RGBNT100 数据集分析与调理")
    print("=" * 80)
    print()
    
    # 从配置文件读取 ROOT_DIR
    config_file = 'configs/RGBNT100/jzb_baseline_optimize.yml'
    root_dir = '/home/zubuntu/workspace/MambaPro/MambaPro/data/'  # 默认路径
    
    if os.path.exists(config_file):
        import yaml
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            root_dir = config.get('DATASETS', {}).get('ROOT_DIR', root_dir)
        except:
            pass
    
    print(f"使用数据集根目录: {root_dir}")
    print()
    
    # 1. 检查目录结构
    dir_results = check_directory_structure(root_dir)
    
    # 2. 分析文件命名
    dataset_dir = os.path.join(root_dir, 'RGBNT100/rgbir')
    if os.path.exists(dataset_dir):
        analyze_file_naming(dataset_dir)
        check_image_integrity(dataset_dir)
    
    # 3. 检查配置文件
    check_config_paths()
    
    # 4. 加载数据集并统计
    dataset = load_and_statistics(root_dir)
    
    # 5. 检查数据加载逻辑
    check_data_loading(dataset)
    
    # 6. 生成报告
    generate_report(root_dir, dataset)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
