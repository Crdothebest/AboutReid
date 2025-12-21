#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 RGBNT100 数据集创建训练集、验证集、测试集划分

功能：
1. 从训练集中划分出验证集（10%）
2. 保持测试集（query + gallery）不变
3. 更新数据集结构
"""

import os
import shutil
import random
import glob
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def create_train_val_test_split(
    source_dir='/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/rgbir',
    val_split=0.1,
    random_seed=42
):
    """
    创建训练集、验证集、测试集划分
    
    Args:
        source_dir: 数据集根目录
        val_split: 验证集比例（从训练集中划分）
        random_seed: 随机种子
    """
    print("=" * 80)
    print("📊 创建训练集、验证集、测试集划分")
    print("=" * 80)
    
    train_dir = os.path.join(source_dir, 'bounding_box_train')
    query_dir = os.path.join(source_dir, 'query')
    gallery_dir = os.path.join(source_dir, 'bounding_box_test')
    val_dir = os.path.join(source_dir, 'bounding_box_val')
    
    # 检查目录是否存在
    if not os.path.exists(train_dir):
        print(f"❌ 训练集目录不存在: {train_dir}")
        return False
    
    if not os.path.exists(query_dir):
        print(f"❌ 查询集目录不存在: {query_dir}")
        return False
    
    if not os.path.exists(gallery_dir):
        print(f"❌ 图库集目录不存在: {gallery_dir}")
        return False
    
    # 如果验证集已存在，询问是否覆盖
    if os.path.exists(val_dir):
        print(f"⚠️  验证集目录已存在: {val_dir}")
        response = input("是否重新创建验证集？(y/n): ").strip().lower()
        if response == 'y':
            print(f"删除现有验证集目录...")
            shutil.rmtree(val_dir)
        else:
            print("跳过验证集创建")
            return True
    
    # 创建验证集目录
    os.makedirs(val_dir, exist_ok=True)
    print(f"✅ 创建验证集目录: {val_dir}")
    
    # 获取所有训练集图像
    train_images = glob.glob(os.path.join(train_dir, '*.jpg'))
    print(f"\n训练集原始图像数: {len(train_images)}")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 按车辆ID分组（从文件名提取）
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    
    images_by_pid = defaultdict(list)
    for img_path in train_images:
        img_name = os.path.basename(img_path)
        match = pattern.search(img_name)
        if match:
            pid = int(match.groups()[0])
            images_by_pid[pid].append(img_path)
    
    print(f"训练集车辆数: {len(images_by_pid)}")
    
    # 从每个车辆中划分验证集
    train_count = 0
    val_count = 0
    
    print(f"\n从训练集中划分 {val_split*100:.1f}% 作为验证集...")
    for pid, img_list in tqdm(images_by_pid.items(), desc="划分验证集"):
        # 打乱图像顺序
        shuffled = img_list.copy()
        random.shuffle(shuffled)
        
        # 计算验证集数量（至少1张，但不超过总数）
        val_size = max(1, int(len(shuffled) * val_split))
        val_size = min(val_size, len(shuffled) - 1)  # 确保训练集至少1张
        
        val_images = shuffled[:val_size]
        train_images_remaining = shuffled[val_size:]
        
        # 移动验证集图像
        for img_path in val_images:
            img_name = os.path.basename(img_path)
            val_img_path = os.path.join(val_dir, img_name)
            shutil.move(img_path, val_img_path)
            val_count += 1
        
        # 保留训练集图像（已在原位置，无需移动）
        train_count += len(train_images_remaining)
    
    print(f"\n划分完成:")
    print(f"  训练集: {train_count} 张图像")
    print(f"  验证集: {val_count} 张图像")
    print(f"  查询集: {len(glob.glob(os.path.join(query_dir, '*.jpg')))} 张图像")
    print(f"  图库集: {len(glob.glob(os.path.join(gallery_dir, '*.jpg')))} 张图像")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='创建训练集、验证集、测试集划分')
    parser.add_argument('--source_dir', type=str,
                        default='/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/rgbir',
                        help='数据集根目录')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例（默认0.1，即10%）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    create_train_val_test_split(
        source_dir=args.source_dir,
        val_split=args.val_split,
        random_seed=args.random_seed
    )
