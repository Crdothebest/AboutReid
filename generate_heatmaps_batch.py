#!/usr/bin/env python
"""
批量生成多模态热力图可视化脚本

功能说明：
为指定的两个模型权重生成多模态（RGB/NI/TI）热力图可视化，
从测试集中随机挑选10个不同的人员ID进行可视化。

使用方法：
python generate_heatmaps_batch.py

作者：MambaPro团队
日期：2024
"""

import os
import sys
import random
import subprocess
from pathlib import Path

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ========== 配置参数 ==========

# 模型权重路径
WEIGHT_PATHS = [
    {
        'name': '79.4mAP_baseline',
        'path': '/home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth',
        'config': 'configs/RGBNT201/yzy_best_Mambapro_moe.yml'
    },
    {
        'name': '77.76_multiscale_4x4+16x16',
        'path': '/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/multiscale/77.76_4x4+16x16_20251217_160700/MambaProbest.pth',
        'config': 'configs/RGBNT201/yzy_best_Mambapro_moe.yml'
    }
]

# 数据集路径
DATASET_ROOT = '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201'

# 输出目录
OUTPUT_BASE_DIR = '/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/Grad_CAM/batch_visualization'

# 从测试集挑选的人员ID数量
NUM_PERSONS = 10

# 测试集路径
TEST_RGB_DIR = os.path.join(DATASET_ROOT, 'test', 'RGB')


def get_random_person_ids(num_persons=10):
    """
    从测试集中随机挑选指定数量的人员ID
    
    Args:
        num_persons (int): 需要挑选的人员ID数量
        
    Returns:
        list: 人员ID列表，如 ['000123', '000456', ...]
    """
    # 获取所有测试图像
    image_files = [f for f in os.listdir(TEST_RGB_DIR) if f.endswith('.jpg')]
    
    # 提取所有唯一的人员ID（从文件名前6位）
    person_ids = list(set([f[:6] for f in image_files]))
    
    # 随机挑选指定数量的人员ID
    selected_ids = random.sample(person_ids, min(num_persons, len(person_ids)))
    selected_ids.sort()  # 排序以便于查看
    
    return selected_ids


def check_paths():
    """
    检查所有必需的路径是否存在
    """
    print("🔍 检查路径...")
    
    # 检查数据集路径
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 数据集路径不存在: {DATASET_ROOT}")
        return False
    
    # 检查测试集路径
    if not os.path.exists(TEST_RGB_DIR):
        print(f"❌ 测试集RGB路径不存在: {TEST_RGB_DIR}")
        return False
    
    # 检查模型权重路径
    for model_info in WEIGHT_PATHS:
        if not os.path.exists(model_info['path']):
            print(f"❌ 权重文件不存在: {model_info['path']}")
            return False
        
        config_path = os.path.join(script_dir, model_info['config'])
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
    
    print("✅ 所有路径检查通过")
    return True


def generate_heatmaps_for_model(model_info, person_ids, output_dir):
    """
    为指定模型生成所有人员ID的热力图
    
    Args:
        model_info (dict): 模型信息字典，包含 name, path, config
        person_ids (list): 人员ID列表
        output_dir (str): 输出目录
    """
    model_name = model_info['name']
    weight_path = model_info['path']
    config_file = os.path.join(script_dir, model_info['config'])
    
    print(f"\n{'='*60}")
    print(f"📊 处理模型: {model_name}")
    print(f"   权重路径: {weight_path}")
    print(f"   配置文件: {config_file}")
    print(f"{'='*60}")
    
    # 为每个人员ID生成多模态热力图
    for i, person_id in enumerate(person_ids, 1):
        print(f"\n[{i}/{len(person_ids)}] 处理人员ID: {person_id}")
        
        # 构建输出路径
        output_path = os.path.join(
            output_dir,
            f"{model_name}_person_{person_id}.png"
        )
        
        # 构建命令行
        cmd = [
            'python', 'visualize_gradcam.py',
            '--config_file', config_file,
            '--weight_path', weight_path,
            '--query_id', person_id,
            '--dataset_root', DATASET_ROOT,
            '--output_dir', output_dir,
            '--multimodal'
        ]
        
        # 执行命令
        try:
            result = subprocess.run(
                cmd,
                cwd=script_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"   ✅ 完成: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 失败: {e}")
            print(f"   错误输出: {e.stderr}")
            continue


def main():
    """
    主函数
    """
    print("="*60)
    print("🎨 批量生成多模态热力图可视化")
    print("="*60)
    
    # 检查路径
    if not check_paths():
        print("\n❌ 路径检查失败，请检查配置")
        return
    
    # 从测试集随机挑选人员ID
    print(f"\n🎲 从测试集随机挑选 {NUM_PERSONS} 个人员ID...")
    person_ids = get_random_person_ids(NUM_PERSONS)
    print(f"✅ 挑选的人员ID: {', '.join(person_ids)}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 为每个模型生成热力图
    for model_info in WEIGHT_PATHS:
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, model_info['name'])
        os.makedirs(model_output_dir, exist_ok=True)
        
        generate_heatmaps_for_model(model_info, person_ids, model_output_dir)
    
    print("\n" + "="*60)
    print("🎉 批量生成完成！")
    print(f"📁 结果保存在: {OUTPUT_BASE_DIR}")
    print("="*60)
    
    # 打印结果摘要
    print("\n📋 结果摘要:")
    for model_info in WEIGHT_PATHS:
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, model_info['name'])
        if os.path.exists(model_output_dir):
            files = [f for f in os.listdir(model_output_dir) if f.endswith('.png')]
            print(f"  - {model_info['name']}: {len(files)} 个热力图文件")


if __name__ == '__main__':
    main()
