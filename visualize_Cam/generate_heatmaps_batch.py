#!/usr/bin/env python
"""
批量生成多模态热力图可视化脚本
为指定的模型权重生成多模态（RGB/NI/TI）热力图可视化
"""

import os
import sys
import random
import subprocess
from pathlib import Path

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)

# 配置参数
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

DATASET_ROOT = '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201'
OUTPUT_BASE_DIR = '/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs/Grad_CAM/batch_visualization'
NUM_PERSONS = 10
TEST_RGB_DIR = os.path.join(DATASET_ROOT, 'test', 'RGB')


def get_random_person_ids(num_persons=10):
    """从测试集中随机挑选指定数量的人员ID"""
    image_files = [f for f in os.listdir(TEST_RGB_DIR) if f.endswith('.jpg')]
    person_ids = list(set([f[:6] for f in image_files]))
    selected_ids = random.sample(person_ids, min(num_persons, len(person_ids)))
    selected_ids.sort()
    return selected_ids


def check_paths():
    """检查所有必需的路径是否存在"""
    print("🔍 检查路径...")
    
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 数据集路径不存在: {DATASET_ROOT}")
        return False
    
    if not os.path.exists(TEST_RGB_DIR):
        print(f"❌ 测试集RGB路径不存在: {TEST_RGB_DIR}")
        return False
    
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
    """为指定模型生成所有人员ID的热力图"""
    model_name = model_info['name']
    weight_path = model_info['path']
    config_file = os.path.join(script_dir, model_info['config'])
    
    print(f"\n{'='*60}")
    print(f"📊 处理模型: {model_name}")
    print(f"   权重路径: {weight_path}")
    print(f"   配置文件: {config_file}")
    print(f"{'='*60}")
    
    for i, person_id in enumerate(person_ids, 1):
        print(f"\n[{i}/{len(person_ids)}] 处理人员ID: {person_id}")
        
        output_path = os.path.join(output_dir, f"{model_name}_person_{person_id}.png")
        
        cmd = [
            'python', 'visualize_Cam/visualize_gradcam.py',
            '--config_file', config_file,
            '--weight_path', weight_path,
            '--query_id', person_id,
            '--dataset_root', DATASET_ROOT,
            '--output_dir', output_dir,
            '--multimodal'
        ]
        
        try:
            subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True, check=True)
            print(f"   ✅ 完成: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 失败: {e}")
            print(f"   错误输出: {e.stderr}")
            continue


def main():
    """主函数"""
    print("="*60)
    print("🎨 批量生成多模态热力图可视化")
    print("="*60)
    
    if not check_paths():
        print("\n❌ 路径检查失败，请检查配置")
        return
    
    print(f"\n🎲 从测试集随机挑选 {NUM_PERSONS} 个人员ID...")
    person_ids = get_random_person_ids(NUM_PERSONS)
    print(f"✅ 挑选的人员ID: {', '.join(person_ids)}")
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    for model_info in WEIGHT_PATHS:
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, model_info['name'])
        os.makedirs(model_output_dir, exist_ok=True)
        generate_heatmaps_for_model(model_info, person_ids, model_output_dir)
    
    print("\n" + "="*60)
    print("🎉 批量生成完成！")
    print(f"📁 结果保存在: {OUTPUT_BASE_DIR}")
    print("="*60)
    
    print("\n📋 结果摘要:")
    for model_info in WEIGHT_PATHS:
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, model_info['name'])
        if os.path.exists(model_output_dir):
            files = [f for f in os.listdir(model_output_dir) if f.endswith('.png')]
            print(f"  - {model_info['name']}: {len(files)} 个热力图文件")


if __name__ == '__main__':
    main()
