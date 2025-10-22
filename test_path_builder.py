#!/usr/bin/env python3
"""
测试路径构建逻辑
"""
from pathlib import Path

def build_image_path(target_id, modality, rank_metric, model_type):
    """
    构建图片路径
    """
    # 模型类型映射
    model_file = 'baseline' if model_type == 'baseline' else 'your_model'
    
    # 构建路径
    path = f"/datasets/Rank_results/{modality}_rank-{rank_metric.replace('rank', '')}_results/run_20251017_175911/multimodal_ranked_list_{target_id}_top{rank_metric.replace('rank', '')}_{model_file}.png"
    
    return path

def test_paths():
    """
    测试各种路径组合
    """
    test_cases = [
        # (target_id, modality, rank_metric, model_type)
        ("000258", "RGB", "rank10", "baseline"),
        ("000258", "RGB", "rank10", "your_model"),
        ("000299", "RGB", "rank10", "baseline"),
        ("000299", "RGB", "rank10", "your_model"),
    ]
    
    print("测试路径构建：")
    print("=" * 80)
    
    for target_id, modality, rank_metric, model_type in test_cases:
        path = build_image_path(target_id, modality, rank_metric, model_type)
        print(f"ID: {target_id}, 模态: {modality}, 指标: {rank_metric}, 模型: {model_type}")
        print(f"路径: {path}")
        print("-" * 80)

if __name__ == "__main__":
    test_paths()
