#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试消融实验启动提示功能
"""

def print_ablation_experiment_info(config_file_path):
    """打印消融实验启动信息"""
    if "ablation_scale4_only" in config_file_path:
        print("=" * 80)
        print("🔥 消融实验启动：4×4小尺度滑动窗口实验")
        print("=" * 80)
        print("📊 实验配置：")
        print("   - 滑动窗口尺度：仅4×4小尺度")
        print("   - MoE融合：禁用")
        print("   - 特征类型：局部细节特征")
        print("   - 预期效果：捕获局部细节和纹理信息")
        print("   - 输出目录：ablation_scale4_only")
        print("=" * 80)
    elif "ablation_scale8_only" in config_file_path:
        print("=" * 80)
        print("🔥 消融实验启动：8×8中尺度滑动窗口实验")
        print("=" * 80)
        print("📊 实验配置：")
        print("   - 滑动窗口尺度：仅8×8中尺度")
        print("   - MoE融合：禁用")
        print("   - 特征类型：结构信息特征")
        print("   - 预期效果：捕获结构信息和对象部件")
        print("   - 输出目录：ablation_scale8_only")
        print("=" * 80)
    elif "ablation_scale16_only" in config_file_path:
        print("=" * 80)
        print("🔥 消融实验启动：16×16大尺度滑动窗口实验")
        print("=" * 80)
        print("📊 实验配置：")
        print("   - 滑动窗口尺度：仅16×16大尺度")
        print("   - MoE融合：禁用")
        print("   - 特征类型：全局上下文特征")
        print("   - 预期效果：捕获全局上下文和场景信息")
        print("   - 输出目录：ablation_scale16_only")
        print("=" * 80)
    elif "ablation" in config_file_path:
        print("=" * 80)
        print("🔥 消融实验启动：多尺度滑动窗口消融实验")
        print("=" * 80)
        print("📊 实验配置：")
        print("   - 滑动窗口尺度：多尺度组合")
        print("   - MoE融合：根据配置")
        print("   - 特征类型：多尺度特征融合")
        print("   - 预期效果：验证不同尺度组合的效果")
        print("=" * 80)

if __name__ == "__main__":
    # 测试不同的配置文件路径
    test_paths = [
        "configs/RGBNT201/ablation_scale4_only.yml",
        "configs/RGBNT201/ablation_scale8_only.yml", 
        "configs/RGBNT201/ablation_scale16_only.yml",
        "configs/RGBNT201/MambaPro.yml"
    ]
    
    for path in test_paths:
        print(f"\n测试路径: {path}")
        print_ablation_experiment_info(path)
        print()
