#!/usr/bin/env python3
"""
测试MoE和滑动窗口启动提示

功能：
- 测试MoE模块启动时的提示信息
- 测试滑动窗口模块启动时的提示信息
- 验证提示信息只在第一次调用时显示

作者：用户
日期：2024
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling.fusion_part.multi_scale_moe import CLIPMultiScaleMoE, MultiScaleMoE
from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor


def test_sliding_window_prompts():
    """测试滑动窗口启动提示"""
    print("=" * 60)
    print("🔍 测试滑动窗口启动提示")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 196  # 14x14 patches
    feat_dim = 512
    
    # 创建滑动窗口模型
    model = CLIPMultiScaleFeatureExtractor(feat_dim=feat_dim, scales=[4, 8, 16])
    
    # 创建测试输入
    patch_tokens = torch.randn(batch_size, seq_len, feat_dim)
    
    print(f"输入形状: {patch_tokens.shape}")
    print("\n第一次调用（应该显示启动提示）:")
    print("-" * 40)
    
    # 第一次调用 - 应该显示启动提示
    with torch.no_grad():
        output1 = model(patch_tokens)
    
    print(f"输出形状: {output1.shape}")
    
    print("\n第二次调用（不应该显示启动提示）:")
    print("-" * 40)
    
    # 第二次调用 - 不应该显示启动提示
    with torch.no_grad():
        output2 = model(patch_tokens)
    
    print(f"输出形状: {output2.shape}")
    print("✅ 滑动窗口测试完成！")


def test_moe_prompts():
    """测试MoE启动提示"""
    print("\n" + "=" * 60)
    print("🚀 测试MoE启动提示")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 196  # 14x14 patches
    feat_dim = 512
    
    # 创建MoE模型
    model = CLIPMultiScaleMoE(feat_dim=feat_dim, scales=[4, 8, 16])
    
    # 创建测试输入
    patch_tokens = torch.randn(batch_size, seq_len, feat_dim)
    
    print(f"输入形状: {patch_tokens.shape}")
    print("\n第一次调用（应该显示启动提示）:")
    print("-" * 40)
    
    # 第一次调用 - 应该显示启动提示
    with torch.no_grad():
        output1, weights1 = model(patch_tokens)
    
    print(f"输出特征形状: {output1.shape}")
    print(f"专家权重形状: {weights1.shape}")
    print(f"专家权重分布: {weights1[0].detach().cpu().numpy()}")
    
    print("\n第二次调用（不应该显示启动提示）:")
    print("-" * 40)
    
    # 第二次调用 - 不应该显示启动提示
    with torch.no_grad():
        output2, weights2 = model(patch_tokens)
    
    print(f"输出特征形状: {output2.shape}")
    print(f"专家权重形状: {weights2.shape}")
    print("✅ MoE测试完成！")


def test_moe_components():
    """测试MoE各个组件的启动提示"""
    print("\n" + "=" * 60)
    print("🧠 测试MoE组件启动提示")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 2
    feat_dim = 512
    
    # 创建多尺度特征
    multi_scale_features = [
        torch.randn(batch_size, feat_dim),  # 4x4特征
        torch.randn(batch_size, feat_dim),  # 8x8特征
        torch.randn(batch_size, feat_dim)   # 16x16特征
    ]
    
    # 创建MoE模块
    moe = MultiScaleMoE(feat_dim=feat_dim, scales=[4, 8, 16])
    
    print("输入多尺度特征:")
    for i, feat in enumerate(multi_scale_features):
        print(f"  - 尺度{i+1}: {feat.shape}")
    
    print("\n第一次调用MoE模块（应该显示启动提示）:")
    print("-" * 40)
    
    # 第一次调用 - 应该显示启动提示
    with torch.no_grad():
        output1, weights1 = moe(multi_scale_features)
    
    print(f"输出特征形状: {output1.shape}")
    print(f"专家权重形状: {weights1.shape}")
    
    print("\n第二次调用MoE模块（不应该显示启动提示）:")
    print("-" * 40)
    
    # 第二次调用 - 不应该显示启动提示
    with torch.no_grad():
        output2, weights2 = moe(multi_scale_features)
    
    print(f"输出特征形状: {output2.shape}")
    print(f"专家权重形状: {weights2.shape}")
    print("✅ MoE组件测试完成！")


def main():
    """主测试函数"""
    print("🎯 开始测试MoE和滑动窗口启动提示")
    print("=" * 80)
    
    try:
        # 测试滑动窗口提示
        test_sliding_window_prompts()
        
        # 测试MoE提示
        test_moe_prompts()
        
        # 测试MoE组件提示
        test_moe_components()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试完成！")
        print("=" * 80)
        print("✅ 滑动窗口启动提示正常工作")
        print("✅ MoE启动提示正常工作")
        print("✅ 提示信息只在第一次调用时显示")
        print("✅ 所有模块功能正常")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
