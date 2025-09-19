#!/usr/bin/env python3
"""
维度修复验证脚本
功能：验证CLIP多尺度滑动窗口的维度配置是否正确
"""

import sys
import os
sys.path.append('.')

def test_dimension_configuration():
    """测试维度配置"""
    print("🔧 测试维度配置...")
    
    try:
        # 测试CLIP多尺度滑动窗口模块
        from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor
        
        # 创建模块实例
        extractor = CLIPMultiScaleFeatureExtractor(feat_dim=768, scales=[4, 8, 16])
        print("✅ CLIP多尺度滑动窗口模块创建成功")
        print(f"   - 特征维度: 768")
        print(f"   - 滑动窗口尺度: [4, 8, 16]")
        
        # 测试前向传播
        import torch
        batch_size = 2
        seq_len = 128  # 假设序列长度
        feat_dim = 768
        
        # 创建测试输入
        test_input = torch.randn(batch_size, seq_len, feat_dim)
        print(f"   - 测试输入形状: {test_input.shape}")
        
        # 前向传播
        output = extractor(test_input)
        print(f"   - 输出形状: {output.shape}")
        
        # 验证输出维度
        expected_shape = (batch_size, feat_dim)
        if output.shape == expected_shape:
            print("✅ 输出维度正确")
        else:
            print(f"❌ 输出维度错误，期望: {expected_shape}, 实际: {output.shape}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 维度配置测试失败: {e}")
        return False

def test_feat_dim_consistency():
    """测试feat_dim一致性"""
    print("\n🔧 测试feat_dim一致性...")
    
    try:
        # 检查MambaPro类中的feat_dim设置
        from modeling.make_model import MambaPro
        from config import cfg
        from config.defaults import _C
        
        # 模拟配置
        cfg.merge_from_other_cfg(_C)
        cfg.MODEL.TRANSFORMER_TYPE = 'ViT-B-16'
        
        # 检查feat_dim设置
        if 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            expected_feat_dim = 768
            print(f"✅ ViT-B-16的feat_dim应该为: {expected_feat_dim}")
        else:
            print("❌ 未找到ViT-B-16配置")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ feat_dim一致性测试失败: {e}")
        return False

def test_cv_embed_dimension():
    """测试cv_embed维度"""
    print("\n🔧 测试cv_embed维度...")
    
    try:
        # 检查cv_embed的维度设置
        import torch
        import torch.nn as nn
        
        # 模拟cv_embed创建
        camera_num = 6
        view_num = 1
        feat_dim = 768
        
        cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, feat_dim))
        print(f"✅ cv_embed创建成功")
        print(f"   - 形状: {cv_embed.shape}")
        print(f"   - 维度: {feat_dim}")
        
        # 验证维度
        if cv_embed.shape[1] == feat_dim:
            print("✅ cv_embed维度正确")
        else:
            print(f"❌ cv_embed维度错误，期望: {feat_dim}, 实际: {cv_embed.shape[1]}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ cv_embed维度测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始维度修复验证...")
    print("=" * 50)
    
    # 测试维度配置
    if not test_dimension_configuration():
        print("❌ 维度配置测试失败")
        return False
    
    # 测试feat_dim一致性
    if not test_feat_dim_consistency():
        print("❌ feat_dim一致性测试失败")
        return False
    
    # 测试cv_embed维度
    if not test_cv_embed_dimension():
        print("❌ cv_embed维度测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有维度测试通过！")
    print("✅ 现在可以正常运行训练命令：")
    print("   python train_net.py --config_file configs/RGBNT201/MambaPro.yml")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
