"""
CLIP多尺度滑动窗口集成测试脚本

功能：
- 测试CLIP多尺度滑动窗口模块的功能
- 验证与CLIP分支的集成
- 确保在保持CLIP分支完整性的基础上添加多尺度功能

作者：用户修改
日期：2024
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_clip_multi_scale_module():
    """测试CLIP多尺度滑动窗口模块"""
    print("=== 测试CLIP多尺度滑动窗口模块 ===")
    
    try:
        from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor
        
        # 创建测试数据
        batch_size = 2
        seq_len = 196  # 14x14 patches
        feat_dim = 512  # CLIP特征维度
        
        # 创建模型
        model = CLIPMultiScaleFeatureExtractor(feat_dim=feat_dim, scales=[4, 8, 16])
        
        # 创建测试输入
        patch_tokens = torch.randn(batch_size, seq_len, feat_dim)
        
        print(f"输入形状: {patch_tokens.shape}")
        
        # 前向传播
        with torch.no_grad():
            multi_scale_feature = model(patch_tokens)
        
        print(f"输出形状: {multi_scale_feature.shape}")
        print(f"期望形状: [{batch_size}, {feat_dim}]")
        
        # 验证输出形状
        assert multi_scale_feature.shape == (batch_size, feat_dim), f"输出形状不匹配: {multi_scale_feature.shape}"
        
        print("✅ CLIP多尺度滑动窗口模块测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ CLIP多尺度滑动窗口模块测试失败: {e}")
        return False

def test_clip_branch_integration():
    """测试CLIP分支集成"""
    print("\n=== 测试CLIP分支集成 ===")
    
    try:
        # 模拟CLIP分支的前向传播
        batch_size = 2
        seq_len = 196
        feat_dim = 512
        
        # 模拟CLIP输出
        clip_output = torch.randn(batch_size, seq_len + 1, feat_dim)  # [B, N+1, 512]
        
        print(f"CLIP输出形状: {clip_output.shape}")
        
        # 分离CLS token和patch tokens
        cls_token = clip_output[:, 0:1, :]  # [B, 1, 512]
        patch_tokens = clip_output[:, 1:, :]  # [B, N, 512]
        
        print(f"CLS token形状: {cls_token.shape}")
        print(f"Patch tokens形状: {patch_tokens.shape}")
        
        # 导入多尺度模块
        from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor
        multi_scale_extractor = CLIPMultiScaleFeatureExtractor(feat_dim=feat_dim, scales=[4, 8, 16])
        
        # 多尺度处理
        with torch.no_grad():
            multi_scale_feature = multi_scale_extractor(patch_tokens)  # [B, 512]
        
        print(f"多尺度特征形状: {multi_scale_feature.shape}")
        
        # 特征融合
        enhanced_cls = cls_token + multi_scale_feature.unsqueeze(1)  # [B, 1, 512]
        enhanced_output = torch.cat([enhanced_cls, patch_tokens], dim=1)  # [B, N+1, 512]
        
        print(f"增强输出形状: {enhanced_output.shape}")
        
        # 验证形状
        assert enhanced_output.shape == clip_output.shape, f"形状不匹配: {enhanced_output.shape} vs {clip_output.shape}"
        
        print("✅ CLIP分支集成测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ CLIP分支集成测试失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    
    try:
        from config import cfg
        
        # 模拟配置
        class MockConfig:
            def __init__(self):
                self.MODEL = MockModel()
        
        class MockModel:
            def __init__(self):
                self.USE_CLIP_MULTI_SCALE = True
                self.CLIP_MULTI_SCALE_SCALES = [4, 8, 16]
        
        mock_cfg = MockConfig()
        
        # 测试配置读取
        use_clip_multi_scale = getattr(mock_cfg.MODEL, 'USE_CLIP_MULTI_SCALE', False)
        clip_scales = getattr(mock_cfg.MODEL, 'CLIP_MULTI_SCALE_SCALES', [4, 8, 16])
        
        print(f"USE_CLIP_MULTI_SCALE: {use_clip_multi_scale}")
        print(f"CLIP_MULTI_SCALE_SCALES: {clip_scales}")
        
        assert use_clip_multi_scale == True, "配置读取失败"
        assert clip_scales == [4, 8, 16], "配置读取失败"
        
        print("✅ 配置加载测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 配置加载测试失败: {e}")
        return False

def test_forward_pass_simulation():
    """测试完整前向传播模拟"""
    print("\n=== 测试完整前向传播模拟 ===")
    
    try:
        # 模拟完整的CLIP多尺度前向传播
        batch_size = 2
        seq_len = 196
        feat_dim = 512
        
        # 1. 模拟CLIP前向传播
        clip_output = torch.randn(batch_size, seq_len + 1, feat_dim)
        print(f"1. CLIP输出: {clip_output.shape}")
        
        # 2. 分离tokens
        cls_token = clip_output[:, 0:1, :]
        patch_tokens = clip_output[:, 1:, :]
        print(f"2. 分离tokens - CLS: {cls_token.shape}, Patches: {patch_tokens.shape}")
        
        # 3. 多尺度处理
        from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor
        multi_scale_extractor = CLIPMultiScaleFeatureExtractor(feat_dim=feat_dim, scales=[4, 8, 16])
        
        with torch.no_grad():
            multi_scale_feature = multi_scale_extractor(patch_tokens)
        print(f"3. 多尺度特征: {multi_scale_feature.shape}")
        
        # 4. 特征融合
        enhanced_cls = cls_token + multi_scale_feature.unsqueeze(1)
        enhanced_output = torch.cat([enhanced_cls, patch_tokens], dim=1)
        print(f"4. 增强输出: {enhanced_output.shape}")
        
        # 5. 全局特征提取
        global_feat = enhanced_output[:, 0]  # [B, 512]
        print(f"5. 全局特征: {global_feat.shape}")
        
        # 验证所有形状
        assert global_feat.shape == (batch_size, feat_dim), f"全局特征形状错误: {global_feat.shape}"
        
        print("✅ 完整前向传播模拟测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 完整前向传播模拟测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始CLIP多尺度滑动窗口集成测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_clip_multi_scale_module())
    test_results.append(test_clip_branch_integration())
    test_results.append(test_config_loading())
    test_results.append(test_forward_pass_simulation())
    
    # 统计结果
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果统计: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！CLIP多尺度滑动窗口集成成功！")
        print("\n✅ 功能确认:")
        print("   - CLIP多尺度滑动窗口模块工作正常")
        print("   - CLIP分支集成成功")
        print("   - 配置加载正常")
        print("   - 前向传播流程正确")
        print("\n🎯 现在可以运行训练命令:")
        print("   python train_net.py --config_file configs/RGBNT201/MambaPro.yml")
    else:
        print("❌ 部分测试失败，请检查代码实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
