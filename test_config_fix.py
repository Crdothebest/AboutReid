#!/usr/bin/env python3
"""
配置修复验证脚本
功能：验证CLIP多尺度滑动窗口配置参数是否正确加载
"""

import sys
import os
sys.path.append('.')

from config import cfg
from config.defaults import _C

def test_config_loading():
    """测试配置加载"""
    print("🔧 测试配置加载...")
    
    try:
        # 测试默认配置
        print(f"✅ 默认配置加载成功")
        print(f"   - USE_CLIP_MULTI_SCALE: {_C.MODEL.USE_CLIP_MULTI_SCALE}")
        print(f"   - CLIP_MULTI_SCALE_SCALES: {_C.MODEL.CLIP_MULTI_SCALE_SCALES}")
        
        # 测试配置文件加载
        config_file = "configs/RGBNT201/MambaPro.yml"
        if os.path.exists(config_file):
            cfg.merge_from_file(config_file)
            print(f"✅ 配置文件加载成功: {config_file}")
            print(f"   - USE_CLIP_MULTI_SCALE: {cfg.MODEL.USE_CLIP_MULTI_SCALE}")
            print(f"   - CLIP_MULTI_SCALE_SCALES: {cfg.MODEL.CLIP_MULTI_SCALE_SCALES}")
            print(f"   - TRANSFORMER_TYPE: {cfg.MODEL.TRANSFORMER_TYPE}")
        else:
            print(f"❌ 配置文件不存在: {config_file}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_parameter_types():
    """测试参数类型"""
    print("\n🔧 测试参数类型...")
    
    try:
        # 测试USE_CLIP_MULTI_SCALE类型
        assert isinstance(cfg.MODEL.USE_CLIP_MULTI_SCALE, bool), "USE_CLIP_MULTI_SCALE应该是布尔类型"
        print("✅ USE_CLIP_MULTI_SCALE类型正确: bool")
        
        # 测试CLIP_MULTI_SCALE_SCALES类型
        assert isinstance(cfg.MODEL.CLIP_MULTI_SCALE_SCALES, list), "CLIP_MULTI_SCALE_SCALES应该是列表类型"
        print("✅ CLIP_MULTI_SCALE_SCALES类型正确: list")
        
        # 测试尺度值
        expected_scales = [4, 8, 16]
        assert cfg.MODEL.CLIP_MULTI_SCALE_SCALES == expected_scales, f"尺度值应该是{expected_scales}"
        print("✅ 尺度值正确: [4, 8, 16]")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数类型测试失败: {e}")
        return False

def test_transformer_type():
    """测试Transformer类型"""
    print("\n🔧 测试Transformer类型...")
    
    try:
        # 测试TRANSFORMER_TYPE
        assert cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16', "TRANSFORMER_TYPE应该是'ViT-B-16'"
        print("✅ TRANSFORMER_TYPE正确: 'ViT-B-16'")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer类型测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始配置修复验证...")
    print("=" * 50)
    
    # 测试配置加载
    if not test_config_loading():
        print("❌ 配置加载测试失败")
        return False
    
    # 测试参数类型
    if not test_parameter_types():
        print("❌ 参数类型测试失败")
        return False
    
    # 测试Transformer类型
    if not test_transformer_type():
        print("❌ Transformer类型测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！配置修复成功！")
    print("✅ 现在可以正常运行训练命令：")
    print("   python train_net.py --config_file configs/RGBNT201/MambaPro.yml")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
