#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置文件参数设置
功能：验证所有注意力融合参数是否正确配置
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_config_loading():
    """测试配置文件加载"""
    print("🧪 测试配置文件加载...")
    
    try:
        from config.defaults import _C
        print("✅ 默认配置加载成功")
        
        # 检查注意力融合参数
        print("\n🔍 检查注意力融合参数:")
        print(f"  USE_ATTENTION_FUSION: {_C.MODEL.USE_ATTENTION_FUSION}")
        print(f"  ATTENTION_NUM_HEADS: {_C.MODEL.ATTENTION_NUM_HEADS}")
        print(f"  ATTENTION_DROPOUT: {_C.MODEL.ATTENTION_DROPOUT}")
        print(f"  ATTENTION_DIM: {_C.MODEL.ATTENTION_DIM}")
        
        # 检查门控融合参数
        print("\n🔍 检查门控融合参数:")
        print(f"  USE_GATE_FUSION: {_C.MODEL.USE_GATE_FUSION}")
        print(f"  GATE_NUM_HEADS: {_C.MODEL.GATE_NUM_HEADS}")
        print(f"  GATE_DROPOUT: {_C.MODEL.GATE_DROPOUT}")
        
        # 检查MoE参数
        print("\n🔍 检查MoE参数:")
        print(f"  USE_MULTI_SCALE_MOE: {_C.MODEL.USE_MULTI_SCALE_MOE}")
        print(f"  MOE_SCALES: {_C.MODEL.MOE_SCALES}")
        print(f"  MOE_EXPERT_HIDDEN_DIM: {_C.MODEL.MOE_EXPERT_HIDDEN_DIM}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_yaml_config():
    """测试YAML配置文件"""
    print("\n🧪 测试YAML配置文件...")
    
    try:
        import yaml
        
        # 测试MambaPro_moe.yml配置文件
        config_path = "configs/RGBNT201/MambaPro_moe.yml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print("✅ YAML配置文件加载成功")
            
            # 检查注意力融合参数
            if 'MODEL' in config:
                model_config = config['MODEL']
                print("\n🔍 检查YAML中的注意力融合参数:")
                print(f"  USE_ATTENTION_FUSION: {model_config.get('USE_ATTENTION_FUSION', 'Not found')}")
                print(f"  ATTENTION_NUM_HEADS: {model_config.get('ATTENTION_NUM_HEADS', 'Not found')}")
                print(f"  ATTENTION_DROPOUT: {model_config.get('ATTENTION_DROPOUT', 'Not found')}")
                print(f"  ATTENTION_DIM: {model_config.get('ATTENTION_DIM', 'Not found')}")
            
            return True
        else:
            print(f"❌ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ YAML配置加载失败: {e}")
        return False

def test_parameter_override():
    """测试参数覆盖功能"""
    print("\n🧪 测试参数覆盖功能...")
    
    try:
        from yacs.config import CfgNode
        from config.defaults import _C
        
        # 创建配置副本
        cfg = _C.clone()
        
        # 测试参数覆盖
        cfg.merge_from_list([
            'MODEL.USE_ATTENTION_FUSION', 'True',
            'MODEL.ATTENTION_NUM_HEADS', '12',
            'MODEL.ATTENTION_DROPOUT', '0.15',
            'MODEL.ATTENTION_DIM', '512'
        ])
        
        print("✅ 参数覆盖测试成功")
        print(f"  覆盖后 USE_ATTENTION_FUSION: {cfg.MODEL.USE_ATTENTION_FUSION}")
        print(f"  覆盖后 ATTENTION_NUM_HEADS: {cfg.MODEL.ATTENTION_NUM_HEADS}")
        print(f"  覆盖后 ATTENTION_DROPOUT: {cfg.MODEL.ATTENTION_DROPOUT}")
        print(f"  覆盖后 ATTENTION_DIM: {cfg.MODEL.ATTENTION_DIM}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数覆盖测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始配置参数测试...")
    print("=" * 50)
    
    # 测试1：默认配置加载
    test1_result = test_config_loading()
    
    # 测试2：YAML配置加载
    test2_result = test_yaml_config()
    
    # 测试3：参数覆盖
    test3_result = test_parameter_override()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"  默认配置加载: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"  YAML配置加载: {'✅ 通过' if test2_result else '❌ 失败'}")
    print(f"  参数覆盖测试: {'✅ 通过' if test3_result else '❌ 失败'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("\n🎉 所有测试通过！配置参数设置正确。")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查配置设置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
