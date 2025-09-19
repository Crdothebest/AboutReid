#!/usr/bin/env python3
"""
快速测试脚本
用于快速验证多尺度MoE模块的功能

作者修改：创建快速测试脚本，验证模块功能而不进行完整训练
功能：快速测试模型加载、前向传播等功能
撤销方法：删除此文件
"""

import torch
import sys
import os
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append('/home/zubuntu/workspace/yzy/MambaPro')

def test_model_loading():
    """
    测试模型加载功能
    
    作者修改：验证多尺度MoE模块是否能正确加载
    功能：测试模型初始化和配置加载
    撤销方法：删除此函数
    """
    print("🔧 测试模型加载功能...")
    
    try:
        from modeling.make_model import make_model
        from config import cfg
        
        # 测试基线配置
        print("📋 测试基线配置...")
        cfg.merge_from_file("configs/RGBNT201/MambaPro_baseline.yml")
        cfg.freeze()
        
        # 创建基线模型
        baseline_model = make_model(cfg, num_class=201, camera_num=6, view_num=2)
        print("✅ 基线模型加载成功")
        
        # 测试MoE配置
        print("📋 测试MoE配置...")
        cfg.merge_from_file("configs/RGBNT201/MambaPro_moe.yml")
        cfg.freeze()
        
        # 创建MoE模型
        moe_model = make_model(cfg, num_class=201, camera_num=6, view_num=2)
        print("✅ MoE模型加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """
    测试前向传播功能
    
    作者修改：验证多尺度MoE模块的前向传播是否正常
    功能：测试模型推理过程
    撤销方法：删除此函数
    """
    print("🔄 测试前向传播功能...")
    
    try:
        from modeling.make_model import make_model
        from config import cfg
        
        # 测试基线模型前向传播
        print("📋 测试基线模型前向传播...")
        cfg.merge_from_file("configs/RGBNT201/MambaPro_baseline.yml")
        cfg.freeze()
        
        baseline_model = make_model(cfg, num_class=201, camera_num=6, view_num=2)
        baseline_model.eval()
        
        # 创建测试数据 - 修复输入格式
        batch_size = 2
        height, width = 256, 128  # 图像尺寸
        channels = 3              # RGB图像通道数
        
        test_data = {
            'RGB': torch.randn(batch_size, channels, height, width),  # [B, 3, H, W]
            'NI': torch.randn(batch_size, channels, height, width),   # [B, 3, H, W]
            'TI': torch.randn(batch_size, channels, height, width)    # [B, 3, H, W]
        }
        
        with torch.no_grad():
            baseline_output = baseline_model(test_data)
            print(f"✅ 基线模型前向传播成功，输出形状: {baseline_output.shape}")
        
        # 测试MoE模型前向传播
        print("📋 测试MoE模型前向传播...")
        cfg.merge_from_file("configs/RGBNT201/MambaPro_moe.yml")
        cfg.freeze()
        
        moe_model = make_model(cfg, num_class=201, camera_num=6, view_num=2)
        moe_model.eval()
        
        with torch.no_grad():
            moe_output = moe_model(test_data)
            print(f"✅ MoE模型前向传播成功，输出形状: {moe_output.shape}")
        
        # 比较输出形状
        if baseline_output.shape == moe_output.shape:
            print("✅ 两个模型输出形状一致")
        else:
            print(f"⚠️  输出形状不一致: 基线 {baseline_output.shape} vs MoE {moe_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_scale_moe_module():
    """
    测试多尺度MoE模块
    
    作者修改：单独测试多尺度MoE模块的功能
    功能：验证MoE模块的各个组件是否正常工作
    撤销方法：删除此函数
    """
    print("🧪 测试多尺度MoE模块...")
    
    try:
        from modeling.fusion_part.multi_scale_moe import MultiScaleMoE, MultiScaleMoEAAM
        
        # 测试参数
        batch_size = 2
        seq_len = 197
        dim = 512
        
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, dim)
        print(f"📊 输入数据形状: {x.shape}")
        
        # 测试MultiScaleMoE模块
        print("🔧 测试MultiScaleMoE模块...")
        moe_module = MultiScaleMoE(dim=dim, scales=[4, 8, 16])
        moe_output = moe_module(x)
        print(f"✅ MultiScaleMoE输出形状: {moe_output.shape}")
        
        # 测试MultiScaleMoEAAM模块
        print("🔧 测试MultiScaleMoEAAM模块...")
        
        # 模拟配置 - 修复配置对象
        class MockConfig:
            MODEL = type('obj', (object,), {
                'MAMBA_BI': False,
                'TRANSFORMER_TYPE': 'ViT-B-16'
            })()
            DATASETS = type('obj', (object,), {
                'NAMES': 'RGBNT201'
            })()
        
        cfg = MockConfig()
        aam_module = MultiScaleMoEAAM(dim=dim, n_layers=2, cfg=cfg)
        
        # 创建三种模态的测试数据
        r = torch.randn(batch_size, seq_len, dim)
        n = torch.randn(batch_size, seq_len, dim)
        t = torch.randn(batch_size, seq_len, dim)
        
        aam_output = aam_module(r, n, t)
        print(f"✅ MultiScaleMoEAAM输出形状: {aam_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 多尺度MoE模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数：执行所有测试
    
    作者修改：创建测试主流程
    功能：按顺序执行所有测试项目
    撤销方法：删除此函数
    """
    parser = argparse.ArgumentParser(description="快速测试多尺度MoE模块")
    parser.add_argument("--test", choices=["all", "loading", "forward", "moe"], 
                       default="all", help="选择测试项目")
    
    args = parser.parse_args()
    
    print("🧪 多尺度MoE模块快速测试")
    print("=" * 50)
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试项目: {args.test}")
    
    test_results = {}
    
    if args.test in ["all", "loading"]:
        test_results["model_loading"] = test_model_loading()
    
    if args.test in ["all", "forward"]:
        test_results["forward_pass"] = test_forward_pass()
    
    if args.test in ["all", "moe"]:
        test_results["moe_module"] = test_multi_scale_moe_module()
    
    # 打印测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！多尺度MoE模块功能正常。")
        print("💡 建议：可以开始进行完整训练实验。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        print("💡 建议：修复问题后再进行训练实验。")

if __name__ == "__main__":
    main()
