"""
测试多尺度MoE模块的功能
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append('/home/zubuntu/workspace/yzy/MambaPro')

def test_multi_scale_moe():
    """测试多尺度MoE模块"""
    print("🧪 开始测试多尺度MoE模块...")
    
    try:
        from modeling.fusion_part.multi_scale_moe import MultiScaleMoE, MultiScaleMoEAAM
        
        # 测试参数
        batch_size = 2
        seq_len = 197  # ViT的patch数量 (14*14 + 1)
        dim = 512      # CLIP ViT-B/16的维度
        
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, dim)
        print(f"✅ 输入数据形状: {x.shape}")
        
        # 测试MultiScaleMoE模块
        moe_module = MultiScaleMoE(dim=dim, scales=[4, 8, 16])
        output = moe_module(x)
        print(f"✅ MultiScaleMoE输出形状: {output.shape}")
        
        # 测试MultiScaleMoEAAM模块
        # 模拟配置
        class MockConfig:
            MODEL = type('obj', (object,), {
                'MAMBA_BI': False,
                'TRANSFORMER_TYPE': 'ViT-B-16'
            })()
        
        cfg = MockConfig()
        aam_module = MultiScaleMoEAAM(dim=dim, n_layers=2, cfg=cfg)
        
        # 创建三种模态的测试数据
        r = torch.randn(batch_size, seq_len, dim)
        n = torch.randn(batch_size, seq_len, dim)
        t = torch.randn(batch_size, seq_len, dim)
        
        output = aam_module(r, n, t)
        print(f"✅ MultiScaleMoEAAM输出形状: {output.shape}")
        
        print("🎉 所有测试通过！多尺度MoE模块工作正常。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_scale_moe()
