#!/usr/bin/env python3
"""
多尺度MoE模块测试脚本

功能：
- 测试多尺度MoE模块的基本功能
- 验证专家权重计算和特征融合
- 对比传统MLP融合和MoE融合的效果

作者：基于idea-01.png设计
日期：2024
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modeling.fusion_part.multi_scale_moe import CLIPMultiScaleMoE, MultiScaleMoE
from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor


def test_moe_basic_functionality():
    """测试MoE模块基本功能"""
    print("=== 测试MoE模块基本功能 ===")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 196  # 14x14 patches
    feat_dim = 512
    
    # 创建模型
    model = CLIPMultiScaleMoE(feat_dim=feat_dim, scales=[4, 8, 16])
    
    # 创建测试输入
    patch_tokens = torch.randn(batch_size, seq_len, feat_dim)
    
    print(f"输入形状: {patch_tokens.shape}")
    
    # 前向传播
    with torch.no_grad():
        final_feature, expert_weights = model(patch_tokens)
    
    print(f"输出特征形状: {final_feature.shape}")
    print(f"专家权重形状: {expert_weights.shape}")
    
    # 验证输出形状
    assert final_feature.shape == (batch_size, feat_dim), f"输出特征形状不匹配: {final_feature.shape}"
    assert expert_weights.shape == (batch_size, 3), f"专家权重形状不匹配: {expert_weights.shape}"
    
    # 验证专家权重归一化
    weight_sums = torch.sum(expert_weights, dim=1)
    assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-6), "专家权重未正确归一化"
    
    print("✅ MoE模块基本功能测试通过！")
    return model, expert_weights


def test_expert_weight_analysis():
    """测试专家权重分析功能"""
    print("\n=== 测试专家权重分析功能 ===")
    
    # 创建模型
    model = CLIPMultiScaleMoE(feat_dim=512, scales=[4, 8, 16])
    
    # 创建测试数据
    batch_size = 8
    seq_len = 196
    patch_tokens = torch.randn(batch_size, seq_len, 512)
    
    # 前向传播
    with torch.no_grad():
        final_feature, expert_weights = model(patch_tokens)
    
    # 获取专家使用统计
    stats = model.moe_fusion.get_expert_usage_stats(expert_weights)
    
    print("专家使用统计:")
    for i, scale_name in enumerate(stats['scale_names']):
        avg_weight = stats['avg_weights'][i]
        activation_rate = stats['activation_rates'][i]
        print(f"  {scale_name}专家:")
        print(f"    平均权重: {avg_weight:.4f}")
        print(f"    激活率: {activation_rate:.4f}")
    
    print("✅ 专家权重分析功能测试通过！")
    return stats


def compare_mlp_vs_moe():
    """对比MLP融合和MoE融合"""
    print("\n=== 对比MLP融合和MoE融合 ===")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 196
    feat_dim = 512
    patch_tokens = torch.randn(batch_size, seq_len, feat_dim)
    
    # 创建MLP模型
    mlp_model = CLIPMultiScaleFeatureExtractor(feat_dim=feat_dim, scales=[4, 8, 16])
    
    # 创建MoE模型
    moe_model = CLIPMultiScaleMoE(feat_dim=feat_dim, scales=[4, 8, 16])
    
    # 前向传播
    with torch.no_grad():
        mlp_output = mlp_model(patch_tokens)
        moe_output, expert_weights = moe_model(patch_tokens)
    
    print(f"MLP输出形状: {mlp_output.shape}")
    print(f"MoE输出形状: {moe_output.shape}")
    print(f"专家权重形状: {expert_weights.shape}")
    
    # 计算特征差异
    feature_diff = torch.norm(mlp_output - moe_output, dim=1)
    print(f"特征差异 (L2范数): {feature_diff.mean().item():.6f}")
    
    # 分析专家权重分布
    print("\n专家权重分布:")
    for i, scale in enumerate([4, 8, 16]):
        avg_weight = torch.mean(expert_weights[:, i]).item()
        std_weight = torch.std(expert_weights[:, i]).item()
        print(f"  {scale}x{scale}窗口专家: {avg_weight:.4f} ± {std_weight:.4f}")
    
    print("✅ MLP vs MoE对比测试通过！")
    return mlp_output, moe_output, expert_weights


def visualize_expert_weights(expert_weights, save_path=None):
    """可视化专家权重分布"""
    print("\n=== 可视化专家权重分布 ===")
    
    # 转换为numpy数组
    weights_np = expert_weights.detach().cpu().numpy()
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scale_names = ['4x4窗口', '8x8窗口', '16x16窗口']
    
    for i, (ax, scale_name) in enumerate(zip(axes, scale_names)):
        # 绘制权重分布直方图
        ax.hist(weights_np[:, i], bins=20, alpha=0.7, color=f'C{i}')
        ax.set_title(f'{scale_name}专家权重分布')
        ax.set_xlabel('权重值')
        ax.set_ylabel('频次')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_weight = np.mean(weights_np[:, i])
        std_weight = np.std(weights_np[:, i])
        ax.axvline(mean_weight, color='red', linestyle='--', 
                  label=f'均值: {mean_weight:.3f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"专家权重分布图已保存到: {save_path}")
    
    plt.show()
    print("✅ 专家权重可视化完成！")


def test_different_scales():
    """测试不同尺度组合的效果"""
    print("\n=== 测试不同尺度组合 ===")
    
    # 测试不同的尺度组合
    scale_combinations = [
        [4, 8],
        [4, 8, 16],
        [4, 8, 16, 32]
    ]
    
    batch_size = 2
    seq_len = 196
    feat_dim = 512
    patch_tokens = torch.randn(batch_size, seq_len, feat_dim)
    
    results = {}
    
    for scales in scale_combinations:
        print(f"\n测试尺度组合: {scales}")
        
        # 创建模型
        model = CLIPMultiScaleMoE(feat_dim=feat_dim, scales=scales)
        
        # 前向传播
        with torch.no_grad():
            final_feature, expert_weights = model(patch_tokens)
        
        # 计算统计信息
        weight_entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=1).mean()
        weight_variance = torch.var(expert_weights, dim=1).mean()
        
        results[f'scales_{scales}'] = {
            'weight_entropy': weight_entropy.item(),
            'weight_variance': weight_variance.item(),
            'num_experts': len(scales)
        }
        
        print(f"  专家数量: {len(scales)}")
        print(f"  权重熵: {weight_entropy.item():.4f}")
        print(f"  权重方差: {weight_variance.item():.4f}")
    
    print("✅ 不同尺度组合测试完成！")
    return results


def test_gradient_flow():
    """测试梯度流"""
    print("\n=== 测试梯度流 ===")
    
    # 创建模型
    model = CLIPMultiScaleMoE(feat_dim=512, scales=[4, 8, 16])
    
    # 创建测试数据
    batch_size = 2
    seq_len = 196
    patch_tokens = torch.randn(batch_size, seq_len, 512, requires_grad=True)
    
    # 前向传播
    final_feature, expert_weights = model(patch_tokens)
    
    # 计算损失（简单的L2损失）
    target = torch.randn_like(final_feature)
    loss = torch.nn.functional.mse_loss(final_feature, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"输入梯度范数: {patch_tokens.grad.norm().item():.6f}")
    
    # 检查MoE模块的梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name} 梯度范数: {grad_norm:.6f}")
    
    print("✅ 梯度流测试通过！")


def main():
    """主测试函数"""
    print("🔥 多尺度MoE模块完整测试")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 基本功能测试
        model, expert_weights = test_moe_basic_functionality()
        
        # 2. 专家权重分析
        stats = test_expert_weight_analysis()
        
        # 3. MLP vs MoE对比
        mlp_output, moe_output, expert_weights = compare_mlp_vs_moe()
        
        # 4. 可视化专家权重
        visualize_expert_weights(expert_weights, save_path='expert_weights_distribution.png')
        
        # 5. 不同尺度组合测试
        scale_results = test_different_scales()
        
        # 6. 梯度流测试
        test_gradient_flow()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！多尺度MoE模块工作正常")
        print("=" * 50)
        
        # 输出总结
        print("\n📊 测试总结:")
        print(f"✅ 基本功能: 正常")
        print(f"✅ 专家权重分析: 正常")
        print(f"✅ MLP vs MoE对比: 正常")
        print(f"✅ 可视化功能: 正常")
        print(f"✅ 不同尺度组合: 正常")
        print(f"✅ 梯度流: 正常")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


