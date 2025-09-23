"""
多尺度Mixture-of-Experts (MoE) 特征融合模块

功能：
- 在现有多尺度滑动窗口基础上，添加MoE专家网络机制
- 通过门控网络动态计算专家权重
- 实现专业化处理不同尺度特征

作者：基于idea-01.png设计
日期：2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpertNetwork(nn.Module):
    """
    🔥 专家网络模块
    
    每个专家专门处理特定尺度的特征，实现专业化分工
    """
    
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1):
        """
        初始化专家网络
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
            dropout (float): Dropout比例
        """
        super(ExpertNetwork, self).__init__()
        
        # 🔥 专家网络结构：两层MLP + 残差连接
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 残差连接的投影层（如果输入输出维度不同）
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        专家网络前向传播
        
        Args:
            x: [B, D] - 输入特征
        Returns:
            output: [B, D] - 专家处理后的特征
        """
        # 专家处理
        expert_output = self.expert(x)
        
        # 残差连接
        residual = self.residual_proj(x)
        output = expert_output + residual
        
        return output


class GatingNetwork(nn.Module):
    """
    🔥 门控网络模块
    
    根据输入特征动态计算各专家的权重分布
    """
    
    def __init__(self, input_dim=1536, num_experts=3, temperature=1.0):
        """
        初始化门控网络
        
        Args:
            input_dim (int): 输入特征维度（多尺度特征拼接后的维度）
            num_experts (int): 专家数量
            temperature (float): 温度参数，控制权重分布的尖锐程度
        """
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        
        # 🔥 门控网络：两层MLP
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_experts)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        门控网络前向传播
        
        Args:
            x: [B, input_dim] - 多尺度特征拼接
        Returns:
            weights: [B, num_experts] - 专家权重分布
        """
        # 计算门控分数
        gate_scores = self.gate(x)  # [B, num_experts]
        
        # 应用温度参数
        gate_scores = gate_scores / self.temperature
        
        # Softmax归一化得到权重分布
        weights = F.softmax(gate_scores, dim=-1)  # [B, num_experts]
        
        return weights


class MultiScaleMoE(nn.Module):
    """
    🔥 多尺度Mixture-of-Experts模块
    
    核心功能：
    - 接收多尺度特征（4x4, 8x8, 16x16）
    - 通过门控网络计算专家权重
    - 使用专家网络处理对应尺度特征
    - 加权融合得到最终特征
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16], expert_hidden_dim=1024, temperature=1.0):
        """
        初始化多尺度MoE模块
        
        Args:
            feat_dim (int): 特征维度
            scales (list): 滑动窗口尺度列表
            expert_hidden_dim (int): 专家网络隐藏层维度
            temperature (float): 门控网络温度参数
        """
        super(MultiScaleMoE, self).__init__()
        self.feat_dim = feat_dim
        self.scales = scales
        self.num_experts = len(scales)
        
        # 🔥 为每个尺度创建专门的专家网络
        self.experts = nn.ModuleList()
        for i, scale in enumerate(scales):
            expert = ExpertNetwork(
                input_dim=feat_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=feat_dim,
                dropout=0.1
            )
            self.experts.append(expert)
        
        # 🔥 门控网络：根据多尺度特征计算专家权重
        gate_input_dim = feat_dim * len(scales)  # 1536维（3个尺度×512维）
        self.gating_network = GatingNetwork(
            input_dim=gate_input_dim,
            num_experts=self.num_experts,
            temperature=temperature
        )
        
        # 🔥 最终融合层：将专家输出融合为单一特征
        self.final_fusion = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        print(f"🔥 多尺度MoE模块初始化完成:")
        print(f"   - 特征维度: {feat_dim}")
        print(f"   - 滑动窗口尺度: {scales}")
        print(f"   - 专家数量: {self.num_experts}")
        print(f"   - 门控输入维度: {gate_input_dim}")
        print(f"   - 专家隐藏层维度: {expert_hidden_dim}")
    
    def forward(self, multi_scale_features):
        """
        多尺度MoE前向传播
        
        Args:
            multi_scale_features: List[Tensor] - 多尺度特征列表
                               每个元素形状为 [B, feat_dim]
        Returns:
            final_feature: [B, feat_dim] - MoE融合后的最终特征
            expert_weights: [B, num_experts] - 专家权重分布（用于分析）
        """
        B = multi_scale_features[0].shape[0]
        
        # 🔥 步骤1：拼接多尺度特征作为门控网络输入
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, feat_dim * num_scales]
        
        # 🔥 步骤2：门控网络计算专家权重
        expert_weights = self.gating_network(concat_features)  # [B, num_experts]
        
        # 🔥 步骤3：专家网络处理对应尺度特征
        expert_outputs = []
        for i, (expert, feature) in enumerate(zip(self.experts, multi_scale_features)):
            expert_output = expert(feature)  # [B, feat_dim]
            expert_outputs.append(expert_output)
        
        # 🔥 步骤4：加权融合专家输出
        # 将专家权重广播到特征维度
        weighted_outputs = []
        for i, expert_output in enumerate(expert_outputs):
            # expert_weights[:, i] 形状为 [B]，需要扩展为 [B, feat_dim]
            weight = expert_weights[:, i:i+1].expand_as(expert_output)  # [B, feat_dim]
            weighted_output = weight * expert_output  # [B, feat_dim]
            weighted_outputs.append(weighted_output)
        
        # 求和得到融合特征
        fused_feature = torch.sum(torch.stack(weighted_outputs, dim=0), dim=0)  # [B, feat_dim]
        
        # 🔥 步骤5：最终融合层处理
        final_feature = self.final_fusion(fused_feature)  # [B, feat_dim]
        
        return final_feature, expert_weights
    
    def get_expert_usage_stats(self, expert_weights):
        """
        获取专家使用统计信息
        
        Args:
            expert_weights: [B, num_experts] - 专家权重分布
        Returns:
            stats: dict - 专家使用统计
        """
        with torch.no_grad():
            # 计算每个专家的平均权重
            avg_weights = torch.mean(expert_weights, dim=0)  # [num_experts]
            
            # 计算每个专家的激活率（权重>阈值的比例）
            threshold = 0.1
            activation_rates = torch.mean((expert_weights > threshold).float(), dim=0)  # [num_experts]
            
            stats = {
                'avg_weights': avg_weights.cpu().numpy(),
                'activation_rates': activation_rates.cpu().numpy(),
                'scale_names': [f'{scale}x{scale}' for scale in self.scales]
            }
            
            return stats


class CLIPMultiScaleMoE(nn.Module):
    """
    🔥 CLIP兼容的多尺度MoE特征提取器
    
    集成多尺度滑动窗口和MoE机制
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16], expert_hidden_dim=1024, temperature=1.0):
        """
        初始化CLIP多尺度MoE模块
        
        Args:
            feat_dim (int): 特征维度
            scales (list): 滑动窗口尺度列表
            expert_hidden_dim (int): 专家网络隐藏层维度
            temperature (float): 门控网络温度参数
        """
        super(CLIPMultiScaleMoE, self).__init__()
        self.feat_dim = feat_dim
        self.scales = scales
        
        # 🔥 多尺度滑动窗口处理（复用现有实现）
        from .clip_multi_scale_sliding_window import CLIPMultiScaleSlidingWindow
        self.multi_scale_extractor = CLIPMultiScaleSlidingWindow(feat_dim, scales)
        
        # 🔥 MoE融合模块
        self.moe_fusion = MultiScaleMoE(
            feat_dim=feat_dim,
            scales=scales,
            expert_hidden_dim=expert_hidden_dim,
            temperature=temperature
        )
        
        print(f"🔥 CLIP多尺度MoE模块初始化完成:")
        print(f"   - 特征维度: {feat_dim}")
        print(f"   - 滑动窗口尺度: {scales}")
        print(f"   - 专家隐藏层维度: {expert_hidden_dim}")
    
    def forward(self, patch_tokens):
        """
        前向传播
        
        Args:
            patch_tokens: [B, N, feat_dim] - CLIP patch tokens
        Returns:
            final_feature: [B, feat_dim] - MoE融合后的特征
            expert_weights: [B, num_experts] - 专家权重分布
        """
        # 🔥 步骤1：多尺度滑动窗口特征提取
        # 这里需要修改现有的多尺度提取器，返回各个尺度的特征而不是融合后的特征
        multi_scale_features = self._extract_multi_scale_features(patch_tokens)
        
        # 🔥 步骤2：MoE融合
        final_feature, expert_weights = self.moe_fusion(multi_scale_features)
        
        return final_feature, expert_weights
    
    def _extract_multi_scale_features(self, patch_tokens):
        """
        提取多尺度特征（修改现有实现以返回各尺度特征）
        
        Args:
            patch_tokens: [B, N, feat_dim] - CLIP patch tokens
        Returns:
            multi_scale_features: List[Tensor] - 各尺度特征列表
        """
        B, N, D = patch_tokens.shape
        
        # 转换为卷积输入格式
        x = patch_tokens.transpose(1, 2)  # [B, feat_dim, N]
        
        multi_scale_features = []
        for i, scale in enumerate(self.scales):
            if N >= scale:
                # 使用1D卷积进行滑动窗口处理
                windowed_feat = self.multi_scale_extractor.sliding_windows[i](x)  # [B, feat_dim, N//scale]
                # 全局平均池化
                pooled_feat = F.adaptive_avg_pool1d(windowed_feat, 1).squeeze(-1)  # [B, feat_dim]
            else:
                # 如果序列长度小于窗口大小，直接使用全局平均池化
                pooled_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, feat_dim]
            
            multi_scale_features.append(pooled_feat)
        
        return multi_scale_features


# 测试代码
if __name__ == "__main__":
    print("=== 多尺度MoE模块测试 ===")
    
    # 创建测试数据
    batch_size = 2
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
    print(f"专家权重分布:")
    for i, scale in enumerate([4, 8, 16]):
        avg_weight = torch.mean(expert_weights[:, i]).item()
        print(f"  {scale}x{scale}窗口专家: {avg_weight:.4f}")
    
    # 获取专家使用统计
    stats = model.moe_fusion.get_expert_usage_stats(expert_weights)
    print(f"专家激活率: {stats['activation_rates']}")
    
    print("✅ 多尺度MoE模块测试通过！")
