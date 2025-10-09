# 这是最重要的代码部分

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
    
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1, num_layers=2):
        """
        初始化专家网络
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
            dropout (float): Dropout比例
            num_layers (int): 网络层数
        """
        super(ExpertNetwork, self).__init__()
        
        # ========== 可配置层数的MLP专家网络：特征增强处理器 ==========
        # 🔥 功能：对单个尺度的特征进行增强处理，提升表达能力
        # 🎯 作用：特征增强 - 让每个尺度的特征变得更"聪明"
        # 📊 输入：input_dim (512维，单个尺度特征)
        # 📊 输出：output_dim (512维，增强后的尺度特征)
        # 🔧 实现：可配置层数的MLP + LayerNorm + GELU激活 + Dropout + 残差连接
        
        layers = []
        current_dim = input_dim
        
        # 构建隐藏层
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # 输出层
        layers.extend([
            nn.Linear(current_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        self.expert = nn.Sequential(*layers)
        
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
        # 🔥 专家网络处理提示（仅在第一次调用时显示）
        if not hasattr(self, '_expert_forward_called'):
            print(f"🧠 专家网络开始处理特征: {x.shape}")
            self._expert_forward_called = True
        
        # ========== MLP专家网络前向传播：特征增强处理 ==========
        # 🔥 功能：通过专家网络MLP对输入特征进行增强处理
        # 🎯 作用：特征增强 - 让每个尺度的特征变得更"聪明"
        # 📊 输入：x [B, 512] (单个尺度特征)
        # 📊 输出：output [B, 512] (增强后的尺度特征)
        expert_output = self.expert(x)  # MLP专家网络处理
        
        # 残差连接：保持原始信息，增强梯度流动
        residual = self.residual_proj(x)
        output = expert_output + residual
        
        return output


class GatingNetwork(nn.Module):
    """
    🔥 门控网络模块
    
    根据输入特征动态计算各专家的权重分布
    """
    
    def __init__(self, input_dim=1536, num_experts=3, temperature=1.0, dropout=0.1, num_layers=2):
        """
        初始化门控网络
        
        Args:
            input_dim (int): 输入特征维度（多尺度特征拼接后的维度）
            num_experts (int): 专家数量
            temperature (float): 温度参数，控制权重分布的尖锐程度
            dropout (float): Dropout比例
            num_layers (int): 网络层数
        """
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        
        # ========== 可配置层数的MLP门控网络：专家权重决策器 ==========
        # 🔥 功能：根据多尺度特征计算各专家的权重分布
        # 🎯 作用：权重计算 - 判断哪个尺度的特征更重要
        # 📊 输入：input_dim (1536维，3个尺度特征拼接)
        # 📊 输出：num_experts (3维，每个专家的权重)
        # 🔧 实现：可配置层数的MLP + LayerNorm + GELU激活 + Dropout
        
        layers = []
        current_dim = input_dim
        
        # 构建隐藏层
        for i in range(num_layers - 1):
            next_dim = current_dim // 2 if i == 0 else current_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, num_experts))
        
        self.gate = nn.Sequential(*layers)
        
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
        # 🔥 门控网络处理提示（仅在第一次调用时显示）
        if not hasattr(self, '_gate_forward_called'):
            print(f"🎯 门控网络开始计算专家权重: 输入{x.shape} → 输出[{x.shape[0]}, {self.num_experts}]")
            self._gate_forward_called = True
        
        # ========== MLP门控网络前向传播：计算专家权重 ==========
        # 🔥 功能：通过门控网络MLP计算各专家的权重分布
        # 🎯 作用：权重计算 - 判断哪个尺度的特征更重要
        # 📊 输入：x [B, 1536] (多尺度特征拼接)
        # 📊 输出：weights [B, 3] (每个专家的权重)
        gate_scores = self.gate(x)  # [B, num_experts] - 门控网络MLP处理
        
        # 应用温度参数：控制权重分布的尖锐程度
        gate_scores = gate_scores / self.temperature
        
        # Softmax归一化得到权重分布
        weights = F.softmax(gate_scores, dim=-1)  # [B, num_experts]
        
        return weights


class MultiHeadAttentionConcat(nn.Module):
    """
    🔥 多头注意力拼接模块
    
    核心功能：
    - 使用多头自注意力机制处理多尺度特征
    - 动态计算注意力权重
    - 实现更智能的特征融合
    """
    
    def __init__(self, feat_dim=512, num_heads=8, scales=[4, 8, 16], dropout=0.1):
        super(MultiHeadAttentionConcat, self).__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.scales = scales
        
        # 多头自注意力层
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(feat_dim * len(scales), feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, len(scales)),
            nn.Softmax(dim=-1)
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, multi_scale_features):
        """
        多头注意力拼接前向传播
        
        Args:
            multi_scale_features: List[Tensor] - 多尺度特征列表
        Returns:
            final_feature: [B, feat_dim] - 注意力融合后的特征
            attention_weights: [B, num_scales] - 注意力权重分布
        """
        B = multi_scale_features[0].shape[0]
        
        # 🔥 步骤1：计算注意力权重
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, feat_dim * num_scales]
        attention_weights = self.attention_weights(concat_features)  # [B, num_scales]
        
        # 🔥 步骤2：加权拼接特征
        weighted_features = []
        for i, (feat, weight) in enumerate(zip(multi_scale_features, attention_weights.unbind(-1))):
            weighted_feat = feat * weight.unsqueeze(-1)  # [B, feat_dim]
            weighted_features.append(weighted_feat)
        
        # 🔥 步骤3：多头自注意力处理
        stacked_features = torch.stack(weighted_features, dim=1)  # [B, num_scales, feat_dim]
        attn_output, attn_weights = self.multi_head_attn(
            stacked_features, stacked_features, stacked_features
        )
        
        # 🔥 步骤4：最终融合
        final_feature = torch.mean(attn_output, dim=1)  # [B, feat_dim]
        final_feature = self.final_fusion(final_feature)
        
        return final_feature, attention_weights


class MultiScaleMoE(nn.Module):
    """
    🔥 多尺度Mixture-of-Experts模块
    
    核心功能：
    - 接收多尺度特征（4x4, 8x8, 16x16）
    - 通过门控网络计算专家权重
    - 使用专家网络处理对应尺度特征
    - 加权融合得到最终特征
    - 支持多头注意力机制增强
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16], expert_hidden_dim=1024, temperature=1.0, 
                 expert_dropout=0.1, gate_dropout=0.1, expert_layers=2, gate_layers=2, 
                 expert_threshold=0.1, residual_weight=1.0, use_multi_head_attention=False,
                 attention_num_heads=8, attention_dropout=0.1):
        """
        初始化多尺度MoE模块
        
        Args:
            feat_dim (int): 特征维度
            scales (list): 滑动窗口尺度列表
            expert_hidden_dim (int): 专家网络隐藏层维度
            temperature (float): 门控网络温度参数
            expert_dropout (float): 专家网络Dropout比例
            gate_dropout (float): 门控网络Dropout比例
            expert_layers (int): 专家网络层数
            gate_layers (int): 门控网络层数
            expert_threshold (float): 专家激活阈值
            residual_weight (float): 残差连接权重
            use_multi_head_attention (bool): 是否使用多头注意力机制
            attention_num_heads (int): 注意力头数
            attention_dropout (float): 注意力Dropout比例
        """
        super(MultiScaleMoE, self).__init__()
        self.feat_dim = feat_dim
        self.scales = scales
        self.num_experts = len(scales)
        self.expert_threshold = expert_threshold
        self.residual_weight = residual_weight
        self.use_multi_head_attention = use_multi_head_attention
        
        # 🔥 多头注意力模块（可选）
        if self.use_multi_head_attention:
            self.multi_head_attention = MultiHeadAttentionConcat(
                feat_dim=feat_dim,
                num_heads=attention_num_heads,
                scales=scales,
                dropout=attention_dropout
            )
            print(f"🔥 启用多头注意力机制: {attention_num_heads}个注意力头")
        else:
            self.multi_head_attention = None
            print("🔥 使用传统MoE融合机制")
        
        # 🔥 为每个尺度创建专门的专家网络（使用配置参数）
        self.experts = nn.ModuleList()
        for i, scale in enumerate(scales):
            expert = ExpertNetwork(
                input_dim=feat_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=feat_dim,
                dropout=expert_dropout,
                num_layers=expert_layers
            )
            self.experts.append(expert)
        
        # 🔥 门控网络：根据多尺度特征计算专家权重（使用配置参数）
        gate_input_dim = feat_dim * len(scales)  # 1536维（3个尺度×512维）
        self.gating_network = GatingNetwork(
            input_dim=gate_input_dim,
            num_experts=self.num_experts,
            temperature=temperature,
            dropout=gate_dropout,
            num_layers=gate_layers
        )
        
        # ========== MLP最终融合层：专家输出融合器 ==========
        # 🔥 功能：将MoE专家网络的输出进行最终融合处理
        # 🎯 作用：特征融合 - 将专家输出融合为单一特征
        # 📊 输入：feat_dim (512维，MoE加权融合后的特征)
        # 📊 输出：feat_dim (512维，最终融合特征)
        # 🔧 实现：单层MLP + LayerNorm + GELU激活 + Dropout
        self.final_fusion = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),  # MLP层：512 -> 512 (特征增强)
            nn.LayerNorm(feat_dim),         # 层归一化：稳定训练过程
            nn.GELU(),                      # GELU激活：增加非线性表达能力
            nn.Dropout(0.1)                 # Dropout正则化：防止过拟合
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
        # 🔥 MoE模块启动提示（仅在第一次调用时显示）
        if not hasattr(self, '_moe_forward_called'):
            print(f"🚀 多尺度MoE模块启动！")
            print(f"   - 输入特征数量: {len(multi_scale_features)}")
            print(f"   - 每个特征形状: {multi_scale_features[0].shape}")
            print(f"   - 滑动窗口尺度: {self.scales}")
            print(f"   - 专家数量: {self.num_experts}")
            print(f"   - 多头注意力: {'启用' if self.use_multi_head_attention else '禁用'}")
            self._moe_forward_called = True
        
        B = multi_scale_features[0].shape[0]
        
        # 🔥 分支1：使用多头注意力机制
        if self.use_multi_head_attention and self.multi_head_attention is not None:
            # 使用多头注意力进行特征融合
            final_feature, attention_weights = self.multi_head_attention(multi_scale_features)
            return final_feature, attention_weights
        
        # 🔥 分支2：传统MoE融合机制
        # 🔥 步骤1：拼接多尺度特征作为门控网络输入
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, feat_dim * num_scales]
        
        # ========== MLP门控网络调用：计算专家权重 ==========
        # 🔥 功能：通过门控网络MLP计算各专家的权重分布
        # 🎯 作用：权重计算 - 判断哪个尺度的特征更重要
        # 📊 输入：concat_features [B, 1536] (多尺度特征拼接)
        # 📊 输出：expert_weights [B, 3] (每个专家的权重)
        expert_weights = self.gating_network(concat_features)  # [B, num_experts]
        
        # ========== MLP专家网络调用：处理各尺度特征 ==========
        # 🔥 功能：通过专家网络MLP处理各尺度的特征
        # 🎯 作用：特征增强 - 让每个尺度的特征变得更"聪明"
        # 📊 输入：multi_scale_features (List[Tensor], 每个元素[B, 512])
        # 📊 输出：expert_outputs (List[Tensor], 每个元素[B, 512])
        expert_outputs = []
        for i, (expert, feature) in enumerate(zip(self.experts, multi_scale_features)):
            expert_output = expert(feature)  # [B, feat_dim] - 专家网络MLP处理
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
        
        # ========== MLP最终融合层调用：专家输出融合 ==========
        # 🔥 功能：通过最终融合层MLP处理MoE加权融合后的特征
        # 🎯 作用：特征融合 - 将专家输出融合为单一特征
        # 📊 输入：fused_feature [B, 512] (MoE加权融合后的特征)
        # 📊 输出：final_feature [B, 512] (最终融合特征)
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
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16], expert_hidden_dim=1024, temperature=1.0,
                 expert_dropout=0.1, gate_dropout=0.1, expert_layers=2, gate_layers=2, 
                 expert_threshold=0.1, residual_weight=1.0, use_multi_head_attention=False,
                 attention_num_heads=8, attention_dropout=0.1):
        """
        初始化CLIP多尺度MoE模块
        
        Args:
            feat_dim (int): 特征维度
            scales (list): 滑动窗口尺度列表
            expert_hidden_dim (int): 专家网络隐藏层维度
            temperature (float): 门控网络温度参数
            expert_dropout (float): 专家网络Dropout比例
            gate_dropout (float): 门控网络Dropout比例
            expert_layers (int): 专家网络层数
            gate_layers (int): 门控网络层数
            expert_threshold (float): 专家激活阈值
            residual_weight (float): 残差连接权重
            use_multi_head_attention (bool): 是否使用多头注意力机制
            attention_num_heads (int): 注意力头数
            attention_dropout (float): 注意力Dropout比例
        """
        super(CLIPMultiScaleMoE, self).__init__()
        self.feat_dim = feat_dim
        self.scales = scales
        
        # 🔥 多尺度滑动窗口处理（复用现有实现）
        from .clip_multi_scale_sliding_window import CLIPMultiScaleSlidingWindow
        self.multi_scale_extractor = CLIPMultiScaleSlidingWindow(feat_dim, scales)
        
        # 🔥 MoE融合模块（使用所有配置参数）
        self.moe_fusion = MultiScaleMoE(
            feat_dim=feat_dim,
            scales=scales,
            expert_hidden_dim=expert_hidden_dim,
            temperature=temperature,
            expert_dropout=expert_dropout,
            gate_dropout=gate_dropout,
            expert_layers=expert_layers,
            gate_layers=gate_layers,
            expert_threshold=expert_threshold,
            residual_weight=residual_weight,
            use_multi_head_attention=use_multi_head_attention,
            attention_num_heads=attention_num_heads,
            attention_dropout=attention_dropout
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
        # 🔥 CLIP多尺度MoE启动提示（仅在第一次调用时显示）
        if not hasattr(self, '_clip_moe_forward_called'):
            print(f"🎯 CLIP多尺度MoE模块启动！")
            print(f"   - 输入patch tokens形状: {patch_tokens.shape}")
            print(f"   - 滑动窗口尺度: {self.scales}")
            print(f"   - 特征维度: {self.feat_dim}")
            self._clip_moe_forward_called = True
        
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
