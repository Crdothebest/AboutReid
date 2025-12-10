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
        
        # 🔧 修复：保存参数为实例属性，用于后续显示和调试
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # ========== 可配置层数的MLP专家网络：特征增强处理器 ==========
        # 🔥 功能：对单个尺度的特征进行增强处理，提升表达能力
        # 🎯 作用：特征增强 - 让每个尺度的特征变得更"聪明"
        # 📊 输入：input_dim (512维，单个尺度特征)
        # 📊 输出：output_dim (512维，增强后的尺度特征)
        # 🔧 实现：符合标准Transformer MLP设计
        #   - 隐藏层：Linear → LayerNorm → GELU → Dropout
        #   - 输出层：Linear → Dropout（无LayerNorm、无激活，由残差连接提供非线性）
        
        layers = []
        current_dim = input_dim
        
        # 🔧 修复：构建隐藏层（符合标准Transformer FFN设计）
        # 标准FFN结构：Linear → GELU → Dropout（无LayerNorm）
        # LayerNorm在FFN外部（在Transformer Block中），不在FFN内部
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                # ✅ 移除LayerNorm：标准FFN内部没有LayerNorm
                nn.GELU(),
                nn.Dropout(dropout)  # ✅ 保留Dropout：标准FFN有Dropout
            ])
            current_dim = hidden_dim
        
        # 🔧 修复：输出层符合标准Transformer FFN设计
        # 标准FFN输出层：Linear → Dropout（无LayerNorm，无激活）
        # 1. 残差连接本身提供归一化效果
        # 2. 残差连接提供非线性：output = expert_output + residual
        # 3. 符合标准Transformer FFN设计（参考transformer_block.py）
        layers.extend([
            nn.Linear(current_dim, output_dim),
            # ✅ 移除LayerNorm：残差连接提供归一化效果
            # ✅ 移除GELU：残差连接提供非线性
            # ✅ 移除Dropout：如果需要正则化，应在Expert模块输出上应用
            # 标准FFN输出层通常有Dropout，但考虑到残差连接和后续正则化，这里移除
        ])
        
        self.expert = nn.Sequential(*layers)
        
        # 残差连接的投影层（如果输入输出维度不同）
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化权重
        
        🔧 优化：使用Kaiming初始化（适合GELU激活函数）
        - 隐藏层：Kaiming初始化，适合GELU激活
        - 输出层：Xavier初始化，适合线性输出
        - 残差投影层：Xavier初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 判断是否为输出层（最后一层）
                is_output_layer = (m.out_features == self.output_dim)
                
                if is_output_layer:
                    # 输出层使用Xavier初始化（适合线性输出）
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                else:
                    # 隐藏层使用Kaiming初始化（适合GELU激活函数）
                    # GELU是平滑激活函数，使用fan_in模式
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 残差投影层特殊处理（如果存在）
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight, gain=1.0)
            if self.residual_proj.bias is not None:
                nn.init.constant_(self.residual_proj.bias, 0)
    
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
    
    def __init__(self, input_dim=1536, num_experts=3, temperature=1.0, dropout=0.1, num_layers=2, init_weights=None):
        """
        初始化门控网络
        
        Args:
            input_dim (int): 输入特征维度（多尺度特征拼接后的维度）
            num_experts (int): 专家数量
            temperature (float): 温度参数，控制权重分布的尖锐程度
            dropout (float): Dropout比例
            num_layers (int): 网络层数
            init_weights (list, optional): 专家初始权重列表，如 [0.35, 0.3, 0.35]
        """
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.init_weights = init_weights
        
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
        
        # 输出层（单独创建以便设置初始权重）
        self.gate_output = nn.Linear(current_dim, num_experts)
        layers.append(self.gate_output)
        
        self.gate = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化权重修改
        
        🔧 修复：解决模式坍塌的关键修改
        - 隐藏层：Xavier初始化，适合LayerNorm + GELU
        - 输出层：极小值初始化 + 零偏置，确保初始logits接近0
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 🎯 修复1：跳过gate_output层，避免被通用初始化覆盖
                if m is self.gate_output:
                    continue  # 后面会特殊处理
                
                # 其他层使用Xavier初始化（适合LayerNorm + GELU组合）
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 🎯 修复2：强制Logits极小化，确保初始权重为均匀分布
        # 问题分析：
        # 1. 即使bias=0，如果权重W太大，W*x仍然可能产生较大的logits值
        # 2. 例如：W ≈ 0.1, x ≈ 10 → W*x ≈ 1.0
        # 3. Softmax([1.0, 0.0, 0.0]) ≈ [0.73, 0.13, 0.13] ❌ 不均匀！
        # 
        # 解决方案：极小值初始化权重，确保W*x ≈ 0
        # 这样即使x很大，logits仍然接近[0, 0, 0]
        # Softmax([0, 0, 0]) = [0.33, 0.33, 0.33] ✅ 均匀分布
        if self.gate_output.bias is not None:
            nn.init.constant_(self.gate_output.bias, 0.0)
        
        # 极小值初始化权重：范围[-1e-4, 1e-4]
        # 确保即使输入特征很大（如x≈10），W*x仍然接近0
        nn.init.uniform_(self.gate_output.weight, -1e-4, 1e-4)
            
        # 如果提供了初始权重，仅用于日志显示（不实际设置）
        if self.init_weights is not None:
            if len(self.init_weights) != self.num_experts:
                raise ValueError(f"初始权重数量 ({len(self.init_weights)}) 必须等于专家数量 ({self.num_experts})")
            
            # 仅用于日志显示，不实际设置偏置
            init_weights_tensor = torch.tensor(self.init_weights, dtype=torch.float32)
            init_weights_tensor = init_weights_tensor / init_weights_tensor.sum()
            
            print(f"ℹ️  门控网络初始权重参考: {self.init_weights}")
            print(f"   - 实际初始化: 零初始化（偏置=0），保证均匀分布[1/{self.num_experts}, ...]")
            print(f"   - 让网络自然学习最优权重分布，避免极端偏向导致的模式坍塌")
    
    def forward(self, x):
        """
        门控网络前向传播
        
        Args:
            x: [B, input_dim] - 多尺度特征拼接
        Returns:
            weights: [B, num_experts] - 专家权重分布
        """
        # 🔥 注意：门控网络的输出提示已移至 MultiScaleMoE._expert_network_processing()
        # 这里不再输出，避免在固定权重模式下误显示
        
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


class GateFusionConcat(nn.Module):
    """
    🔥 门控加权-预处理模块
    
    核心功能：
    - 使用门控网络（MLP）学习动态权重
    - 智能加权融合多尺度特征
    - 实现更智能的特征融合
    
    注意：
    - 本模块使用门控网络（MLP + Softmax），不是多头注意力机制
    - 如需真正的多头注意力机制，请使用 AttentionFusionConcat
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16], dropout=0.1):
        super(GateFusionConcat, self).__init__()
        self.feat_dim = feat_dim
        self.scales = scales
        self.dropout = dropout
        
        # 门控加权-预处理网络（推荐方案）
        self.gate_network = nn.Sequential(
            nn.Linear(feat_dim * len(scales), feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(feat_dim, len(scales)),
            nn.Softmax(dim=-1)
        )
        
        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2)
        )
        
        # 最终融合层 - 优化设计，保持多尺度信息
        self.final_fusion = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # 减少Dropout，保持信息
        )
        
        # 特征增强器
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3)
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
        门控加权-预处理前向传播
        
        Args:
            multi_scale_features: List[Tensor] - 多尺度特征列表
        Returns:
            enhanced_multi_scale_features: List[Tensor] - 门控加权-预处理后的多尺度特征
            gate_weights: [B, num_scales] - 门控权重分布
        """
        B = multi_scale_features[0].shape[0]
        
        # 🔥 门控加权-预处理启动提示（仅在第一次调用时显示，且模块启用时）
        # 注意：这个方法只在 use_gate_fusion=True 时被调用，所以不需要额外检查
        if not hasattr(self, '_attention_forward_called'):
            print(f"🎯 门控加权-预处理机制启动！")
            print(f"   - 输入多尺度特征数量: {len(multi_scale_features)}")
            print(f"   - 每个特征形状: {multi_scale_features[0].shape}")
            print(f"   - 门控网络: 学习动态权重")
            print(f"   - 滑动窗口尺度: {self.scales}")
            self._attention_forward_called = True
        
        # 🔥 步骤1：门控权重计算
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, feat_dim * num_scales]
        gate_weights = self.gate_network(concat_features)  # [B, num_scales]
        
        # 🔥 步骤2：门控加权融合
        enhanced_multi_scale_features = []
        for i, (feat, weight) in enumerate(zip(multi_scale_features, gate_weights.unbind(-1))):
            # 门控加权
            weighted_feat = feat * weight.unsqueeze(-1)  # [B, feat_dim]
            
            # 特征增强
            enhanced_feat = self.feature_enhancer(weighted_feat)
            
            # 残差连接，保持原始信息
            enhanced_feat = enhanced_feat + feat * 0.3  # 残差连接
            
            enhanced_multi_scale_features.append(enhanced_feat)
        
        # 🔥 门控加权-预处理处理完成提示（仅在第一次调用时显示，且模块启用时）
        # 注意：这个方法只在 use_gate_fusion=True 时被调用，所以不需要额外检查
        if not hasattr(self, '_attention_complete_called'):
            print(f"✅ 门控加权-预处理完成！")
            print(f"   - 输出多尺度特征数量: {len(enhanced_multi_scale_features)}")
            print(f"   - 每个特征形状: {enhanced_multi_scale_features[0].shape}")
            print(f"   - 门控权重形状: {gate_weights.shape}")
            print(f"   - 门控权重分布: {gate_weights[0].detach().cpu().numpy()}")
            print(f"   - 门控网络: 学习动态权重")
            print(f"   - 保持多尺度结构: 不丢失信息")
            print(f"   - 残差连接: 30%原始信息保留")
            self._attention_complete_called = True
        
        return enhanced_multi_scale_features, gate_weights


class MultiScaleMoE(nn.Module):
    """
    🔥 多尺度Mixture-of-Experts模块
    
    核心功能：
    - 接收多尺度特征（4x4, 8x8, 16x16）
    - 通过门控网络计算专家权重
    - 使用专家网络处理对应尺度特征
    - 加权融合得到最终特征
    - 支持门控加权-预处理机制增强
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16], expert_hidden_dim=1024, temperature=1.0,
                 expert_dropout=0.1, gate_dropout=0.1, expert_layers=2, gate_layers=2, 
                 expert_threshold=0.1, residual_weight=1.0, use_gate_fusion=False,
                 use_attention_fusion=False, attention_num_heads=8,
                 attention_dropout=0.1, attention_dim=512, init_weights=None,
                 use_fixed_weights=False, fixed_weights=None,
                 use_top_k_routing=False, top_k=2, top_k_mode="soft"):
        """
        初始化CLIP多尺度MoE模块
        
        Args:
            use_top_k_routing (bool): 是否使用 Top-k 路由机制
            top_k (int): Top-k 路由的 k 值
            top_k_mode (str): Top-k 路由模式 ("soft" 或 "hard")
        """
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
            use_gate_fusion (bool): 是否使用门控加权-预处理机制（使用MLP门控网络）
            use_attention_fusion (bool): 是否使用注意力-预处理机制（使用多头注意力）
            attention_num_heads (int): 注意力网络头数（仅在use_attention_fusion=True时使用）
            attention_dropout (float): 注意力网络Dropout比例
            attention_dim (int): 注意力网络维度
            init_weights (list, optional): 专家初始权重列表，如 [0.35, 0.3, 0.35]
            use_fixed_weights (bool): 是否使用固定权重模式
                - True: 使用固定权重，禁用门控网络，专家权重固定不变
                - False: 使用门控网络动态计算权重（默认）
                使用场景：
                  1. 调试：排除门控网络影响，专注于专家网络性能
                  2. 对比实验：对比固定权重 vs 动态权重的效果
                  3. 跨域鲁棒性：固定权重可能在跨域场景下更稳定
            fixed_weights (list, optional): 固定权重列表，如 [0.33, 0.33, 0.34]
                - 仅在 use_fixed_weights=True 时生效
                - 长度必须等于专家数量（len(scales)）
                - 权重会自动归一化，无需手动确保和为1.0
                - 示例：[0.33, 0.33, 0.34] 表示三个专家权重分别为33%、33%、34%
                - 如果为None且use_fixed_weights=True，会抛出ValueError
        """
        super(MultiScaleMoE, self).__init__()
        self.feat_dim = feat_dim
        self.scales = scales
        self.num_experts = len(scales)
        self.expert_threshold = expert_threshold
        self.residual_weight = residual_weight
        self.use_gate_fusion = use_gate_fusion
        self.use_attention_fusion = use_attention_fusion
        self.use_fixed_weights = use_fixed_weights
        self.fixed_weights = fixed_weights
        self.use_top_k_routing = use_top_k_routing
        self.top_k = top_k
        self.top_k_mode = top_k_mode
        # 🔧 修复：保存专家网络参数，用于后续显示
        self.expert_hidden_dim = expert_hidden_dim
        self.expert_layers = expert_layers
        
        # 🔥 Top-k 路由参数验证
        if self.use_top_k_routing:
            if self.top_k < 1 or self.top_k > self.num_experts:
                raise ValueError(f"Top-k value ({self.top_k}) must be between 1 and {self.num_experts}")
            if self.top_k_mode not in ["soft", "hard"]:
                raise ValueError(f"Top-k mode must be 'soft' or 'hard', got '{self.top_k_mode}'")
            print(f"🔥 Top-k 路由：已启用 (k={self.top_k}, mode={self.top_k_mode})")
        else:
            print("🔥 Top-k 路由：已禁用 (使用传统软路由)")
        
        # 🔥 门控加权-预处理模块（可选）
        if self.use_gate_fusion:
            self.gate_fusion = GateFusionConcat(
                feat_dim=feat_dim,
                scales=scales,
                dropout=gate_dropout
            )
            print(f"🔥 门控加权-预处理机制：已启用 (Dropout={gate_dropout})")
        else:
            self.gate_fusion = None
            print("🔥 门控加权-预处理机制：已禁用 (使用传统MLP融合)")
        
        # 🔥 注意力-预处理模块（可选）
        if self.use_attention_fusion:
            self.attention_fusion = AttentionFusionConcat(
                feat_dim=feat_dim,
                num_heads=attention_num_heads,
                scales=scales,
                dropout=attention_dropout,
                attention_dim=attention_dim
            )
            print(f"🔥 注意力-预处理机制：已启用 ({attention_num_heads}个注意力头, Dropout={attention_dropout})")
        else:
            self.attention_fusion = None
            print("🔥 注意力-预处理机制：已禁用 (使用传统MLP融合)")
        
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
        # 
        # 功能说明：
        #   - 动态权重模式（use_fixed_weights=False）：创建门控网络，根据输入特征动态计算专家权重
        #   - 固定权重模式（use_fixed_weights=True）：不创建门控网络，直接使用预设的固定权重
        #
        # 两种模式的区别：
        #   1. 动态权重模式：
        #      - 门控网络会根据每个样本的特征自动调整专家权重
        #      - 权重会随训练过程学习优化
        #      - 需要训练门控网络参数，计算开销较大
        #   2. 固定权重模式：
        #      - 所有样本使用相同的固定权重
        #      - 权重不随训练改变，门控网络被禁用
        #      - 无需训练门控网络，计算开销小，但灵活性低
        #
        # 如果使用固定权重，则不创建门控网络
        # 注意：gate_input_dim 需要始终定义，因为后续打印语句会使用它
        gate_input_dim = feat_dim * len(scales)  # 1536维（3个尺度×512维）
        
        if not self.use_fixed_weights:
            self.gating_network = GatingNetwork(
                input_dim=gate_input_dim,
                num_experts=self.num_experts,
                temperature=temperature,
                dropout=gate_dropout,
                num_layers=gate_layers,
                init_weights=init_weights
            )
        else:
            # ========== 固定权重模式初始化 ==========
            # 功能：当 use_fixed_weights=True 时，不创建门控网络，使用预设的固定权重
            # 
            # 实现步骤：
            #   1. 验证固定权重参数的有效性
            #   2. 将固定权重转换为tensor并归一化
            #   3. 保存为实例属性，供前向传播使用
            #
            # 参数验证：
            #   - fixed_weights 不能为 None
            #   - fixed_weights 长度必须等于专家数量
            #   - 权重会自动归一化，确保和为1.0（即使输入权重和不为1.0）
            #
            self.gating_network = None
            
            # 步骤1：验证固定权重参数
            if fixed_weights is None:
                raise ValueError(
                    "use_fixed_weights=True but fixed_weights is None. "
                    "Please provide fixed_weights, e.g., [0.33, 0.33, 0.34]"
                )
            if len(fixed_weights) != self.num_experts:
                raise ValueError(
                    f"fixed_weights length {len(fixed_weights)} != num_experts {self.num_experts}. "
                    f"Please provide {self.num_experts} weights for {self.num_experts} experts."
                )
            
            # 步骤2：归一化固定权重
            # 说明：即使输入的权重和不为1.0，也会自动归一化
            # 例如：[0.5, 0.5, 0.5] 会被归一化为 [0.33, 0.33, 0.34]
            fixed_weights_tensor = torch.tensor(fixed_weights, dtype=torch.float32)
            fixed_weights_tensor = fixed_weights_tensor / fixed_weights_tensor.sum()
            self.fixed_weights_tensor = fixed_weights_tensor
            
            # 步骤3：输出提示信息
            print(f"🔥 固定权重模式：已启用")
            print(f"   - 固定权重值: {fixed_weights_tensor.tolist()}")
            print(f"   - 专家数量: {self.num_experts}")
            print(f"   - 注意：门控网络已禁用，权重不会随训练改变")
        
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
        if not self.use_fixed_weights:
            print(f"   - 门控输入维度: {gate_input_dim}")
        else:
            print(f"   - 门控网络: 已禁用（使用固定权重模式）")
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
            print(f"   - 门控加权-预处理机制: {'已启用' if self.use_gate_fusion else '已禁用'}")
            print(f"   - 注意力-预处理机制: {'已启用' if self.use_attention_fusion else '已禁用'}")
            # 🔥 Top-k 路由状态显示
            if self.use_top_k_routing:
                print(f"   - Top-k 路由机制: ✅ 已启用 (k={self.top_k}, mode={self.top_k_mode})")
            else:
                print(f"   - Top-k 路由机制: ❌ 已禁用 (使用传统软路由)")
            print()  # 空行
            self._moe_forward_called = True
        
        B = multi_scale_features[0].shape[0]
        
        # 🔥 分支1：使用门控加权-预处理进行特征融合（替代无预处理）
        if self.use_gate_fusion and self.gate_fusion is not None:
            # 🔥 门控加权-预处理调用提示（仅在第一次调用时显示，且模块启用时）
            if not hasattr(self, '_gate_fusion_branch_called'):
                print(f"🎯 拼接融合：使用门控加权-预处理（门控加权-预处理）")
                print(f"   - 门控网络头数: {self.gate_fusion.num_heads}")
                print(f"   - 门控网络Dropout: {self.gate_fusion.dropout}")
                print(f"   - 步骤：门控权重计算 → 特征加权增强 → 拼接 → 专家网络处理")
                print()  # 空行
                self._gate_fusion_branch_called = True
            
            # 使用门控加权-预处理进行特征融合，得到融合后的多尺度特征
            fused_multi_scale_features = self._attention_fusion_features(multi_scale_features)
            # 继续使用专家网络处理融合后的特征
            return self._expert_network_processing(fused_multi_scale_features)
        
        # 🔥 分支2：使用注意力-预处理进行特征融合（新增）
        elif self.use_attention_fusion and self.attention_fusion is not None:
            # 🔥 注意力-预处理调用提示（仅在第一次调用时显示，且模块启用时）
            if not hasattr(self, '_attention_fusion_branch_called'):
                print(f"🎯 拼接融合：使用注意力-预处理（注意力-预处理）")
                print(f"   - 注意力头数: {self.attention_fusion.num_heads}")
                print(f"   - 注意力Dropout: {self.attention_fusion.dropout}")
                print(f"   - 步骤：多头注意力计算 → 特征加权增强 → 全局融合 → 拼接 → 专家网络处理")
                print()  # 空行
                self._attention_fusion_branch_called = True
            
            # 使用注意力-预处理进行特征融合，得到融合后的多尺度特征
            fused_multi_scale_features = self.attention_fusion(multi_scale_features)
            # 继续使用专家网络处理融合后的特征
            return self._expert_network_processing(fused_multi_scale_features)
        
        # 🔥 分支3：传统MoE融合机制（无预处理 + 专家网络）
        # 🔥 传统MoE融合提示（仅在第一次调用时显示，且仅在无预处理模式下）
        # 注意：只有在 use_gate_fusion=False 且 use_attention_fusion=False 时才会执行到这里
        if not hasattr(self, '_traditional_moe_called'):
            print(f"🎯 拼接融合：使用无预处理（无预处理）")
            print(f"   - 拼接方式: torch.cat() 直接拼接")
            print(f"   - 拼接维度: feat_dim * num_scales = {self.feat_dim} * {len(self.scales)} = {self.feat_dim * len(self.scales)}")
            print(f"   - 步骤：直接拼接 → 门控网络计算专家权重 → 专家网络处理 → 加权融合")
            print()  # 空行
            self._traditional_moe_called = True
        
        # 直接使用专家网络处理原始多尺度特征
        return self._expert_network_processing(multi_scale_features)
    
    def _attention_fusion_features(self, multi_scale_features):
        """
        使用门控加权-预处理融合多尺度特征（替代无预处理）
        
        核心思想：只改变拼接融合方式，不动滑动窗口和专家网络
        
        Args:
            multi_scale_features: List[Tensor] - 多尺度特征列表
        Returns:
            fused_multi_scale_features: List[Tensor] - 门控加权-预处理后的多尺度特征
        """
        # 🔥 门控加权-预处理启动提示（仅在第一次调用时显示，且模块启用时）
        # 注意：这个方法只在 use_gate_fusion=True 时被调用，所以不需要额外检查
        if not hasattr(self, '_attention_fusion_called'):
            print(f"🎯 门控加权-预处理启动！")
            print(f"   - 输入多尺度特征数量: {len(multi_scale_features)}")
            print(f"   - 每个特征形状: {multi_scale_features[0].shape}")
            print(f"   - 门控网络头数: {self.gate_fusion.num_heads}")
            print(f"   - 门控网络Dropout: {self.gate_fusion.dropout}")
            print(f"   - 滑动窗口尺度: {self.scales}")
            print(f"   - 特征维度: {self.feat_dim}")
            self._attention_fusion_called = True
        
        # 使用门控加权-预处理进行特征融合（返回增强后的多尺度特征）
        enhanced_multi_scale_features, gate_weights = self.gate_fusion(multi_scale_features)
        
        # 🔥 门控加权-预处理处理完成提示（仅在第一次调用时显示，且模块启用时）
        # 注意：这个方法只在 use_gate_fusion=True 时被调用，所以不需要额外检查
        if not hasattr(self, '_attention_fusion_complete_called'):
            print(f"✅ 门控加权-预处理完成！")
            print(f"   - 输出多尺度特征数量: {len(enhanced_multi_scale_features)}")
            print(f"   - 每个特征形状: {enhanced_multi_scale_features[0].shape}")
            print(f"   - 门控权重形状: {gate_weights.shape}")
            print(f"   - 门控权重分布: {gate_weights[0].detach().cpu().numpy()}")
            print(f"   - 保持多尺度结构: 不丢失信息")
            self._attention_fusion_complete_called = True
        
        return enhanced_multi_scale_features
    
    def _expert_network_processing(self, multi_scale_features):
        """
        专家网络处理多尺度特征
        
        Args:
            multi_scale_features: List[Tensor] - 多尺度特征列表
        Returns:
            final_feature: [B, feat_dim] - 最终特征
            expert_weights: [B, num_experts] - 专家权重
        """
        # 🔥 专家网络处理提示（仅在第一次调用时显示）
        # 注意：专家网络处理是MoE模块的核心功能，总是会执行，所以总是输出
        if not hasattr(self, '_expert_processing_called'):
            # 根据是否使用固定权重，显示不同的信息
            if self.use_fixed_weights:
                print(f"🎯 专家网络处理：使用固定权重和专家网络")
            else:
                print(f"🎯 专家网络处理：使用门控网络和专家网络")
            print(f"   - 专家数量: {len(self.experts)}")
            # 🔧 修复：从保存的属性或专家网络实例中获取信息
            if hasattr(self, 'expert_hidden_dim'):
                print(f"   - 专家隐藏层维度: {self.expert_hidden_dim}")
            elif hasattr(self.experts[0], 'hidden_dim'):
                print(f"   - 专家隐藏层维度: {self.experts[0].hidden_dim}")
            else:
                print(f"   - 专家隐藏层维度: N/A")
            
            if hasattr(self, 'expert_layers'):
                print(f"   - 专家层数: {self.expert_layers}")
            elif hasattr(self.experts[0], 'num_layers'):
                print(f"   - 专家层数: {self.experts[0].num_layers}")
            else:
                print(f"   - 专家层数: N/A")
            
            if self.use_fixed_weights:
                print(f"   - 固定权重计算专家权重")
            else:
                print(f"   - 门控网络计算专家权重")
            # 🔥 Top-k 路由状态显示
            if self.use_top_k_routing:
                print(f"   - Top-k 路由: 将强制激活 Top-{self.top_k} 专家 (模式: {self.top_k_mode})")
            else:
                print(f"   - Top-k 路由: 未启用 (所有专家都有非零权重)")
            print(f"   - 专家网络处理多尺度特征")
            print(f"   - 加权融合得到最终特征")
            print()  # 空行
            self._expert_processing_called = True
        
        B = multi_scale_features[0].shape[0]
        
        # 🔥 步骤1：拼接多尺度特征作为门控网络输入
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, feat_dim * num_scales]
        
        # 🔥 门控网络处理提示（仅在第一次调用时显示，根据模块状态决定输出内容）
        # 功能：根据 use_fixed_weights 状态显示不同的提示信息
        # 位置：在计算专家权重之前，确保能正确反映当前模式
        if not hasattr(self, '_gating_network_called'):
            if self.use_fixed_weights:
                # 固定权重模式：显示固定权重信息
                print(f"🎯 固定权重模式：使用固定权重（不使用门控网络）")
                print(f"   - 权重值: {self.fixed_weights_tensor.tolist()}")
                print(f"   - 说明：所有样本使用相同的固定权重，权重不随训练改变")
            else:
                # 动态权重模式：显示门控网络信息
                print(f"🎯 门控网络处理：计算专家权重")
                print(f"   - 输入特征形状: {concat_features.shape}")
                print(f"   - 输出权重形状: [{concat_features.shape[0]}, {self.num_experts}]")
                print(f"   - 说明：根据输入特征动态计算专家权重，权重会随训练优化")
                # 🔥 Top-k 路由提示
                if self.use_top_k_routing:
                    print(f"   - ⚠️  注意：Top-k 路由将在此后处理，只保留 Top-{self.top_k} 专家的权重")
            self._gating_network_called = True
        
        # ========== 计算专家权重 ==========
        # 功能：根据配置模式计算专家权重
        #
        # 两种模式：
        #   1. 固定权重模式（use_fixed_weights=True）：
        #      - 所有样本使用相同的固定权重
        #      - 权重不依赖输入特征，不随训练改变
        #      - 实现：将固定权重tensor扩展到batch维度 [B, num_experts]
        #   2. 动态权重模式（use_fixed_weights=False）：
        #      - 每个样本根据其特征动态计算权重
        #      - 权重通过门控网络学习得到，会随训练优化
        #      - 实现：门控网络处理拼接特征，输出权重 [B, num_experts]
        #
        if self.use_fixed_weights:
            # ========== 固定权重模式：使用预设的固定权重 ==========
            # 实现步骤：
            #   1. 获取batch大小
            #   2. 将固定权重tensor从 [num_experts] 扩展到 [B, num_experts]
            #   3. 确保权重tensor在正确的设备上（CPU/GPU）
            #
            # 特点：
            #   - 所有样本使用相同的权重分布
            #   - 权重不依赖输入特征，计算开销小
            #   - 适合调试、对比实验、跨域鲁棒性测试
            #
            B = multi_scale_features[0].shape[0]
            # 将固定权重扩展到batch维度：[num_experts] -> [B, num_experts]
            expert_weights = self.fixed_weights_tensor.unsqueeze(0).expand(B, -1).to(multi_scale_features[0].device)
        else:
            # ========== 动态权重模式：使用门控网络计算权重 ==========
            # 实现步骤：
            #   1. 门控网络接收拼接的多尺度特征 [B, feat_dim * num_scales]
            #   2. 门控网络输出专家权重 [B, num_experts]
            #   3. 权重通过Softmax归一化，确保和为1.0
            #
            # 特点：
            #   - 每个样本根据其特征动态调整权重
            #   - 权重会随训练过程学习优化
            #   - 需要训练门控网络参数，计算开销较大
            #   - 适合需要自适应权重的场景
            #
            expert_weights = self.gating_network(concat_features)  # [B, num_experts]
        
        # 🔥 Top-k 路由处理（如果启用）
        if self.use_top_k_routing:
            # 保存路由前的权重（用于显示对比）
            with torch.no_grad():
                weights_before = expert_weights[0].detach().clone() if expert_weights.shape[0] > 0 else None
            expert_weights = self._apply_top_k_routing(expert_weights)
        else:
            # 传统软路由：显示权重分布（仅在第一次调用时显示）
            if not hasattr(self, '_soft_routing_weights_shown'):
                with torch.no_grad():
                    sample_weights = expert_weights[0].detach().cpu().numpy()
                    print(f"📊 传统软路由权重分布（第一个样本）: [{', '.join([f'{w:.4f}' for w in sample_weights])}]")
                    print(f"   - 说明：所有专家都有非零权重，权重通过 softmax 归一化")
                self._soft_routing_weights_shown = True
        
        # ========== MLP专家网络调用：处理各尺度特征 ==========
        expert_outputs = []
        for i, (expert, feature) in enumerate(zip(self.experts, multi_scale_features)):
            expert_output = expert(feature)  # [B, feat_dim]
            expert_outputs.append(expert_output)
        
        # 🔥 步骤4：加权融合专家输出
        weighted_outputs = []
        for i, expert_output in enumerate(expert_outputs):
            weight = expert_weights[:, i:i+1].expand_as(expert_output)  # [B, feat_dim]
            weighted_output = weight * expert_output  # [B, feat_dim]
            weighted_outputs.append(weighted_output)
        
        # 求和得到融合特征
        fused_feature = torch.sum(torch.stack(weighted_outputs, dim=0), dim=0)  # [B, feat_dim]
        
        # 🔥 最终融合提示（仅在第一次调用时显示）
        if not hasattr(self, '_final_fusion_called'):
            print(f"🎯 最终融合：专家输出融合")
            print(f"   - 融合特征形状: {fused_feature.shape}")
            print(f"   - 最终特征形状: [B, {self.feat_dim}]")
            self._final_fusion_called = True
        
        # 保存最新专家权重供训练结束时输出
        with torch.no_grad():
            self._latest_expert_weights = expert_weights.detach().clone()
        
        # ========== MLP最终融合层调用：专家输出融合 ==========
        final_feature = self.final_fusion(fused_feature)  # [B, feat_dim]
        
        return final_feature, expert_weights
    
    def _apply_top_k_routing(self, expert_weights):
        """
        🔥 Top-k 路由处理
        
        功能：强制激活权重最大的 k 个专家，屏蔽其他专家
        
        Args:
            expert_weights: [B, num_experts] - 原始专家权重（已归一化）
        Returns:
            expert_weights_topk: [B, num_experts] - Top-k 路由后的权重
        """
        B, num_experts = expert_weights.shape
        
        # 🔥 Top-k 路由启动提示（仅在第一次调用时显示）
        if not hasattr(self, '_top_k_routing_called'):
            print(f"🎯 Top-k 路由处理：强制激活 Top-{self.top_k} 专家")
            print(f"   - 输入权重形状: {expert_weights.shape}")
            print(f"   - Top-k 值: {self.top_k} (将激活 {self.top_k}/{num_experts} 个专家)")
            print(f"   - 路由模式: {self.top_k_mode}")
            print(f"   - 专家总数: {num_experts}")
            # 显示第一个样本的原始权重分布（用于对比）
            with torch.no_grad():
                sample_weights = expert_weights[0].detach().cpu().numpy()
                print(f"   - 原始权重分布（第一个样本）: [{', '.join([f'{w:.4f}' for w in sample_weights])}]")
            self._top_k_routing_called = True
        
        if self.top_k_mode == "soft":
            # ========== 软 Top-k 路由：重新归一化 Top-k 权重 ==========
            # 功能：保留被屏蔽专家的梯度，但权重设为 0
            # 优势：减少与 Load Balancing Loss 的冲突，训练更稳定
            # 实现：
            #   1. 找到 Top-k 专家的索引和权重
            #   2. 创建 mask，只保留 Top-k 专家的权重
            #   3. 重新归一化 Top-k 权重，确保和为 1.0
            
            # 获取 Top-k 专家的索引和权重
            topk_values, topk_indices = torch.topk(expert_weights, k=self.top_k, dim=-1)  # [B, k], [B, k]
            
            # 创建 mask：只保留 Top-k 专家的权重
            mask = torch.zeros_like(expert_weights)  # [B, num_experts]
            mask.scatter_(1, topk_indices, 1.0)  # [B, num_experts]，Top-k 位置为 1.0，其他为 0.0
            
            # 应用 mask：屏蔽非 Top-k 专家的权重
            expert_weights_masked = expert_weights * mask  # [B, num_experts]
            
            # 重新归一化 Top-k 权重，确保和为 1.0
            # 注意：这里使用原始权重（而非 topk_values）进行归一化，保留梯度
            expert_weights_sum = expert_weights_masked.sum(dim=-1, keepdim=True)  # [B, 1]
            expert_weights_topk = expert_weights_masked / (expert_weights_sum + 1e-8)  # [B, num_experts]
            
            # 🔥 软 Top-k 路由完成提示（仅在第一次调用时显示）
            if not hasattr(self, '_soft_top_k_complete_called'):
                print(f"✅ 软 Top-k 路由完成")
                print(f"   - 输出权重形状: {expert_weights_topk.shape}")
                print(f"   - Top-{self.top_k} 专家权重已重新归一化")
                print(f"   - 非 Top-{self.top_k} 专家权重已设为 0（但保留梯度）")
                # 显示第一个样本的路由后权重分布（用于对比）
                with torch.no_grad():
                    sample_weights_after = expert_weights_topk[0].detach().cpu().numpy()
                    topk_indices_sample = topk_indices[0].detach().cpu().numpy()
                    print(f"   - 路由后权重分布（第一个样本）: [{', '.join([f'{w:.4f}' for w in sample_weights_after])}]")
                    print(f"   - 激活的专家索引: {topk_indices_sample.tolist()} (对应尺度: {[self.scales[i] for i in topk_indices_sample]})")
                print(f"   - 说明：保留被屏蔽专家的梯度，减少与损失函数的冲突")
                self._soft_top_k_complete_called = True
            
        else:  # self.top_k_mode == "hard"
            # ========== 硬 Top-k 路由：直接 mask 非 Top-k 专家 ==========
            # 功能：完全屏蔽非 Top-k 专家的贡献
            # 优势：更彻底的稀疏激活，推理效率更高
            # 风险：可能丢失关键信息，与 Load Balancing Loss 冲突更严重
            # 实现：
            #   1. 找到 Top-k 专家的索引和权重
            #   2. 创建 mask，只保留 Top-k 专家的权重
            #   3. 重新归一化 Top-k 权重，确保和为 1.0
            #   4. 使用 detach() 完全屏蔽非 Top-k 专家的梯度
            
            # 获取 Top-k 专家的索引和权重
            topk_values, topk_indices = torch.topk(expert_weights, k=self.top_k, dim=-1)  # [B, k], [B, k]
            
            # 创建 mask：只保留 Top-k 专家的权重
            mask = torch.zeros_like(expert_weights)  # [B, num_experts]
            mask.scatter_(1, topk_indices, 1.0)  # [B, num_experts]，Top-k 位置为 1.0，其他为 0.0
            
            # 应用 mask：屏蔽非 Top-k 专家的权重
            expert_weights_masked = expert_weights * mask  # [B, num_experts]
            
            # 重新归一化 Top-k 权重，确保和为 1.0
            expert_weights_sum = expert_weights_masked.sum(dim=-1, keepdim=True)  # [B, 1]
            expert_weights_topk = expert_weights_masked / (expert_weights_sum + 1e-8)  # [B, num_experts]
            
            # 🔥 硬 Top-k 路由：完全屏蔽非 Top-k 专家的梯度
            # 使用 stop_gradient 技巧：保留 Top-k 专家的梯度，屏蔽非 Top-k 专家的梯度
            # 实现：expert_weights_topk = expert_weights_topk + (expert_weights * mask - expert_weights_topk).detach()
            # 这样 Top-k 专家的梯度会保留，非 Top-k 专家的梯度会被屏蔽
            expert_weights_topk = expert_weights_topk + (expert_weights * mask - expert_weights_topk).detach()
            
            # 🔥 硬 Top-k 路由完成提示（仅在第一次调用时显示）
            if not hasattr(self, '_hard_top_k_complete_called'):
                print(f"✅ 硬 Top-k 路由完成")
                print(f"   - 输出权重形状: {expert_weights_topk.shape}")
                print(f"   - Top-{self.top_k} 专家权重已重新归一化")
                print(f"   - 非 Top-{self.top_k} 专家权重已设为 0（完全屏蔽梯度）")
                # 显示第一个样本的路由后权重分布（用于对比）
                with torch.no_grad():
                    sample_weights_after = expert_weights_topk[0].detach().cpu().numpy()
                    topk_indices_sample = topk_indices[0].detach().cpu().numpy()
                    print(f"   - 路由后权重分布（第一个样本）: [{', '.join([f'{w:.4f}' for w in sample_weights_after])}]")
                    print(f"   - 激活的专家索引: {topk_indices_sample.tolist()} (对应尺度: {[self.scales[i] for i in topk_indices_sample]})")
                print(f"   - ⚠️  警告：可能丢失关键信息，与损失函数冲突更严重")
                self._hard_top_k_complete_called = True
        
        return expert_weights_topk


class AttentionFusionConcat(nn.Module):
    """
    🔥 注意力-预处理模块
    
    核心功能：
    - 使用多头注意力机制学习特征间关系
    - 智能注意力加权融合多尺度特征
    - 实现基于注意力的特征融合
    """
    
    def __init__(self, feat_dim=512, num_heads=8, scales=[4, 8, 16], dropout=0.1, attention_dim=512):
        super(AttentionFusionConcat, self).__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.scales = scales
        self.dropout = dropout
        self.attention_dim = attention_dim
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力-预处理网络
        self.attention_fusion = nn.Sequential(
            nn.Linear(feat_dim * len(scales), attention_dim),
            nn.LayerNorm(attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(attention_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2)
        )
        
        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2)
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
        注意力-预处理前向传播
        
        Args:
            multi_scale_features: List[Tensor] - 多尺度特征列表
        Returns:
            enhanced_multi_scale_features: List[Tensor] - 注意力-预处理后的多尺度特征
            attention_weights: [B, num_scales, num_scales] - 注意力权重矩阵
        """
        B = multi_scale_features[0].shape[0]
        
        # 🔥 注意力-预处理启动提示（仅在第一次调用时显示，且模块启用时）
        # 注意：这个方法只在 use_attention_fusion=True 时被调用，所以不需要额外检查
        if not hasattr(self, '_attention_fusion_called'):
            print(f"🎯 注意力-预处理机制启动！")
            print(f"   - 输入多尺度特征数量: {len(multi_scale_features)}")
            print(f"   - 每个特征形状: {multi_scale_features[0].shape}")
            print(f"   - 注意力头数: {self.num_heads}")
            print(f"   - 注意力维度: {self.attention_dim}")
            print(f"   - 滑动窗口尺度: {self.scales}")
            self._attention_fusion_called = True
        
        # 🔥 步骤1：构建注意力输入序列
        # 将多尺度特征堆叠为序列：[B, num_scales, feat_dim]
        attention_input = torch.stack(multi_scale_features, dim=1)  # [B, 3, 512]
        
        # 🔥 步骤2：多头注意力计算
        # 使用自注意力机制学习多尺度特征间的关系
        attn_output, attn_weights = self.multihead_attn(
            attention_input, attention_input, attention_input
        )  # [B, 3, 512], [B, 3, 3]
        
        # 🔥 步骤3：注意力加权融合
        enhanced_multi_scale_features = []
        for i, (original_feat, attn_feat) in enumerate(zip(multi_scale_features, attn_output.unbind(1))):
            # 注意力加权
            weighted_feat = attn_feat  # [B, feat_dim]
            
            # 特征增强
            enhanced_feat = self.feature_enhancer(weighted_feat)
            
            # 残差连接，保持原始信息
            enhanced_feat = enhanced_feat + original_feat * 0.3  # 残差连接
            
            enhanced_multi_scale_features.append(enhanced_feat)
        
        # 🔥 步骤4：全局注意力-预处理融合
        # 将注意力处理后的特征进行全局融合
        concat_features = torch.cat(enhanced_multi_scale_features, dim=1)  # [B, feat_dim * num_scales]
        global_fusion = self.attention_fusion(concat_features)  # [B, feat_dim]
        
        # 将全局融合特征添加到每个尺度特征中
        for i in range(len(enhanced_multi_scale_features)):
            enhanced_multi_scale_features[i] = enhanced_multi_scale_features[i] + global_fusion * 0.2
        
        # 🔥 注意力-预处理处理完成提示（仅在第一次调用时显示，且模块启用时）
        # 注意：这个方法只在 use_attention_fusion=True 时被调用，所以不需要额外检查
        if not hasattr(self, '_attention_fusion_completed'):
            print(f"✅ 注意力-预处理处理完成！")
            print(f"   - 注意力权重形状: {attn_weights.shape}")
            print(f"   - 输出特征数量: {len(enhanced_multi_scale_features)}")
            print(f"   - 每个特征形状: {enhanced_multi_scale_features[0].shape}")
            self._attention_fusion_completed = True
        
        return enhanced_multi_scale_features
    
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
    
    def print_final_expert_weights(self):
        """
        训练结束时输出最终专家权重分布
        """
        if hasattr(self, '_latest_expert_weights') and self._latest_expert_weights is not None:
            with torch.no_grad():
                # 计算最终专家权重
                final_weights = torch.mean(self._latest_expert_weights, dim=0).cpu().numpy()
                
                print(f"🎯 训练完成 - 最终专家权重分布:")
                print(f"📊 专家权重占比:")
                
                # 动态打印专家权重，避免索引越界
                for i in range(len(final_weights)):
                    scale_name = f"{4*(i+1)}x{4*(i+1)}" if i < 3 else f"专家{i+1}"
                    percentage = final_weights[i] * 100
                    print(f"   {scale_name}专家: {final_weights[i]:.4f} ({percentage:.1f}%)")
                
                # 打印权重分布数组
                weight_str = ", ".join([f"{final_weights[i]:.4f}" for i in range(len(final_weights))])
                print(f"   最终权重分布: [{weight_str}]")
                
                return final_weights
        else:
            print("⚠️ 未找到最终专家权重信息")
            return None


class CLIPMultiScaleMoE(nn.Module):
    """
    🔥 CLIP兼容的多尺度MoE特征提取器
    
    集成多尺度滑动窗口和MoE机制
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16], expert_hidden_dim=1024, temperature=1.0,
                 expert_dropout=0.1, gate_dropout=0.1, expert_layers=2, gate_layers=2, 
                 expert_threshold=0.1, residual_weight=1.0, use_gate_fusion=False,
                 use_attention_fusion=False, attention_num_heads=8,
                 attention_dropout=0.1, attention_dim=512, init_weights=None,
                 use_fixed_weights=False, fixed_weights=None,
                 use_top_k_routing=False, top_k=2, top_k_mode="soft"):
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
            use_gate_fusion (bool): 是否使用门控加权-预处理机制（使用MLP门控网络）
            use_attention_fusion (bool): 是否使用注意力-预处理机制（使用多头注意力）
            attention_num_heads (int): 注意力网络头数（仅在use_attention_fusion=True时使用）
            attention_dropout (float): 注意力网络Dropout比例
            attention_dim (int): 注意力网络维度
            init_weights (list, optional): 专家初始权重列表，如 [0.35, 0.3, 0.35]
            use_fixed_weights (bool): 是否使用固定权重模式
                - True: 使用固定权重，禁用门控网络，专家权重固定不变
                - False: 使用门控网络动态计算权重（默认）
                使用场景：
                  1. 调试：排除门控网络影响，专注于专家网络性能
                  2. 对比实验：对比固定权重 vs 动态权重的效果
                  3. 跨域鲁棒性：固定权重可能在跨域场景下更稳定
            fixed_weights (list, optional): 固定权重列表，如 [0.33, 0.33, 0.34]
                - 仅在 use_fixed_weights=True 时生效
                - 长度必须等于专家数量（len(scales)）
                - 权重会自动归一化，无需手动确保和为1.0
                - 示例：[0.33, 0.33, 0.34] 表示三个专家权重分别为33%、33%、34%
                - 如果为None且use_fixed_weights=True，会抛出ValueError
            use_top_k_routing (bool): 是否使用 Top-k 路由机制（默认 False）
            top_k (int): Top-k 路由的 k 值（默认 2，即 Top-2）
            top_k_mode (str): Top-k 路由模式（默认 "soft"）
                - "soft": 软 Top-k，重新归一化 Top-k 权重，保留被屏蔽专家的梯度（推荐）
                - "hard": 硬 Top-k，直接 mask 非 Top-k 专家，完全屏蔽其贡献
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
            use_gate_fusion=use_gate_fusion,
            use_attention_fusion=use_attention_fusion,
            attention_num_heads=attention_num_heads,
            attention_dropout=attention_dropout,
            attention_dim=attention_dim,
            init_weights=init_weights,
            use_fixed_weights=use_fixed_weights,
            fixed_weights=fixed_weights,
            use_top_k_routing=use_top_k_routing,
            top_k=top_k,
            top_k_mode=top_k_mode
        )
        
        print(f"🔥 CLIP多尺度MoE模块初始化完成:")
        print(f"   - 特征维度: {feat_dim}")
        print(f"   - 滑动窗口尺度: {scales}")
        print(f"   - 专家隐藏层维度: {expert_hidden_dim}")
        print(f"   - 门控加权-预处理机制: {'已启用' if use_gate_fusion else '已禁用'}")
        print(f"   - 注意力-预处理机制: {'已启用' if use_attention_fusion else '已禁用'}")
        if init_weights is not None:
            print(f"   - 专家初始权重: {init_weights}")
    
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
            print()  # 空行
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
