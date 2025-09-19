"""
多尺度Mixture-of-Experts (MoE) 模块
基于idea-01.png的创新设计

作者修改：新增多尺度MoE模块，实现基于idea-01.png的创新设计
功能：实现多尺度特征提取和专家网络融合机制
撤销方法：删除整个文件

核心功能：
1. 多尺度特征提取（4x4, 8x8, 16x16滑动窗口）
2. 专家网络分配（每个尺度对应一个专家）
3. 动态权重计算
4. 特征融合

技术原理：
- 滑动窗口：将token序列按不同尺度分组，捕获多尺度信息
- 专家网络：为每个尺度训练专门的专家网络
- MoE门控：动态计算专家权重，自适应选择重要信息
- 特征融合：将多尺度专家输出进行加权融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.vit_pytorch import trunc_normal_


class MultiScaleSlidingWindow(nn.Module):
    """
    多尺度滑动窗口特征提取
    
    作者修改：实现idea-01.png中的多尺度特征提取部分
    功能：使用不同大小的滑动窗口（4x4, 8x8, 16x16）提取多尺度特征
    撤销方法：删除整个类定义
    """
    
    def __init__(self, dim, scales=[4, 8, 16]):
        super().__init__()
        self.scales = scales  # 滑动窗口尺度列表 [4, 8, 16]
        self.dim = dim        # 特征维度
        
        # 为每个尺度创建专门的特征提取器
        # 每个提取器包含：线性变换 + 层归一化 + GELU激活
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),    # 线性变换
                nn.LayerNorm(dim),      # 层归一化
                nn.GELU()               # GELU激活函数
            ) for _ in scales  # 为每个尺度创建一个提取器
        ])
        
    def forward(self, x):
        """
        多尺度滑动窗口特征提取前向传播
        
        作者修改：实现idea-01.png中的滑动窗口分组和特征提取逻辑
        功能：将输入序列按不同尺度分组，提取多尺度特征
        撤销方法：删除整个forward方法
        
        Args:
            x: [B, N, D] - 输入token序列 (batch_size, sequence_length, feature_dim)
        
        Returns:
            scale_features: List[Tensor] - 多尺度特征列表，每个元素为[B, D]
        """
        B, N, D = x.shape
        scale_features = []  # 存储每个尺度的特征
        
        for i, scale in enumerate(self.scales):
            # 计算滑动窗口
            if N >= scale:
                # 滑动窗口分组：将序列按指定尺度分组
                num_windows = N - scale + 1  # 可生成的窗口数量
                windows = []
                for j in range(num_windows):
                    # 提取第j个窗口的token序列
                    window_tokens = x[:, j:j+scale, :]  # [B, scale, D]
                    windows.append(window_tokens)
                
                # 堆叠所有窗口：[B, num_windows, scale, D]
                windows = torch.stack(windows, dim=1)
                # 重塑为：[B*num_windows, scale, D] 便于批量处理
                windows = windows.view(B * num_windows, scale, D)
                
                # 通过对应的特征提取器处理
                scale_feature = self.scale_extractors[i](windows)  # [B*num_windows, scale, D]
                
                # 全局平均池化得到尺度特征
                scale_feature = torch.mean(scale_feature, dim=1)  # [B*num_windows, D]
                scale_feature = scale_feature.view(B, num_windows, D)
                scale_feature = torch.mean(scale_feature, dim=1)  # [B, D] 最终尺度特征
                
            else:
                # 如果序列长度小于窗口大小，直接使用全局特征
                scale_feature = torch.mean(x, dim=1)  # [B, D] 全局平均池化
                scale_feature = self.scale_extractors[i](scale_feature.unsqueeze(1)).squeeze(1)
            
            scale_features.append(scale_feature)  # 添加到特征列表
        
        return scale_features


class ExpertNetwork(nn.Module):
    """
    专家网络 - 处理特定尺度的特征
    
    作者修改：实现idea-01.png中的专家网络部分
    功能：为每个尺度创建专门的专家网络，处理对应尺度的特征
    撤销方法：删除整个类定义
    """
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        # 设置隐藏层维度，默认为输入维度的2倍
        if hidden_dim is None:
            hidden_dim = dim * 2
            
        # 专家网络结构：输入 -> 隐藏层 -> 输出
        self.expert = nn.Sequential(
            nn.Linear(dim, hidden_dim),    # 第一层线性变换
            nn.LayerNorm(hidden_dim),      # 层归一化
            nn.GELU(),                     # GELU激活函数
            nn.Dropout(0.1),               # Dropout防止过拟合
            nn.Linear(hidden_dim, dim),    # 第二层线性变换
            nn.LayerNorm(dim)              # 输出层归一化
        )
        
    def forward(self, x):
        """
        专家网络前向传播
        
        作者修改：实现专家网络的特征处理逻辑
        功能：将输入特征通过专家网络处理，输出增强后的特征
        撤销方法：删除整个forward方法
        
        Args:
            x: [B, D] - 输入特征
        
        Returns:
            [B, D] - 专家网络处理后的特征
        """
        return self.expert(x)


class MoEGate(nn.Module):
    """
    MoE门控网络 - 计算专家权重
    
    作者修改：实现idea-01.png中的专家权重计算部分
    功能：动态计算每个专家的权重，实现自适应特征融合
    撤销方法：删除整个类定义
    """
    
    def __init__(self, dim, num_experts):
        super().__init__()
        # 门控网络结构：输入 -> 压缩 -> 专家数量 -> Softmax
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),        # 第一层：维度压缩
            nn.GELU(),                       # GELU激活函数
            nn.Linear(dim // 2, num_experts), # 第二层：输出专家数量
            nn.Softmax(dim=-1)               # Softmax归一化，确保权重和为1
        )
        
    def forward(self, x):
        """
        MoE门控网络前向传播
        
        作者修改：实现专家权重的动态计算逻辑
        功能：根据输入特征计算每个专家的权重分布
        撤销方法：删除整个forward方法
        
        Args:
            x: [B, D] - 输入特征
        
        Returns:
            [B, num_experts] - 专家权重分布
        """
        return self.gate(x)


class MultiScaleMoE(nn.Module):
    """
    多尺度Mixture-of-Experts模块
    
    作者修改：实现idea-01.png中的完整多尺度MoE流程
    功能：整合多尺度特征提取、专家网络处理和特征融合
    撤销方法：删除整个类定义
    """
    
    def __init__(self, dim, scales=[4, 8, 16], num_experts_per_scale=1):
        super().__init__()
        self.dim = dim                    # 特征维度
        self.scales = scales              # 滑动窗口尺度列表
        self.num_scales = len(scales)     # 尺度数量
        
        # 多尺度特征提取模块
        self.multi_scale_extractor = MultiScaleSlidingWindow(dim, scales)
        
        # 专家网络：为每个尺度创建一个专家
        self.experts = nn.ModuleList([
            ExpertNetwork(dim) for _ in range(self.num_scales)
        ])
        
        # MoE门控网络：计算专家权重
        self.moe_gate = MoEGate(dim, self.num_scales)
        
        # 特征融合：将多尺度特征融合为统一表示
        self.fusion = nn.Sequential(
            nn.Linear(dim * self.num_scales, dim),  # 维度压缩
            nn.LayerNorm(dim),                      # 层归一化
            nn.GELU()                               # GELU激活
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        多尺度MoE前向传播
        
        作者修改：实现idea-01.png中的完整多尺度MoE处理流程
        功能：多尺度特征提取 -> 专家网络处理 -> 权重计算 -> 特征融合
        撤销方法：删除整个forward方法
        
        Args:
            x: [B, N, D] - 输入token序列 (batch_size, sequence_length, feature_dim)
        
        Returns:
            [B, D] - 融合后的特征
        """
        # 1. 多尺度特征提取：使用滑动窗口提取不同尺度的特征
        scale_features = self.multi_scale_extractor(x)  # List of [B, D]
        
        # 2. 专家网络处理：每个专家处理对应尺度的特征
        expert_outputs = []
        for i, scale_feature in enumerate(scale_features):
            expert_output = self.experts[i](scale_feature)  # 专家i处理尺度i的特征
            expert_outputs.append(expert_output)
        
        # 3. 计算专家权重：使用门控网络计算每个专家的权重
        # 使用第一个尺度的特征作为门控输入（也可以使用其他策略）
        gate_weights = self.moe_gate(scale_features[0])  # [B, num_scales]
        
        # 4. 加权融合专家输出：根据权重对专家输出进行加权
        weighted_outputs = []
        for i, expert_output in enumerate(expert_outputs):
            weight = gate_weights[:, i:i+1]  # [B, 1] 获取第i个专家的权重
            weighted_output = expert_output * weight  # 加权处理
            weighted_outputs.append(weighted_output)
        
        # 5. 拼接所有加权输出：将多尺度特征拼接
        fused_features = torch.cat(weighted_outputs, dim=-1)  # [B, D*num_scales]
        
        # 6. 最终融合：通过融合网络得到最终特征
        final_features = self.fusion(fused_features)  # [B, D]
        
        return final_features


class MultiScaleMoEAAM(nn.Module):
    """
    集成多尺度MoE的AAM模块
    
    作者修改：将多尺度MoE模块集成到原有的AAM模块中
    功能：结合原始Mamba机制和多尺度MoE，实现更强大的特征融合
    撤销方法：删除整个类定义
    """
    
    def __init__(self, dim, n_layers, cfg, scales=[4, 8, 16]):
        super().__init__()
        self.dim = dim        # 特征维度
        self.scales = scales  # 滑动窗口尺度
        
        # 原始Mamba块：保持原有的Mamba处理能力
        from modeling.fusion_part.mamba import MM_SS2D
        self.ma_block = nn.Sequential(*[MM_SS2D(d_model=dim, cfg=cfg) for _ in range(n_layers)])
        
        # 多尺度MoE模块：新增的多尺度特征处理能力
        self.multi_scale_moe = MultiScaleMoE(dim, scales)
        
        # 特征降维：将CLS token和MoE特征融合后降维
        self.linear_reduction_r = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        self.linear_reduction_n = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        self.linear_reduction_t = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        
        print("Multi-Scale MoE AAM HERE!!!")  # 调试信息
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, r, n, t):
        """
        多尺度MoE AAM前向传播
        
        作者修改：实现结合Mamba和多尺度MoE的特征融合流程
        功能：原始Mamba处理 + 多尺度MoE处理 + 特征融合
        撤销方法：删除整个forward方法
        
        Args:
            r: [B, N, D] - RGB模态的token序列
            n: [B, N, D] - NI模态的token序列  
            t: [B, N, D] - TI模态的token序列
        
        Returns:
            [B, 3*D] - 融合后的三模态特征
        """
        # 提取CLS token：每个模态的全局表示
        cls_r = r[:, 0]  # [B, D] RGB的CLS token
        cls_n = n[:, 0]  # [B, D] NI的CLS token
        cls_t = t[:, 0]  # [B, D] TI的CLS token
        
        # 提取patch tokens：每个模态的局部特征
        r_patches = r[:, 1:]  # [B, N-1, D] RGB的patch tokens
        n_patches = n[:, 1:]  # [B, N-1, D] NI的patch tokens
        t_patches = t[:, 1:]  # [B, N-1, D] TI的patch tokens
        
        # 1. 原始Mamba处理：保持原有的Mamba特征交互能力
        for i in range(len(self.ma_block)):
            r_patches, n_patches, t_patches = self.ma_block[i](r_patches, n_patches, t_patches)
        
        # 2. 多尺度MoE处理：使用新的多尺度特征提取和专家网络
        r_moe_feature = self.multi_scale_moe(r_patches)  # [B, D] RGB的多尺度MoE特征
        n_moe_feature = self.multi_scale_moe(n_patches)  # [B, D] NI的多尺度MoE特征
        t_moe_feature = self.multi_scale_moe(t_patches)  # [B, D] TI的多尺度MoE特征
        
        # 3. 传统patch特征（平均池化）：作为对比基准
        patch_r = torch.mean(r_patches, dim=1)  # [B, D] RGB的全局平均特征
        patch_n = torch.mean(n_patches, dim=1)  # [B, D] NI的全局平均特征
        patch_t = torch.mean(t_patches, dim=1)  # [B, D] TI的全局平均特征
        
        # 4. 特征融合：将CLS token和多尺度MoE特征进行融合
        # 注意：这里只使用CLS token和MoE特征，不使用传统patch特征
        r_feature = self.linear_reduction_r(torch.cat([cls_r, r_moe_feature], dim=-1))  # [B, D]
        n_feature = self.linear_reduction_n(torch.cat([cls_n, n_moe_feature], dim=-1))  # [B, D]
        t_feature = self.linear_reduction_t(torch.cat([cls_t, t_moe_feature], dim=-1))  # [B, D]
        
        # 5. 最终输出：拼接三种模态的融合特征
        return torch.cat([r_feature, n_feature, t_feature], dim=-1)  # [B, 3*D]
