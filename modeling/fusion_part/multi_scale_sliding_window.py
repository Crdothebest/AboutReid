"""
T2T-ViT兼容的多尺度滑动窗口模块

功能：
- 为T2T-ViT模型提供多尺度滑动窗口特征提取
- 实现4x4、8x8、16x16滑动窗口的多尺度特征融合
- 适配T2T-ViT的特征维度（如512维）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureExtractor(nn.Module):
    """
    T2T-ViT兼容的多尺度滑动窗口模块
    
    核心功能：
    - 实现4x4、8x8、16x16多尺度滑动窗口特征提取
    - 适配T2T-ViT的特征维度（如512维）
    - 通过MLP融合多尺度特征
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16]):
        """
        初始化多尺度滑动窗口模块
        
        Args:
            feat_dim (int): 特征维度，T2T-ViT通常为512维
            scales (list): 滑动窗口尺度列表 [4, 8, 16]
        """
        super(MultiScaleFeatureExtractor, self).__init__()
        self.feat_dim = feat_dim
        self.scales = scales
        
        # 为每个尺度创建滑动窗口处理层
        self.sliding_windows = nn.ModuleList()
        for scale in scales:
            self.sliding_windows.append(
                nn.Conv1d(feat_dim, feat_dim, kernel_size=scale, stride=scale, padding=0)
            )
        
        # MLP融合层：多尺度特征融合器
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * len(scales), feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim)
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
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, patch_tokens):
        """
        多尺度滑动窗口前向传播
        
        Args:
            patch_tokens: [B, N, feat_dim] - T2T-ViT的patch tokens
        
        Returns:
            multi_scale_feature: [B, feat_dim] - 多尺度融合特征
        """
        B, N, D = patch_tokens.shape
        
        # 转换为卷积输入格式 [B, D, N]
        x = patch_tokens.transpose(1, 2)  # [B, feat_dim, N]
        
        multi_scale_features = []
        for i, scale in enumerate(self.scales):
            if N >= scale:
                windowed_feat = self.sliding_windows[i](x)  # [B, feat_dim, N//scale]
                pooled_feat = F.adaptive_avg_pool1d(windowed_feat, 1).squeeze(-1)  # [B, feat_dim]
            else:
                pooled_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, feat_dim]
            
            multi_scale_features.append(pooled_feat)
        
        # 拼接多尺度特征
        concat_feat = torch.cat(multi_scale_features, dim=1)  # [B, feat_dim*3]
        
        # MLP融合处理
        multi_scale_feature = self.fusion(concat_feat)  # [B, feat_dim*3] -> [B, feat_dim]
        
        return multi_scale_feature

