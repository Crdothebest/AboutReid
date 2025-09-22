"""
CLIP兼容的多尺度滑动窗口模块

功能：
- 在保持CLIP分支完整性的基础上，添加多尺度滑动窗口特征提取
- 适配CLIP的768维特征到多尺度处理
- 实现4x4、8x8、16x16滑动窗口的多尺度特征融合

作者：用户修改
日期：2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPMultiScaleSlidingWindow(nn.Module):
    """
    🔥 CLIP兼容的多尺度滑动窗口模块
    
    核心功能：
    - 实现4x4、8x8、16x16多尺度滑动窗口特征提取
    - 适配CLIP的512维投影特征
    - 通过MLP融合多尺度特征
    """
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16]):
        """
        🎯 初始化多尺度滑动窗口模块
        
        Args:
            feat_dim (int): 特征维度，CLIP投影输出512维
            scales (list): 滑动窗口尺度列表 [4, 8, 16]
        """
        super(CLIPMultiScaleSlidingWindow, self).__init__()
        self.feat_dim = feat_dim  # CLIP投影的512维输出
        self.scales = scales      # 滑动窗口尺度 [4, 8, 16]
        
        # 🔥 为每个尺度创建滑动窗口处理层
        # 使用1D卷积实现滑动窗口效果：kernel_size=scale, stride=scale
        self.sliding_windows = nn.ModuleList()
        for scale in scales:
            # 每个尺度独立处理：4x4, 8x8, 16x16
            self.sliding_windows.append(
                nn.Conv1d(feat_dim, feat_dim, kernel_size=scale, stride=scale, padding=0)
            )
        
        # 🔥 特征融合层 (MLP) - 关键创新点
        # 将所有尺度的特征拼接后，通过MLP融合回原始维度
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * len(scales), feat_dim), # 第一层：1536 -> 512 (3个尺度×512维)
            nn.ReLU(),                                   # 激活函数
            nn.Dropout(0.1),                             # Dropout正则化
            nn.Linear(feat_dim, feat_dim)                # 第二层：512 -> 512 (保持维度)
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
        🎯 多尺度滑动窗口前向传播
        
        Args:
            patch_tokens: [B, N, 512] - CLIP投影的patch tokens
        Returns:
            multi_scale_feature: [B, 512] - 多尺度融合特征
        """
        B, N, D = patch_tokens.shape
        
        # 🔥 转换为卷积输入格式 [B, D, N]
        # 1D卷积需要 [B, C, L] 格式，所以需要转置
        x = patch_tokens.transpose(1, 2)  # [B, 512, N]
        
        multi_scale_features = []
        for i, scale in enumerate(self.scales):
            # 🔥 滑动窗口处理 - 核心算法
            if N >= scale:
                # 使用1D卷积进行滑动窗口处理
                # 每个窗口处理scale个tokens，输出N//scale个特征
                windowed_feat = self.sliding_windows[i](x)  # [B, 512, N//scale]
                # 全局平均池化：将每个尺度的特征聚合为单个向量
                pooled_feat = F.adaptive_avg_pool1d(windowed_feat, 1)  # [B, 512, 1]
                pooled_feat = pooled_feat.squeeze(-1)  # [B, 512]
            else:
                # 如果序列长度小于窗口大小，直接使用全局平均池化
                pooled_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, 512]
            
            multi_scale_features.append(pooled_feat)
        
        # 🔥 拼接多尺度特征
        # 将3个尺度的特征拼接：4x4 + 8x8 + 16x16 = 1536维
        concat_feat = torch.cat(multi_scale_features, dim=1)  # [B, 512*3] = [B, 1536]
        
        # 🔥 特征融合 (MLP) - 关键创新点
        # 通过两层MLP将1536维融合回512维
        multi_scale_feature = self.fusion(concat_feat)  # [B, 1536] -> [B, 512]
        
        return multi_scale_feature


class CLIPMultiScaleFeatureExtractor(nn.Module):
    """CLIP多尺度特征提取器包装类"""
    
    def __init__(self, feat_dim=512, scales=[4, 8, 16]):
        """
        初始化CLIP多尺度特征提取器
        
        Args:
            feat_dim (int): 特征维度，CLIP投影输出512维
            scales (list): 滑动窗口尺度列表
        """
        super(CLIPMultiScaleFeatureExtractor, self).__init__()
        self.multi_scale_window = CLIPMultiScaleSlidingWindow(feat_dim, scales)
        
    def forward(self, patch_tokens):
        """
        前向传播
        
        Args:
            patch_tokens: [B, N, 512] - CLIP投影的patch tokens
        Returns:
            multi_scale_feature: [B, 512] - 多尺度特征
        """
        return self.multi_scale_window(patch_tokens)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 2
    seq_len = 196  # 14x14 patches
    feat_dim = 768  # CLIP实际维度
    
    # 创建模型
    model = CLIPMultiScaleFeatureExtractor(feat_dim=feat_dim, scales=[4, 8, 16])
    
    # 创建测试输入
    patch_tokens = torch.randn(batch_size, seq_len, feat_dim)
    
    print("=== CLIP多尺度滑动窗口模块测试 ===")
    print(f"输入形状: {patch_tokens.shape}")
    
    # 前向传播
    with torch.no_grad():
        multi_scale_feature = model(patch_tokens)
    
    print(f"输出形状: {multi_scale_feature.shape}")
    print(f"期望形状: [{batch_size}, {feat_dim}]")
    
    # 验证输出形状
    assert multi_scale_feature.shape == (batch_size, feat_dim), f"输出形状不匹配: {multi_scale_feature.shape}"
    
    print("✅ 测试通过！CLIP多尺度滑动窗口模块工作正常")
    print(f"   - 输入: {patch_tokens.shape}")
    print(f"   - 输出: {multi_scale_feature.shape}")
    print(f"   - 滑动窗口尺度: [4, 8, 16]")
    print(f"   - 特征维度: {feat_dim}")
