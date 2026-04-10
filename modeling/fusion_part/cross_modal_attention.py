"""
跨模态注意力融合模块 (Cross-modal Attention Fusion)

功能：
- 实现视觉特征与文本特征的双向注意力交互
- 支持多种融合策略：注意力融合、特征拼接、残差增强
- 开关控制设计，支持动态启用/禁用

作者：AboutReid项目组
基于：IDEA项目的跨模态注意力机制
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union


class CrossModalAttentionFusion(nn.Module):
    """
    跨模态注意力融合模块

    实现视觉特征与文本特征的注意力机制交互：
    - Vis → Text: 用视觉特征作为Query，文本特征作为Key/Value
    - Text → Vis: 用文本特征作为Query，视觉特征作为Key/Value
    - 双向融合: 结合两种注意力结果

    Args:
        embed_dim: 特征维度 (默认512)
        num_heads: 注意力头数 (默认8)
        dropout: Dropout比例 (默认0.1)
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1, input_dim: int = None, text_dim: int = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.input_dim = input_dim or embed_dim  # 如果没有指定输入维度，默认使用embed_dim
        self.text_dim = text_dim or embed_dim    # 如果没有指定文本维度，默认使用embed_dim

        # 自适应投影层：将任意维度的输入投影到embed_dim
        self.vis_proj = nn.Linear(self.input_dim, embed_dim) if self.input_dim != embed_dim else nn.Identity()
        self.text_proj = nn.Linear(self.text_dim, embed_dim)  # 支持不同文本输入维度

        # 视觉→文本注意力
        self.vis2text_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 文本→视觉注意力
        self.text2vis_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 特征融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 输出投影 (保持维度)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, visual_tokens: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            visual_tokens: 视觉特征 [B, seq_len, embed_dim] 或 [B, embed_dim]
            text_feat: 文本特征 [B, embed_dim]

        Returns:
            torch.Tensor: 融合后的特征 [B, embed_dim]
        """
        batch_size = visual_tokens.size(0)

        # 处理视觉特征维度
        if visual_tokens.dim() == 2:
            # 如果是全局特征，扩展为序列维度
            visual_global = visual_tokens.unsqueeze(1)  # [B, 1, embed_dim]
        else:
            # 如果是tokens序列，取CLS token作为全局特征
            visual_global = visual_tokens[:, :1]  # [B, 1, embed_dim] - CLS token

        # 确保文本特征是正确的维度
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)  # [B, 1, embed_dim]

        # 投影到相同空间
        visual_proj = self.vis_proj(visual_global)  # [B, 1, embed_dim]
        text_proj = self.text_proj(text_feat)       # [B, 1, embed_dim]

        # 视觉引导文本注意力 (Vis → Text)
        vis_enhanced_text, _ = self.vis2text_attention(
            query=visual_proj,     # [B, 1, embed_dim]
            key=text_proj,         # [B, 1, embed_dim]
            value=text_proj        # [B, 1, embed_dim]
        )  # 输出: [B, 1, embed_dim]

        # 文本引导视觉注意力 (Text → Vis)
        text_enhanced_vis, _ = self.text2vis_attention(
            query=text_proj,       # [B, 1, embed_dim]
            key=visual_proj,       # [B, 1, embed_dim]
            value=visual_proj      # [B, 1, embed_dim]
        )  # 输出: [B, 1, embed_dim]

        # 特征拼接与融合
        combined = torch.cat([
            text_enhanced_vis.squeeze(1),  # [B, embed_dim]
            vis_enhanced_text.squeeze(1)   # [B, embed_dim]
        ], dim=-1)  # [B, embed_dim * 2]

        # MLP融合
        fused = self.fusion_mlp(combined)  # [B, embed_dim]

        return fused


class TextConcatFusion(nn.Module):
    """
    文本特征拼接融合 (简单高效版本)

    直接将视觉特征和文本特征拼接，然后通过MLP降维融合
    """

    def __init__(self, embed_dim: int = 512, dropout: float = 0.1, input_dim: int = None, text_dim: int = None):
        super().__init__()
        self.input_dim = input_dim or embed_dim
        self.text_dim = text_dim or embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.input_dim + self.text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, visual_tokens: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        # 提取视觉全局特征
        if visual_tokens.dim() == 2:
            visual_global = visual_tokens  # [B, embed_dim]
        else:
            visual_global = visual_tokens[:, 0]  # [B, embed_dim] - CLS token

        # 拼接特征
        combined = torch.cat([visual_global, text_feat], dim=-1)  # [B, embed_dim * 2]

        # MLP融合
        return self.fusion_mlp(combined)  # [B, embed_dim]


class TextResidualFusion(nn.Module):
    """
    文本残差增强融合 (保留原始视觉信息)

    使用文本特征作为残差项增强视觉特征，保留原始视觉信息
    """

    def __init__(self, embed_dim: int = 512, fusion_weight: float = 0.3, dropout: float = 0.1, text_dim: int = None):
        super().__init__()
        self.fusion_weight = fusion_weight
        self.text_dim = text_dim or embed_dim

        # 文本特征适配器
        self.text_adapter = nn.Sequential(
            nn.Linear(self.text_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, visual_tokens: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        # 提取视觉全局特征
        if visual_tokens.dim() == 2:
            visual_global = visual_tokens  # [B, embed_dim]
        else:
            visual_global = visual_tokens[:, 0]  # [B, embed_dim] - CLS token

        # 文本特征适配
        text_guidance = self.text_adapter(text_feat)  # [B, embed_dim]

        # 残差融合: 视觉特征 + 文本引导
        enhanced = visual_global + self.fusion_weight * text_guidance

        return enhanced  # [B, embed_dim]


def create_text_fusion_module(method: str = "attention", **kwargs) -> nn.Module:
    """
    工厂函数：创建文本融合模块

    Args:
        method: 融合方法 ("attention", "concat", "residual")
        **kwargs: 传递给具体模块的参数

    Returns:
        nn.Module: 对应的融合模块
    """
    if method == "attention":
        return CrossModalAttentionFusion(**kwargs)
    elif method == "concat":
        return TextConcatFusion(**kwargs)
    elif method == "residual":
        residual_kwargs = {k: v for k, v in kwargs.items() if k != 'input_dim'}
        return TextResidualFusion(**residual_kwargs)
    else:
        raise ValueError(f"Unknown text fusion method: {method}")


# 使用示例
if __name__ == "__main__":
    # 测试代码
    batch_size, embed_dim = 4, 512

    # 创建测试数据
    visual_tokens = torch.randn(batch_size, 129, embed_dim)  # [B, 129, 512]
    text_features = torch.randn(batch_size, embed_dim)       # [B, 512]

    # 测试注意力融合
    attention_fusion = CrossModalAttentionFusion(embed_dim=embed_dim)
    fused_attention = attention_fusion(visual_tokens, text_features)
    print(f"注意力融合输出: {fused_attention.shape}")  # [4, 512]

    # 测试拼接融合
    concat_fusion = TextConcatFusion(embed_dim=embed_dim)
    fused_concat = concat_fusion(visual_tokens, text_features)
    print(f"拼接融合输出: {fused_concat.shape}")  # [4, 512]

    # 测试残差融合
    residual_fusion = TextResidualFusion(embed_dim=embed_dim)
    fused_residual = residual_fusion(visual_tokens, text_features)
    print(f"残差融合输出: {fused_residual.shape}")  # [4, 512]
