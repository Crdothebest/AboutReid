# Category: vis_utils (可视化工具)
# Description: 提供热力图 (CAM)、t-SNE 降维、检索结果展示等模型可视化功能

#!/usr/bin/env python
"""
Grad-CAM 工具函数：用于处理 Transformer/ViT 模型的 reshape 转换

根据解决方案文档，MambaPro/ViT 模型输出的特征图是序列格式 [B, N, D]，
需要正确转换为 2D 空间格式 [B, C, H, W] 才能生成热力图。
"""

import torch
import numpy as np


def mamba_reshape_transform(tensor, height=16, width=8):
    """
    Mamba/ViT 专用的 Reshape 转换函数
    
    功能说明：
    - 将 Transformer 输出的序列格式 [B, N, D] 转换为 2D 空间格式 [B, C, H, W]
    - 处理 CLS token：去掉第一个 token（CLS token），只保留 patch tokens
    - 将 patch tokens reshape 为 2D 网格格式
    
    参数：
    - tensor: Transformer 层输出，形状为 [B, N, D]
        - B: batch size
        - N: sequence length（通常是 129 = 1 CLS token + 128 patch tokens）
        - D: feature dimension（通常是 768 或 512）
    - height: 目标高度（patch 网格的高度，默认 16）
    - width: 目标宽度（patch 网格的宽度，默认 8）
    
    返回：
    - result: 转换后的特征图，形状为 [B, C, H, W]
        - B: batch size
        - C: feature dimension (D)
        - H: height (16)
        - W: width (8)
    
    示例：
        >>> # Transformer 输出: [1, 129, 768]
        >>> output = torch.randn(1, 129, 768)
        >>> reshaped = mamba_reshape_transform(output, height=16, width=8)
        >>> print(reshaped.shape)  # [1, 768, 16, 8]
    """
    # tensor 的形状通常是 [Batch, Sequence_Length, Channels]
    # 例如 [1, 129, 768] (1个 CLS token + 128个 image tokens)
    
    B, N, D = tensor.shape
    
    # 1. 把 Channel 维度换到前面: [B, N, D] -> [B, D, N]
    result = tensor.transpose(1, 2)  # [B, D, N]
    
    # 2. 处理 CLS Token (如果有的话)
    # 很多 ReID 模型（如 TransReID, MambaPro）第一个 token 是分类用的，没有空间位置
    # 我们需要把它剥离掉，只保留后面的图像部分
    # 如果 sequence length 是 129，那就去掉第1个；如果是 128，就不用去掉。
    expected_patch_count = height * width  # 16 * 8 = 128
    
    if N != expected_patch_count:
        # 假设有一个 CLS token，去掉它
        # 从 [B, D, N] 中取 [B, D, 1:] 去掉第一个 token
        result = result[:, :, 1:]  # [B, D, N-1] = [B, D, 128]
    
    # 3. 强制重塑为 2D 图像特征图
    # [B, D, L] -> [B, D, H, W]
    # 注意：ReID 图片通常是长方形，所以 H 和 W 不一样！
    result = result.reshape(B, D, height, width)  # [B, D, 16, 8]
    
    return result


def get_patch_grid_size(image_size=(256, 128), patch_size=16):
    """
    根据输入图像尺寸和 patch size 计算网格大小
    
    参数：
    - image_size: 输入图像尺寸 (height, width)，默认 (256, 128)
    - patch_size: patch 大小，默认 16
    
    返回：
    - (height, width): patch 网格尺寸 (h_patches, w_patches)
    
    示例：
        >>> h, w = get_patch_grid_size((256, 128), 16)
        >>> print(h, w)  # 16, 8
    """
    img_h, img_w = image_size
    h_patches = img_h // patch_size  # 256 // 16 = 16
    w_patches = img_w // patch_size  # 128 // 16 = 8
    return h_patches, w_patches
