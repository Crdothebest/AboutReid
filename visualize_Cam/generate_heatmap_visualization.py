#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成多模态热力图可视化脚本

功能说明：
生成类似 heatmap_000274.png 效果的多模态热力图可视化
- 3行×2列布局：RGB、NIR、TIR 三种模态
- 左列：原始图像
- 右列：叠加了热力图的图像

使用方法：
python generate_heatmap_visualization.py \
    --weight_path path/to/model.pth \
    --config_file configs/RGBNT201/MambaPro.yml \
    --query_id 000274 \
    --dataset_root /path/to/RGBNT201 \
    --output_path heatmap_000274.png

作者：MambaPro团队
日期：2024
"""

import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)

from config import cfg
from modeling import make_model
from visualize_Cam.visualize_gradcam import (
    build_transforms,
    load_image,
    detect_camera_num_from_weights,
    get_target_layer_name
)
from visualize_Cam.grad_cam import GradCAM, EigenCAM


def mamba_reid_reshape_transform(tensor, height=16, width=8):
    """
    专门处理 Mamba/ViT ReID 模型的 1D output -> 2D feature map
    根据解决方案文档实现，解决"四角对称光斑"问题
    
    输入形状: [Batch, Sequence_Length, Channels] (e.g., [1, 129, 768])
    输出形状: [Batch, Channels, Height, Width] (e.g., [1, 768, 16, 8])
    """
    target_h, target_w = height, width
    
    # 剥离 CLS Token
    if tensor.shape[1] == target_h * target_w + 1:
        tensor = tensor[:, 1:, :]
    
    # 维度置换 [B, L, C] -> [B, C, L]
    result = tensor.transpose(1, 2)
    
    # 强制 Reshape
    result = result.reshape(tensor.size(0), result.size(1), target_h, target_w)
    
    return result


def generate_multimodal_heatmap(
    model: nn.Module,
    query_id: str,
    dataset_root: str,
    transform,
    device: torch.device,
    target_layer: str,
    output_path: str,
    alpha: float = 0.4,
    method: str = 'gradcam'
):
    """
    生成多模态热力图可视化（3行×2列布局）
    
    功能说明：
    - 加载同一行人的三种模态图像（RGB、NIR、TIR）
    - 为每种模态生成热力图（Grad-CAM 或 EigenCAM）
    - 创建 3行×2列 布局：左列原始图像，右列叠加热力图
    
    Args:
        model (nn.Module): 训练好的 ReID 模型
        query_id (str): 查询人员ID（如 '000274'）
        dataset_root (str): 数据集根目录
        transform: 图像预处理变换
        device (torch.device): 计算设备
        target_layer (str): 目标层路径
        output_path (str): 输出图像路径
        alpha (float): 热力图透明度，默认 0.4
        method (str): 热力图方法，'gradcam' 或 'eigencam'，默认 'gradcam'
    """
    modalities = ['RGB', 'NI', 'TI']
    modality_names = ['RGB', 'NIR', 'TIR']
    
    # 加载三种模态的图像
    images = {}
    for mod, mod_name in zip(modalities, modality_names):
        # 构建图像路径
        test_dir = os.path.join(dataset_root, 'test', mod)
        if not os.path.exists(test_dir):
            print(f"⚠️  目录不存在: {test_dir}")
            continue
        
        # 查找匹配的图像文件
        matching_files = [
            f for f in os.listdir(test_dir) 
            if f.startswith(query_id) and f.endswith('.jpg')
        ]
        if not matching_files:
            print(f"⚠️  未找到 {mod_name} 模态图像: {test_dir}")
            continue
        
        image_path = os.path.join(test_dir, matching_files[0])
        try:
            images[mod] = load_image(image_path)
        except Exception as e:
            print(f"⚠️  加载 {mod_name} 图像失败: {e}")
            continue
    
    if not images:
        raise ValueError(f"未找到任何模态的图像，Query ID: {query_id}")
    
    # 创建可视化图像（3行×2列：原始、叠加）
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
    # 创建 CAM 对象（GradCAM 或 EigenCAM）
    cam = None
    try:
        if method.lower() == 'eigencam':
            print(f"🔧 使用 EigenCAM 方法（目标层: {target_layer}）")
            cam = EigenCAM(
                model, 
                target_layer=target_layer, 
                use_cuda=device.type == 'cuda',
                reshape_transform=mamba_reid_reshape_transform
            )
        else:
            print(f"🔧 使用 GradCAM 方法（目标层: {target_layer}）")
            cam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
    except Exception as e:
        print(f"⚠️  创建 CAM 对象失败: {e}")
        from visualize_Cam.grad_cam import find_target_layers
        layers = find_target_layers(model, nn.Module)
        if layers:
            print(f"   找到 {len(layers)} 个可用层，使用第一个: {layers[0][0]}")
            target_layer = layers[0][0]
            if method.lower() == 'eigencam':
                cam = EigenCAM(
                    model, 
                    target_layer=target_layer, 
                    use_cuda=device.type == 'cuda',
                    reshape_transform=mamba_reid_reshape_transform
                )
            else:
                cam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
        else:
            raise RuntimeError("无法找到合适的目标层，请手动指定 --target_layer 参数")
    
    # 🔥 第一步：收集所有模态的原始热力图，用于统一归一化
    raw_heatmaps = {}  # 存储原始热力图（归一化前）
    original_images_dict = {}  # 存储原始图像
    
    for mod, mod_name in zip(modalities, modality_names):
        if mod not in images:
            continue
        
        original_image, pil_image = images[mod]
        original_images_dict[mod] = original_image
        
        # 预处理
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),
            'NI': torch.zeros_like(img_tensor),
            'TI': torch.zeros_like(img_tensor)
        }
        input_dict[mod] = img_tensor
        
        # 准备标签
        cam_label = torch.tensor([0]).to(device)
        view_label = torch.tensor([0]).to(device)
        
        # 🔥 A. 调整归一化与缩放顺序：先计算每个模态自身的显著性，不进行置信度缩放
        # 不要在全局归一化之前进行置信度缩放，应该先计算每个模态自身的显著性
        # 最后再通过透明度或全局亮度来体现差异
        try:
            if method.lower() == 'eigencam':
                raw_heatmap = cam.generate_cam(
                    input_dict,
                    cam_label=cam_label,
                    view_label=view_label
                )
                # 不在这里应用置信度缩放，保留原始激活值
                raw_heatmaps[mod] = raw_heatmap
            else:
                # GradCAM 也收集原始热力图
                heatmap, _ = cam.generate_gradcam(
                    input_dict, original_image, target_class=None, alpha=0.0,
                    cam_label=cam_label, view_label=view_label
                )
                # 不在这里应用置信度缩放，保留原始激活值
                raw_heatmaps[mod] = heatmap
        except Exception as e:
            print(f"  ⚠️  生成 {mod_name} 模态热力图失败: {e}")
            continue
    
    # 🔥 第二步：统一归一化（使用全模态的最大激活值）
    # 这是唯一一次归一化，确保模态间的对比度得以保留
    if raw_heatmaps:
        global_max = max(h.max() for h in raw_heatmaps.values())
        global_min = min(h.min() for h in raw_heatmaps.values())
        print(f"  📊 全局归一化范围: [{global_min:.4f}, {global_max:.4f}]")
        print(f"  📊 RGB最大值: {raw_heatmaps.get('RGB', np.array([0])).max():.4f}, "
              f"NIR最大值: {raw_heatmaps.get('NI', np.array([0])).max():.4f}, "
              f"TIR最大值: {raw_heatmaps.get('TI', np.array([0])).max():.4f}")
    else:
        global_max = 1.0
        global_min = 0.0
    
    # 🔥 第三步：为每种模态生成可视化
    for row, (mod, mod_name) in enumerate(zip(modalities, modality_names)):
        if mod not in images or mod not in raw_heatmaps:
            # 如果缺少某个模态，显示空白
            axes[row, 0].axis('off')
            axes[row, 1].axis('off')
            continue
        
        original_image = original_images_dict[mod]
        heatmap = raw_heatmaps[mod].copy()
        
        # 🔥 EigenCAM 美化处理
        if method.lower() == 'eigencam':
            # 🔍 调试：打印原始热力图的统计信息
            print(f"  🔍 {mod_name} 原始热力图: min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}, non_zero={(heatmap > 0).sum()}/{heatmap.size}")
            
            # 1. 🔥 先归一化：将当前模态缩放到 [0, 1]
            # 使用当前模态的最大值和最小值，计算每个模态自身的显著性
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
            else:
                heatmap = np.zeros_like(heatmap)
            heatmap = np.clip(heatmap, 0, 1)
            
            # 🔍 调试：打印归一化后的统计信息
            print(f"  🔍 {mod_name} 归一化后: min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}, >0.5={(heatmap > 0.5).sum()}, >0.8={(heatmap > 0.8).sum()}")
            
            # 2. 🔥 增强对比度：使用大于 1 的 Gamma 值，让红色区域更突出
            # 目的：让热力中心呈现深红色，而不是黄色
            # 对于已归一化的值 [0, 1]，使用 gamma > 1 可以压缩中等权重，让最强区域达到 1.0（红色）
            # 例如：0.8^1.5 = 0.72（变暗），但 1.0^1.5 = 1.0（保持红色）
            if mod == 'RGB':
                gamma = 1.5  # RGB 使用 1.5，增强对比度
            elif mod == 'NI':
                gamma = 1.5  # NIR 使用 1.5，增强对比度
            else:  # TI
                gamma = 1.4  # TIR 使用 1.4，增强对比度
            heatmap = np.power(heatmap, gamma)
            
            # 3. 🔥 强力阈值过滤：提高阈值，更彻底地过滤背景噪声
            # 目的：过滤掉那些极低分、弥散在背景中的颜色，让红色更聚焦于人体
            threshold_ratio = 0.15  # 提高到 15% 阈值，更彻底地过滤背景
            
            # 使用当前模态的最大值（归一化后为 1.0）作为阈值基准
            threshold_base = heatmap.max() * threshold_ratio
            heatmap[heatmap < threshold_base] = 0
            
            # 🔍 调试：检查阈值过滤后的值
            print(f"  🔍 {mod_name} 阈值过滤后: min={heatmap.min():.6f}, max={heatmap.max():.6f}, >0.5={(heatmap > 0.5).sum()}")
            
            # 3. 🔥 裁剪检测边界：暂时完全移除边缘裁剪
            # 去除边缘填充（Padding）和拼图效应造成的边缘亮斑
            # 注意：暂时完全移除裁剪，避免去除有效区域
            # h, w = heatmap.shape
            # border_h = int(h * 0.02)  # 2% 边界
            # border_w = int(w * 0.02)
            # heatmap_edge_cleaned = heatmap.copy()
            # heatmap_edge_cleaned[:border_h, :] = 0  # 顶部
            # heatmap_edge_cleaned[-border_h:, :] = 0  # 底部
            # heatmap_edge_cleaned[:, :border_w] = 0  # 左侧
            # heatmap_edge_cleaned[:, -border_w:] = 0  # 右侧
            heatmap_edge_cleaned = heatmap.copy()  # 暂时不裁剪
            
            # 4. 放大：使用双三次插值（Bicubic）提升平滑度
            heatmap_resized = cv2.resize(
                heatmap_edge_cleaned,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
            
            # 5. 高斯模糊：消除生硬边缘，实现"云雾感"效果
            # 使用更大的核 (45, 45) 进行初步平滑
            heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (45, 45), 0)
            
            # 6. 重新归一化（确保值域在 [0, 1]）
            heatmap_max = heatmap_blurred.max()
            heatmap_min = heatmap_blurred.min()
            
            # 🔍 调试：检查模糊后的值
            print(f"  🔍 {mod_name} 模糊后: min={heatmap_min:.6f}, max={heatmap_max:.6f}, >0.5={(heatmap_blurred > 0.5).sum()}")
            
            if heatmap_max > heatmap_min:
                heatmap_blurred = (heatmap_blurred - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
            elif heatmap_max > 0:
                heatmap_blurred = heatmap_blurred / heatmap_max
            else:
                heatmap_blurred = np.zeros_like(heatmap_blurred)
            
            # 🔍 调试：检查重新归一化后的值
            print(f"  🔍 {mod_name} 重新归一化后: min={heatmap_blurred.min():.6f}, max={heatmap_blurred.max():.6f}, >0.5={(heatmap_blurred > 0.5).sum()}")
            
            # 7. 全局亮度对比：在颜色映射前引入全局对比系数
            if global_max > 0:
                if mod == 'TI':
                    global_contrast = 1.0  # TIR 保持 100%
                else:
                    mod_max = raw_heatmaps[mod].max()
                    if mod_max >= global_max * 0.95:
                        global_contrast = 0.65  # 降低到 65%
                    else:
                        global_contrast = min(0.75, 0.5 + (mod_max / global_max) * 0.25)
                heatmap_blurred = heatmap_blurred * global_contrast
                print(f"  🔍 {mod_name} 全局对比系数: {global_contrast:.3f}")
            else:
                global_contrast = 1.0
            
            # 8. 重新归一化：应用全局对比系数后，重新归一化让最大值回到 1.0
            # 这样能确保有红色区域（值接近 1.0）
            heatmap_max = heatmap_blurred.max()
            heatmap_min = heatmap_blurred.min()
            if heatmap_max > heatmap_min:
                heatmap_normalized = (heatmap_blurred - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
            elif heatmap_max > 0:
                heatmap_normalized = heatmap_blurred / heatmap_max
            else:
                heatmap_normalized = np.zeros_like(heatmap_blurred)
            heatmap_normalized = np.clip(heatmap_normalized, 0, 1)
            
            # 🔍 调试：打印颜色映射前的统计信息
            print(f"  🔍 {mod_name} 颜色映射前: min={heatmap_normalized.min():.6f}, max={heatmap_normalized.max():.6f}, mean={heatmap_normalized.mean():.6f}, >0.5={(heatmap_normalized > 0.5).sum()}, >0.8={(heatmap_normalized > 0.8).sum()}")
            
            # 10. 论文级叠加：使用改进的 overlay_heatmap 方法
            # overlay_heatmap 内部会再次进行平滑处理和颜色映射，产生光晕效果
            # 对于 TIR 模态，保持原图的冷蓝色调，使用 JET 颜色映射
            if mod == 'TI':
                # TIR 保持原图颜色，使用 JET 颜色映射（蓝-绿-黄-红）
                # 背景的冷蓝色调会通过 overlay_heatmap 的动态 mask 保持
                overlay = cam.overlay_heatmap(original_image, heatmap_normalized, alpha=0.6, colormap=cv2.COLORMAP_JET)
            else:
                # RGB/NIR 使用标准叠加，alpha 设为 0.5 防止原图色彩稀释
                overlay = cam.overlay_heatmap(original_image, heatmap_normalized, alpha=0.5, colormap=cv2.COLORMAP_JET)
        else:
            # GradCAM 标准处理（也使用相同的优化逻辑）
            heatmap = raw_heatmaps[mod].copy()
            
            # 1. 先归一化：将当前模态缩放到 [0, 1]
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
            else:
                heatmap = np.zeros_like(heatmap)
            heatmap = np.clip(heatmap, 0, 1)
            
            # 2. 增强对比度：使用大于 1 的 Gamma 值，让红色区域更突出
            # 对于已归一化的值 [0, 1]，使用 gamma > 1 可以压缩中等权重，让最强区域达到 1.0（红色）
            if mod == 'RGB':
                gamma = 1.5  # RGB 使用 1.5，增强对比度
            elif mod == 'NI':
                gamma = 1.5  # NIR 使用 1.5，增强对比度
            else:  # TI
                gamma = 1.4  # TIR 使用 1.4，增强对比度
            heatmap = np.power(heatmap, gamma)
            
            # 3. 强力阈值过滤：提高阈值，更彻底地过滤背景噪声
            threshold_ratio = 0.15  # 提高到 15% 阈值，更彻底地过滤背景
            threshold_base = heatmap.max() * threshold_ratio
            heatmap[heatmap < threshold_base] = 0
            
            # 4. 裁剪检测边界（2% 边界，减少裁剪）
            h, w = heatmap.shape
            border_h = int(h * 0.02)
            border_w = int(w * 0.02)
            heatmap_edge_cleaned = heatmap.copy()
            heatmap_edge_cleaned[:border_h, :] = 0
            heatmap_edge_cleaned[-border_h:, :] = 0
            heatmap_edge_cleaned[:, :border_w] = 0
            heatmap_edge_cleaned[:, -border_w:] = 0
            
            # 5. 放大和模糊（增大核大小到 45x45）
            heatmap_resized = cv2.resize(
                heatmap_edge_cleaned,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
            heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (45, 45), 0)
            
            # 6. 重新归一化（用于颜色映射）
            heatmap_max = heatmap_blurred.max()
            if heatmap_max > 0:
                heatmap_blurred = heatmap_blurred / heatmap_max
            else:
                heatmap_blurred = np.zeros_like(heatmap_blurred)
            
            # 7. 恢复全局亮度对比：在颜色映射前引入全局对比系数
            if global_max > 0:
                if mod == 'TI':
                    global_contrast = 1.0
                else:
                    mod_max = raw_heatmaps[mod].max()
                    global_contrast = min(0.7, mod_max / global_max)
                heatmap_blurred = heatmap_blurred * global_contrast
            else:
                global_contrast = 1.0
            
            # 8. 颜色映射（不反转）
            heatmap_normalized = np.clip(heatmap_blurred, 0, 1)
            heatmap_uint8 = np.uint8(255 * heatmap_normalized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # 9. 论文级叠加：使用改进的 overlay_heatmap 方法
            # 对于 TIR 模态，保持原图的冷蓝色调，使用 JET 颜色映射
            if mod == 'TI':
                # TIR 保持原图颜色，使用 JET 颜色映射（蓝-绿-黄-红）
                # 背景的冷蓝色调会通过 overlay_heatmap 的动态 mask 保持
                overlay = cam.overlay_heatmap(original_image, heatmap_normalized, alpha=0.6, colormap=cv2.COLORMAP_JET)
            else:
                # RGB/NIR 使用标准叠加，alpha 设为 0.5 防止原图色彩稀释
                overlay = cam.overlay_heatmap(original_image, heatmap_normalized, alpha=0.5, colormap=cv2.COLORMAP_JET)
        
        # 显示原始图像（左列）
        axes[row, 0].imshow(original_image)
        axes[row, 0].set_title(f'{mod_name}', fontsize=12, fontweight='bold', pad=10)
        axes[row, 0].axis('off')
        
        # 显示叠加图像（右列）- 热力图叠加在原始图像上
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f'{mod_name}', fontsize=12, fontweight='bold', pad=10)
        axes[row, 1].axis('off')
        
    # 设置布局
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 已保存多模态热力图可视化: {output_path}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='生成多模态热力图可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python generate_heatmap_visualization.py \\
    --weight_path outputs/best_model.pth \\
    --config_file configs/RGBNT201/MambaPro.yml \\
    --query_id 000274 \\
    --dataset_root /path/to/RGBNT201 \\
    --output_path heatmap_000274.png

  # 指定目标层和透明度
  python generate_heatmap_visualization.py \\
    --weight_path outputs/best_model.pth \\
    --config_file configs/RGBNT201/MambaPro.yml \\
    --query_id 000274 \\
    --dataset_root /path/to/RGBNT201 \\
    --output_path heatmap_000274.png \\
    --target_layer BACKBONE.base.transformer.resblocks.11 \\
    --alpha 0.4
        """
    )
    
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='模型权重文件路径（.pth 文件）'
    )
    
    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='配置文件路径（YAML 格式）'
    )
    
    parser.add_argument(
        '--query_id',
        type=str,
        required=True,
        help='查询人员ID（如 "000274"）'
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201',
        help='数据集根目录，默认: /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='输出图像路径，如果不指定则自动保存到 outputs/Grad_CAM/{weight_name}/heatmap_{query_id}.png'
    )
    
    parser.add_argument(
        '--target_layer',
        type=str,
        default=None,
        help='目标层路径（用于 Grad-CAM），如果不指定则自动检测'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='热力图透明度（0.0-1.0），默认: 0.4'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        choices=['gradcam', 'eigencam'],
        help='热力图生成方法：gradcam（Grad-CAM）或 eigencam（EigenCAM），默认: gradcam'
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    print("="*60)
    print("生成多模态热力图可视化")
    print("="*60)
    
    # 检查路径
    if not os.path.exists(args.weight_path):
        print(f"❌ 权重文件不存在: {args.weight_path}")
        return
    
    if not os.path.exists(args.config_file):
        print(f"❌ 配置文件不存在: {args.config_file}")
        return
    
    if not os.path.exists(args.dataset_root):
        print(f"❌ 数据集路径不存在: {args.dataset_root}")
        return
    
    # 先保存原始权重路径（可能在检测 TorchScript 后会被修改）
    original_weight_path = args.weight_path
    
    # 设置输出路径（默认保存到 outputs/Grad_CAM/ 目录下）
    # 注意：这里先不设置，等确定最终使用的权重文件后再设置
    output_path_auto = None
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms()
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    print("\n📦 加载模型配置和权重...")
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # 检查是否是 TorchScript 模型
    is_torchscript = False
    try:
        # 先尝试加载看看是否是 TorchScript
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            checkpoint = torch.load(args.weight_path, map_location='cpu')
            is_torchscript = isinstance(checkpoint, torch.jit.ScriptModule) or isinstance(checkpoint, torch.jit.ScriptFunction)
    except Exception as e:
        print(f"⚠️  检查权重文件格式时出错: {e}")
        is_torchscript = False
    
    if is_torchscript:
        print("⚠️  检测到 TorchScript 模型（.pt 文件）")
        print("   TorchScript 模型不支持 Grad-CAM 热力图生成")
        print("   原因：TorchScript 模型是编译后的模型，无法访问内部层结构")
        print("\n🔍 正在查找可用的 PyTorch 权重文件（.pth）...")
        
        # 自动查找可用的 .pth 文件
        possible_paths = [
            '/home/zhanghaoyang/Desktop/yzy/MambaPro/outputs/baseline/RGBNT201/77.0mAP_20251218_164722.pth',
            '/home/zhanghaoyang/Desktop/yzy/MambaPro/outputs/baseline/RGBNT201',
            '/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs',
            os.path.dirname(args.weight_path),
        ]
        
        found_pth = None
        for base_path in possible_paths:
            if not os.path.exists(base_path):
                continue
            if os.path.isfile(base_path) and base_path.endswith('.pth'):
                found_pth = base_path
                break
            elif os.path.isdir(base_path):
                # 在目录中查找 .pth 文件
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith('.pth') and 'RGBNT201' in root:
                            found_pth = os.path.join(root, file)
                            break
                    if found_pth:
                        break
            if found_pth:
                break
        
        if found_pth and os.path.exists(found_pth):
            print(f"✅ 找到可用的权重文件: {found_pth}")
            print(f"   将使用此文件替代 TorchScript 模型")
            args.weight_path = found_pth
            # 重新检查（这次应该是普通权重文件）
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    checkpoint = torch.load(args.weight_path, map_location='cpu')
                    is_torchscript = isinstance(checkpoint, torch.jit.ScriptModule) or isinstance(checkpoint, torch.jit.ScriptFunction)
                if is_torchscript:
                    print("❌ 错误：找到的文件仍然是 TorchScript 模型")
                    print("   请手动指定一个 .pth 格式的权重文件")
                    return
            except:
                pass
        else:
            print("❌ 未找到可用的 .pth 权重文件")
            print("\n💡 解决方案：")
            print("   1. 使用训练好的 PyTorch 权重文件（.pth 文件）")
            print("   2. 或者使用 torch.save() 保存的完整模型权重")
            print("\n   可用的权重文件示例：")
            print("   - /home/zhanghaoyang/Desktop/yzy/MambaPro/outputs/baseline/RGBNT201/77.0mAP_20251218_164722.pth")
            print("\n   示例命令：")
            print("   python generate_heatmap_visualization.py \\")
            print("       --weight_path /home/zhanghaoyang/Desktop/yzy/MambaPro/outputs/baseline/RGBNT201/77.0mAP_20251218_164722.pth \\")
            print("       --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \\")
            print("       --query_id 000274 \\")
            print("       --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \\")
            print("       --output_path heatmap_000274.png")
            return
    
    # 加载模型（无论是原始权重还是替代权重）
    camera_num = detect_camera_num_from_weights(args.weight_path)
    num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    
    model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model.load_param(args.weight_path)
    model.eval()
    print("✅ 模型加载完成")
    
    # 现在确定最终使用的权重文件，设置输出路径
    if args.output_path is None:
        # 从权重文件的上一级目录名提取模型名称（用于创建子目录）
        # 例如：/path/to/77.76_4x4+16x16_20251217_160700/MambaProbest.pth
        # 提取：77.76_4x4+16x16_20251217_160700
        weight_dir = os.path.dirname(os.path.abspath(args.weight_path))
        weight_name = os.path.basename(weight_dir)  # 获取上一级目录名
        
        # 根据方法选择输出目录
        if args.method.lower() == 'eigencam':
            output_subdir = 'EigenCAM'
            filename_prefix = 'eigencam'
        else:
            output_subdir = 'Grad_CAM'
            filename_prefix = 'heatmap'
        
        # 创建输出目录结构：outputs/{subdir}/{weight_name}/
        output_base_dir = os.path.join(script_dir, 'outputs', output_subdir, weight_name)
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 输出文件路径
        args.output_path = os.path.join(output_base_dir, f"{filename_prefix}_{args.query_id}.png")
    else:
        # 如果用户指定了输出路径，确保目录存在
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 输出路径: {args.output_path}")
    
    # 确定目标层
    if args.target_layer:
        target_layer = args.target_layer
    else:
        print("🔍 自动检测目标层...")
        if args.method.lower() == 'eigencam':
            # EigenCAM 推荐使用 ln_post 层
            if hasattr(model, 'BACKBONE') and hasattr(model.BACKBONE, 'base'):
                if hasattr(model.BACKBONE.base, 'ln_post'):
                    target_layer = 'BACKBONE.base.ln_post'
                    print(f"   使用 EigenCAM 推荐目标层: {target_layer}")
                else:
                    target_layer = get_target_layer_name(model)
                    print(f"   使用自动检测的目标层: {target_layer}")
            else:
                target_layer = get_target_layer_name(model)
                print(f"   使用自动检测的目标层: {target_layer}")
        else:
            target_layer = get_target_layer_name(model)
            print(f"   使用目标层: {target_layer}")
    
    # 生成热力图
    print(f"\n🖼️  生成热力图可视化: Query ID = {args.query_id}, 方法 = {args.method.upper()}")
    try:
        generate_multimodal_heatmap(
            model, args.query_id, args.dataset_root,
            transform, device, target_layer,
            args.output_path, args.alpha, args.method
        )
        print(f"\n🎉 完成！")
        print(f"📁 结果保存在: {args.output_path}")
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
