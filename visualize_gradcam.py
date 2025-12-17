#!/usr/bin/env python
"""
Grad-CAM 热力图可视化工具

功能说明：
本工具用于为 ReID 模型生成 Grad-CAM 热力图可视化，展示模型在提取特征时关注图像的哪些区域。
支持单张图像、批量图像和多模态（RGB/NI/TI）可视化。

主要功能：
1. 加载训练好的 ReID 模型
2. 对指定图像生成 Grad-CAM 热力图
3. 支持多模态可视化（RGB、NIR、TIR）
4. 支持批量处理
5. 保存可视化结果

使用示例：
  # 单张图像可视化
  python visualize_gradcam.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --weight_path outputs/best_model.pth \
    --image_path data/RGBNT201/test/RGB/000123_cam1_0_01.jpg \
    --output_dir outputs/Grad_CAM

  # 批量可视化
  python visualize_gradcam.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --weight_path outputs/best_model.pth \
    --image_dir data/RGBNT201/test/RGB \
    --output_dir outputs/Grad_CAM \
    --num_images 10

  # 多模态可视化
  python visualize_gradcam.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --weight_path outputs/best_model.pth \
    --query_id 000123 \
    --dataset_root data/RGBNT201 \
    --output_dir outputs/Grad_CAM \
    --multimodal

作者：MambaPro团队
日期：2024
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
from typing import Tuple

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import cfg
from data import make_dataloader
from modeling import make_model
from grad_cam import GradCAM, find_target_layers


def build_transforms():
    """
    构建图像预处理变换管道
    
    功能说明：
    - 将图像调整为 ReID 标准尺寸（256×128）
    - 转换为张量并归一化（ImageNet 标准化参数）
    - 用于测试模式，不包含数据增强
    
    Returns:
        transforms.Compose: 图像变换管道
        
    注意：
        - 变换后的图像会进行归一化，值域不再是 [0, 255]
        - 如果需要可视化，需要使用原始图像（未归一化）
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 均值
        std=[0.229, 0.224, 0.225]    # ImageNet 标准差
    )
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)),  # 调整图像尺寸为 ReID 标准尺寸
        transforms.ToTensor(),          # 转换为张量 [0, 1]
        normalize,                      # 标准化（ImageNet 参数）
    ])
    return transform


def load_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    加载图像（同时返回原始图像和 PIL 图像）
    
    功能说明：
    - 使用 OpenCV 加载图像（用于可视化）
    - 使用 PIL 加载图像（用于预处理）
    - 确保两种格式的图像内容一致
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        tuple: (original_image, pil_image)
            - original_image: OpenCV 格式图像，形状为 [H, W, 3]（BGR，值域 [0, 255]）
            - pil_image: PIL 格式图像（RGB，值域 [0, 255]）
    
    Raises:
        FileNotFoundError: 如果图像文件不存在
        ValueError: 如果图像无法加载
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 使用 OpenCV 加载（BGR 格式）
    original_image_bgr = cv2.imread(image_path)
    if original_image_bgr is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 转换为 RGB 格式（用于可视化）
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    
    # 使用 PIL 加载（用于预处理）
    pil_image = Image.open(image_path).convert('RGB')
    
    return original_image_rgb, pil_image


def detect_camera_num_from_weights(weight_path: str) -> int:
    """
    从模型权重文件中自动检测相机数量
    
    功能说明：
    - 加载权重文件
    - 查找包含 'cv_embed' 的键（相机/视角嵌入层）
    - 从嵌入层的形状推断相机数量
    
    Args:
        weight_path (str): 模型权重文件路径
        
    Returns:
        int: 检测到的相机数量，默认为 4
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    for key in checkpoint:
        if 'BACKBONE.cv_embed' in key or 'cv_embed' in key:
            # 从 cv_embed 层的形状推断相机数量
            # 形状通常是 [camera_num, embed_dim]
            return checkpoint[key].shape[0]
    return 4  # 默认相机数量


def get_target_layer_name(model: nn.Module, model_type: str = 'auto') -> str:
    """
    根据模型类型自动获取目标层名称
    
    功能说明：
    - 自动检测模型类型（CLIP ViT、标准 ViT、ResNet 等）
    - 返回适合 Grad-CAM 的目标层路径
    - 目标层通常是最后一层卷积层或 Transformer 的最后一层
    
    Args:
        model (nn.Module): 模型对象
        model_type (str): 模型类型
            - 'auto': 自动检测
            - 'clip_vit': CLIP ViT 模型
            - 'vit': 标准 ViT 模型
            - 'resnet': ResNet 模型
    
    Returns:
        str: 目标层路径，如 'BACKBONE.image_encoder.transformer.resblocks.11'
    """
    if model_type == 'auto':
        # 自动检测模型类型
        if hasattr(model, 'BACKBONE'):
            backbone = model.BACKBONE
            if hasattr(backbone, 'image_encoder'):
                # CLIP 模型
                model_type = 'clip_vit'
            elif hasattr(backbone, 'base'):
                # 标准 ViT 或 ResNet
                if hasattr(backbone.base, 'blocks'):
                    model_type = 'vit'
                else:
                    model_type = 'resnet'
    
    # 根据模型类型返回目标层路径
    if model_type == 'clip_vit':
        # CLIP ViT-B-16: 最后一层 Transformer 块
        # 通常有 12 层，索引为 0-11，最后一层是 11
        return 'BACKBONE.image_encoder.transformer.resblocks.11'
    elif model_type == 'vit':
        # 标准 ViT: 最后一层 Transformer 块
        return 'BACKBONE.base.blocks.11'  # 假设 12 层 ViT
    elif model_type == 'resnet':
        # ResNet: 最后一层卷积层
        return 'BACKBONE.base.layer4'
    else:
        # 默认：尝试 CLIP ViT
        return 'BACKBONE.image_encoder.transformer.resblocks.11'


def visualize_single_image(
    model: nn.Module,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    modality: str,
    target_layer: str,
    output_path: str,
    alpha: float = 0.4
):
    """
    为单张图像生成 Grad-CAM 热力图可视化
    
    功能说明：
    1. 加载图像（原始图像用于可视化，预处理后的图像用于模型推理）
    2. 创建 Grad-CAM 对象
    3. 生成热力图
    4. 叠加到原始图像
    5. 保存结果
    
    Args:
        model (nn.Module): 训练好的 ReID 模型
        image_path (str): 图像文件路径
        transform (transforms.Compose): 图像预处理变换
        device (torch.device): 计算设备（CPU/GPU）
        modality (str): 模态类型（'RGB'、'NI'、'TI'）
        target_layer (str): 目标层路径
        output_path (str): 输出图像路径
        alpha (float): 热力图透明度，默认 0.4
    """
    # 加载图像
    original_image, pil_image = load_image(image_path)
    
    # 预处理图像
    img_tensor = transform(pil_image).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # 构建多模态输入字典
    input_dict = {
        'RGB': torch.zeros_like(img_tensor),
        'NI': torch.zeros_like(img_tensor),
        'TI': torch.zeros_like(img_tensor)
    }
    input_dict[modality] = img_tensor  # 激活当前模态
    
    # 创建 Grad-CAM 对象
    try:
        gradcam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
    except Exception as e:
        print(f"⚠️  创建 Grad-CAM 失败: {e}")
        print(f"   尝试查找可用的目标层...")
        # 尝试查找可用的层
        layers = find_target_layers(model, nn.Module)
        if layers:
            print(f"   找到 {len(layers)} 个可用层，使用第一个: {layers[0][0]}")
            target_layer = layers[0][0]
            gradcam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
        else:
            raise RuntimeError("无法找到合适的目标层，请手动指定 --target_layer 参数")
    
    # 生成热力图和叠加图像
    try:
        heatmap, overlay = gradcam.generate_gradcam(
            input_dict,
            original_image,
            target_class=None,  # 使用模型预测的类别
            alpha=alpha
        )
    except Exception as e:
        print(f"❌ 生成 Grad-CAM 失败: {e}")
        raise
    
    # 创建可视化图像（原始图像 + 热力图 + 叠加图像）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 热力图
    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 叠加图像
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (α=0.4)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'Grad-CAM Visualization - {modality} Modality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存可视化结果: {output_path}")


def visualize_multimodal(
    model: nn.Module,
    query_id: str,
    dataset_root: str,
    transform: transforms.Compose,
    device: torch.device,
    target_layer: str,
    output_path: str,
    alpha: float = 0.4
):
    """
    为多模态图像（RGB、NIR、TIR）生成 Grad-CAM 热力图可视化
    
    功能说明：
    - 加载同一行人的三种模态图像（RGB、NIR、TIR）
    - 为每种模态生成 Grad-CAM 热力图
    - 并排显示，便于对比分析
    
    Args:
        model (nn.Module): 训练好的 ReID 模型
        query_id (str): 查询人员ID（如 '000123'）
        dataset_root (str): 数据集根目录
        transform (transforms.Compose): 图像预处理变换
        device (torch.device): 计算设备
        target_layer (str): 目标层路径
        output_path (str): 输出图像路径
        alpha (float): 热力图透明度
    """
    modalities = ['RGB', 'NI', 'TI']
    modality_names = ['RGB', 'NIR', 'TIR']
    
    # 加载三种模态的图像
    images = {}
    for mod, mod_name in zip(modalities, modality_names):
        # 构建图像路径（假设文件命名格式：{query_id}_cam*_*.jpg）
        test_dir = os.path.join(dataset_root, 'test', mod)
        # 查找匹配的图像文件
        matching_files = [f for f in os.listdir(test_dir) if f.startswith(query_id) and f.endswith('.jpg')]
        if not matching_files:
            print(f"⚠️  未找到 {mod_name} 模态图像: {test_dir}")
            continue
        image_path = os.path.join(test_dir, matching_files[0])
        images[mod] = load_image(image_path)
    
    if not images:
        raise ValueError(f"未找到任何模态的图像，Query ID: {query_id}")
    
    # 创建可视化图像（3行×3列：原始、热力图、叠加）
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # 创建 Grad-CAM 对象
    try:
        gradcam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
    except Exception as e:
        print(f"⚠️  创建 Grad-CAM 失败: {e}")
        layers = find_target_layers(model, nn.Module)
        if layers:
            target_layer = layers[0][0]
            gradcam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
        else:
            raise RuntimeError("无法找到合适的目标层")
    
    # 为每种模态生成可视化
    for row, (mod, mod_name) in enumerate(zip(modalities, modality_names)):
        if mod not in images:
            continue
        
        original_image, pil_image = images[mod]
        
        # 预处理
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),
            'NI': torch.zeros_like(img_tensor),
            'TI': torch.zeros_like(img_tensor)
        }
        input_dict[mod] = img_tensor
        
        # 生成热力图
        try:
            heatmap, overlay = gradcam.generate_gradcam(
                input_dict, original_image, target_class=None, alpha=alpha
            )
        except Exception as e:
            print(f"⚠️  {mod_name} 模态生成失败: {e}")
            continue
        
        # 显示原始图像
        axes[row, 0].imshow(original_image)
        axes[row, 0].set_title(f'{mod_name} - Original', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        # 显示热力图
        im = axes[row, 1].imshow(heatmap, cmap='jet')
        axes[row, 1].set_title(f'{mod_name} - Heatmap', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        # 显示叠加图像
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f'{mod_name} - Overlay', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')
    
    plt.suptitle(f'Multi-modal Grad-CAM Visualization - Person ID: {query_id}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存多模态可视化结果: {output_path}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='Grad-CAM 热力图可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 单张图像可视化
  python visualize_gradcam.py \\
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \\
    --weight_path outputs/best_model.pth \\
    --image_path data/RGBNT201/test/RGB/000123_cam1_0_01.jpg \\
    --output_dir outputs/Grad_CAM

  # 批量可视化
  python visualize_gradcam.py \\
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \\
    --weight_path outputs/best_model.pth \\
    --image_dir data/RGBNT201/test/RGB \\
    --output_dir outputs/Grad_CAM \\
    --num_images 10

  # 多模态可视化
  python visualize_gradcam.py \\
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \\
    --weight_path outputs/best_model.pth \\
    --query_id 000123 \\
    --dataset_root data/RGBNT201 \\
    --output_dir outputs/Grad_CAM \\
    --multimodal
        """
    )
    
    # ========== 必需参数 ==========
    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='配置文件路径（YAML 格式）'
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='模型权重文件路径（.pth 文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径（可视化结果将保存在此目录）'
    )
    
    # ========== 输入参数（三选一）==========
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image_path',
        type=str,
        help='单张图像路径（用于单张图像可视化）'
    )
    input_group.add_argument(
        '--image_dir',
        type=str,
        help='图像目录路径（用于批量可视化）'
    )
    input_group.add_argument(
        '--query_id',
        type=str,
        help='查询人员ID（用于多模态可视化，如 "000123"）'
    )
    
    # ========== 可选参数 ==========
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='data/RGBNT201',
        help='数据集根目录（用于多模态可视化），默认: data/RGBNT201'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='RGB',
        choices=['RGB', 'NI', 'TI'],
        help='模态类型（单张图像或批量可视化时使用），默认: RGB'
    )
    parser.add_argument(
        '--multimodal',
        action='store_true',
        help='启用多模态模式（需要 --query_id），同时可视化 RGB、NIR、TIR 三种模态'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=10,
        help='批量可视化时的图像数量（仅当使用 --image_dir 时有效），默认: 10'
    )
    parser.add_argument(
        '--target_layer',
        type=str,
        default=None,
        help='目标层路径（用于 Grad-CAM），如 "BACKBONE.image_encoder.transformer.resblocks.11"。'
              '如果未指定，将自动检测模型类型并选择合适的目标层'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='热力图透明度（0.0-1.0），默认: 0.4。'
              '0.0 表示完全透明（只显示原始图像），1.0 表示完全不透明（热力图完全覆盖）'
    )
    parser.add_argument(
        '--list_layers',
        action='store_true',
        help='列出模型中所有可用的层（用于选择目标层），然后退出'
    )
    
    return parser.parse_args()


def main():
    """
    主函数：执行 Grad-CAM 热力图可视化
    """
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查路径
    if not os.path.exists(args.config_file):
        print(f"❌ 配置文件不存在: {args.config_file}")
        return
    
    if not os.path.exists(args.weight_path):
        print(f"❌ 权重文件不存在: {args.weight_path}")
        return
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms()
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    print("📦 加载模型配置和权重...")
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    camera_num = detect_camera_num_from_weights(args.weight_path)
    num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    
    model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model.load_param(args.weight_path)
    model.eval()
    print("✅ 模型加载完成")
    
    # 确定目标层
    if args.target_layer:
        target_layer = args.target_layer
    else:
        # 自动检测
        print("🔍 自动检测目标层...")
        target_layer = get_target_layer_name(model)
        print(f"   使用目标层: {target_layer}")
    
    # 如果只是列出层，则列出后退出
    if args.list_layers:
        print("\n📋 模型中所有可用的层:")
        layers = find_target_layers(model, nn.Module)
        for i, (name, layer) in enumerate(layers[:20]):  # 只显示前20个
            print(f"  [{i+1}] {name}: {type(layer).__name__}")
        if len(layers) > 20:
            print(f"  ... 还有 {len(layers) - 20} 个层未显示")
        print(f"\n💡 使用 --target_layer 参数指定目标层，例如:")
        if layers:
            print(f"   --target_layer {layers[0][0]}")
        return
    
    # 根据输入类型执行不同的可视化
    if args.image_path:
        # ========== 单张图像可视化 ==========
        print(f"\n🖼️  单张图像可视化: {args.image_path}")
        output_path = os.path.join(
            args.output_dir,
            f"gradcam_{os.path.basename(args.image_path).replace('.jpg', '')}_{args.modality}.png"
        )
        visualize_single_image(
            model, args.image_path, transform, device,
            args.modality, target_layer, output_path, args.alpha
        )
        
    elif args.image_dir:
        # ========== 批量可视化 ==========
        print(f"\n🖼️  批量可视化: {args.image_dir}")
        image_files = [f for f in os.listdir(args.image_dir) if f.endswith('.jpg')]
        image_files = image_files[:args.num_images]
        
        for i, image_file in enumerate(tqdm(image_files, desc="处理图像")):
            image_path = os.path.join(args.image_dir, image_file)
            output_path = os.path.join(
                args.output_dir,
                f"gradcam_{image_file.replace('.jpg', '')}_{args.modality}.png"
            )
            try:
                visualize_single_image(
                    model, image_path, transform, device,
                    args.modality, target_layer, output_path, args.alpha
                )
            except Exception as e:
                print(f"⚠️  处理 {image_file} 失败: {e}")
                continue
        
        print(f"\n✅ 批量可视化完成，共处理 {len(image_files)} 张图像")
        
    elif args.query_id:
        # ========== 多模态可视化 ==========
        if args.multimodal:
            print(f"\n🖼️  多模态可视化: Query ID = {args.query_id}")
            output_path = os.path.join(
                args.output_dir,
                f"gradcam_multimodal_{args.query_id}.png"
            )
            visualize_multimodal(
                model, args.query_id, args.dataset_root, transform,
                device, target_layer, output_path, args.alpha
            )
        else:
            # 单模态（使用 RGB）
            print(f"\n🖼️  单模态可视化: Query ID = {args.query_id}, Modality = {args.modality}")
            # 构建图像路径
            test_dir = os.path.join(args.dataset_root, 'test', args.modality)
            matching_files = [
                f for f in os.listdir(test_dir) 
                if f.startswith(args.query_id) and f.endswith('.jpg')
            ]
            if not matching_files:
                print(f"❌ 未找到图像: {test_dir}")
                return
            image_path = os.path.join(test_dir, matching_files[0])
            output_path = os.path.join(
                args.output_dir,
                f"gradcam_{args.query_id}_{args.modality}.png"
            )
            visualize_single_image(
                model, image_path, transform, device,
                args.modality, target_layer, output_path, args.alpha
            )
    
    print(f"\n🎉 可视化完成！")
    print(f"📁 结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
