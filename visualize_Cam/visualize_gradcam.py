#!/usr/bin/env python
"""
Grad-CAM 热力图可视化工具
支持单张图像、批量图像和多模态（RGB/NI/TI）可视化
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
from typing import Tuple

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)

from config import cfg
from data import make_dataloader
from modeling import make_model
from visualize_Cam.grad_cam import GradCAM, find_target_layers


def build_transforms():
    """构建图像预处理变换管道"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def load_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    """加载图像（同时返回原始图像和 PIL 图像）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    original_image_bgr = cv2.imread(image_path)
    if original_image_bgr is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.open(image_path).convert('RGB')
    
    return original_image_rgb, pil_image


def detect_camera_num_from_weights(weight_path: str) -> int:
    """从模型权重文件中自动检测相机数量"""
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        if isinstance(checkpoint, torch.jit.ScriptModule) or isinstance(checkpoint, torch.jit.ScriptFunction):
            print(f"⚠️  检测到 TorchScript 模型，无法自动检测相机数量，使用默认值 4")
            return 4
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            for key in state_dict.keys():
                if 'BACKBONE.cv_embed' in key or 'cv_embed' in key:
                    cv_embed_shape = state_dict[key].shape
                    if len(cv_embed_shape) >= 1:
                        print(f"✅ 从权重文件检测到相机数量: {cv_embed_shape[0]}")
                        return cv_embed_shape[0]
        
        print(f"⚠️  无法从权重文件检测相机数量，使用默认值 4")
        return 4
    except Exception as e:
        print(f"⚠️  加载权重文件时出错: {e}，使用默认相机数量 4")
        return 4


def get_target_layer_name(model: nn.Module, model_type: str = 'auto') -> str:
    """根据模型类型自动获取目标层名称"""
    if model_type == 'auto':
        if hasattr(model, 'BACKBONE'):
            backbone = model.BACKBONE
            if hasattr(backbone, 'image_encoder'):
                model_type = 'clip_vit'
            elif hasattr(backbone, 'base'):
                if hasattr(backbone.base, 'transformer'):
                    if hasattr(backbone.base.transformer, 'resblocks'):
                        model_type = 'vit_transformer'
                    elif hasattr(backbone.base.transformer, 'blocks'):
                        model_type = 'vit'
                    else:
                        model_type = 'vit'
                elif hasattr(backbone.base, 'blocks'):
                    model_type = 'vit'
                else:
                    model_type = 'resnet'
    
    if model_type == 'clip_vit':
        return 'BACKBONE.image_encoder.transformer.resblocks.11'
    elif model_type == 'vit_transformer':
        return 'BACKBONE.base.transformer.resblocks.11'
    elif model_type == 'vit':
        return 'BACKBONE.base.blocks.11'
    elif model_type == 'resnet':
        return 'BACKBONE.base.layer4'
    else:
        return 'BACKBONE.base.transformer.resblocks.11'


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
    """为单张图像生成 Grad-CAM 热力图可视化"""
    original_image, pil_image = load_image(image_path)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    input_dict = {
        'RGB': torch.zeros_like(img_tensor),
        'NI': torch.zeros_like(img_tensor),
        'TI': torch.zeros_like(img_tensor)
    }
    input_dict[modality] = img_tensor
    
    try:
        gradcam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
    except Exception as e:
        print(f"⚠️  创建 Grad-CAM 失败: {e}")
        print(f"   尝试查找可用的目标层...")
        layers = find_target_layers(model, nn.Module)
        if layers:
            print(f"   找到 {len(layers)} 个可用层，使用第一个: {layers[0][0]}")
            target_layer = layers[0][0]
            gradcam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
        else:
            raise RuntimeError("无法找到合适的目标层，请手动指定 --target_layer 参数")
    
    try:
        heatmap, overlay = gradcam.generate_gradcam(
            input_dict,
            original_image,
            target_class=None,
            alpha=alpha
        )
    except Exception as e:
        print(f"❌ 生成 Grad-CAM 失败: {e}")
        raise
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
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
    """为多模态图像（RGB、NIR、TIR）生成 Grad-CAM 热力图可视化"""
    modalities = ['RGB', 'NI', 'TI']
    modality_names = ['RGB', 'NIR', 'TIR']
    
    images = {}
    for mod, mod_name in zip(modalities, modality_names):
        test_dir = os.path.join(dataset_root, 'test', mod)
        matching_files = [f for f in os.listdir(test_dir) if f.startswith(query_id) and f.endswith('.jpg')]
        if not matching_files:
            print(f"⚠️  未找到 {mod_name} 模态图像: {test_dir}")
            continue
        image_path = os.path.join(test_dir, matching_files[0])
        images[mod] = load_image(image_path)
    
    if not images:
        raise ValueError(f"未找到任何模态的图像，Query ID: {query_id}")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
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
    
    for row, (mod, mod_name) in enumerate(zip(modalities, modality_names)):
        if mod not in images:
            continue
        
        original_image, pil_image = images[mod]
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),
            'NI': torch.zeros_like(img_tensor),
            'TI': torch.zeros_like(img_tensor)
        }
        input_dict[mod] = img_tensor
        
        try:
            heatmap, overlay = gradcam.generate_gradcam(
                input_dict, original_image, target_class=None, alpha=alpha
            )
        except Exception as e:
            print(f"⚠️  {mod_name} 模态生成失败: {e}")
            continue
        
        axes[row, 0].imshow(original_image)
        axes[row, 0].set_title(f'{mod_name}', fontsize=12, fontweight='bold', pad=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f'{mod_name}', fontsize=12, fontweight='bold', pad=10)
        axes[row, 1].axis('off')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 已保存多模态可视化结果: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Grad-CAM 热力图可视化工具')
    
    parser.add_argument('--config_file', type=str, required=True, help='配置文件路径')
    parser.add_argument('--weight_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_path', type=str, help='单张图像路径')
    input_group.add_argument('--image_dir', type=str, help='图像目录路径')
    input_group.add_argument('--query_id', type=str, help='查询人员ID')
    
    parser.add_argument('--dataset_root', type=str, default='data/RGBNT201', help='数据集根目录')
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'NI', 'TI'], help='模态类型')
    parser.add_argument('--multimodal', action='store_true', help='启用多模态模式')
    parser.add_argument('--num_images', type=int, default=10, help='批量可视化时的图像数量')
    parser.add_argument('--target_layer', type=str, default=None, help='目标层路径')
    parser.add_argument('--alpha', type=float, default=0.4, help='热力图透明度')
    parser.add_argument('--list_layers', action='store_true', help='列出模型中所有可用的层')
    
    return parser.parse_args()


def main():
    """主函数：执行 Grad-CAM 热力图可视化"""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.config_file):
        print(f"❌ 配置文件不存在: {args.config_file}")
        return
    
    if not os.path.exists(args.weight_path):
        print(f"❌ 权重文件不存在: {args.weight_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms()
    print(f"🔧 使用设备: {device}")
    
    print("📦 加载模型配置和权重...")
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    camera_num = detect_camera_num_from_weights(args.weight_path)
    num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    
    model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model.load_param(args.weight_path)
    model.eval()
    print("✅ 模型加载完成")
    
    if args.target_layer:
        target_layer = args.target_layer
    else:
        print("🔍 自动检测目标层...")
        target_layer = get_target_layer_name(model)
        print(f"   使用目标层: {target_layer}")
    
    if args.list_layers:
        print("\n📋 模型中所有可用的层:")
        layers = find_target_layers(model, nn.Module)
        for i, (name, layer) in enumerate(layers[:20]):
            print(f"  [{i+1}] {name}: {type(layer).__name__}")
        if len(layers) > 20:
            print(f"  ... 还有 {len(layers) - 20} 个层未显示")
        print(f"\n💡 使用 --target_layer 参数指定目标层，例如:")
        if layers:
            print(f"   --target_layer {layers[0][0]}")
        return
    
    if args.image_path:
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
            print(f"\n🖼️  单模态可视化: Query ID = {args.query_id}, Modality = {args.modality}")
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
