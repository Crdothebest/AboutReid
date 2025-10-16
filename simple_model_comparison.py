"""
简化的模型对比可视化脚本
模仿小波脚本的方式，直接使用model.base避免相机嵌入层问题

作者：MambaPro团队
日期：2024
"""

import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from yacs.config import CfgNode as CN
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入自定义的模型结构
from modeling.make_model import make_model


def load_config(cfg_path):
    """加载YAML配置文件"""
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = CN(cfg_dict)
    return cfg


def add_missing_config(cfg, is_your_model=False):
    """添加缺失的配置参数"""
    if not hasattr(cfg, 'MODEL'):
        cfg.MODEL = CN()
    
    # 添加缺失的MODEL参数
    missing_params = {
        'FLOPS_TEST': False,
        'SIE_CAMERA': False,
        'SIE_VIEW': False,
        'SIE_COE': False,
        'DIRECT': False,
        'ID_LOSS_WEIGHT': 1.0,
        'TRIPLET_LOSS_WEIGHT': 1.0,
        'PROMPT': False,
        'ADAPTER': False,
        'MAMBA': False,
        'FROZEN': False,
        'ID_LOSS_TYPE': 'softmax',
        'TRANSFORMER_TYPE': 'ViT-B-16',
        'STRIDE_SIZE': [32, 32],
        'PRETRAIN_PATH_T': '',
        'NECK': 'bnneck',
        'NECK_FEAT': 256,
        'JPM': False,
        'LAST_STRIDE': 1,
        'MAMBA_BI': False,
        'MAMBA_BI_LAYER': 0,
        'MAMBA_BI_DIM': 768,
        'FEAT_DIM': 256,
        'NUM_CLASSES': 1051,
        'CAMERA_NUM': 6,
        'VIEW_NUM': 2
    }
    
    for param, default_value in missing_params.items():
        if not hasattr(cfg.MODEL, param):
            setattr(cfg.MODEL, param, default_value)
    
    # 如果是您的模型，启用多尺度和MoE
    if is_your_model:
        cfg.MODEL.USE_CLIP_MULTI_SCALE = True
        cfg.MODEL.USE_MULTI_SCALE_MOE = True


def load_image(image_path, input_size):
    """加载和预处理输入图像"""
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    rgb_image = np.array(image.resize(input_size)) / 255.0
    
    return input_tensor, rgb_image


def get_target_layer(model, layer_name):
    """从模型中获取指定的目标层"""
    try:
        parts = layer_name.split('.')
        current = model
        for part in parts:
            current = getattr(current, part)
        return current
    except AttributeError:
        print(f"⚠️  未找到目标层: {layer_name}")
        return None


def load_model(cfg_path, weight_path, is_your_model=False):
    """加载模型"""
    print(f"📦 加载配置文件: {cfg_path}")
    cfg = load_config(cfg_path)
    add_missing_config(cfg, is_your_model)
    
    print("🔄 初始化模型...")
    model = make_model(cfg, 
                      num_classes=cfg.MODEL.NUM_CLASSES,
                      camera_num=cfg.MODEL.CAMERA_NUM,
                      view_num=cfg.MODEL.VIEW_NUM)
    
    print(f"📥 加载模型权重: {weight_path}")
    model.load_param(weight_path)
    model.eval()
    
    return model, cfg


def get_gradcam_heatmap(model, input_tensor, target_layer_name):
    """获取Grad-CAM热力图 - 模仿小波脚本的方式"""
    try:
        # 优先尝试使用model.base（模仿小波脚本）
        if hasattr(model, 'BACKBONE') and hasattr(model.BACKBONE, 'base'):
            print("🔄 使用model.base（模仿小波脚本方式）...")
            base_model = model.BACKBONE.base
            
            # 获取目标层
            target_layer = get_target_layer(base_model, target_layer_name)
            if target_layer is None:
                print("⚠️  在model.base中找不到目标层")
                return None
            
            print(f"✅ 找到目标层: {target_layer_name}")
            print(f"✅ 目标层类型: {type(target_layer).__name__}")
            
            # 检查GradCAM构造函数参数
            import inspect
            sig = inspect.signature(GradCAM.__init__)
            gradcam_kwargs = {}
            if 'use_cuda' in sig.parameters:
                gradcam_kwargs['use_cuda'] = True
            
            # 直接使用base模型创建GradCAM（模仿小波脚本）
            cam = GradCAM(model=base_model, target_layers=[target_layer], **gradcam_kwargs)
            
            # 计算Grad-CAM
            try:
                print("🔄 正在计算Grad-CAM...")
                grayscale_cam = cam(input_tensor=input_tensor)[0]
                print(f"✅ Grad-CAM计算成功，形状: {grayscale_cam.shape}")
                return grayscale_cam
            except Exception as e:
                print(f"⚠️  Grad-CAM计算失败: {e}")
                return None
            finally:
                # 清理GradCAM对象
                try:
                    if hasattr(cam, 'activations_and_grads'):
                        cam.activations_and_grads.release()
                    del cam
                except:
                    pass
        else:
            print("⚠️  找不到model.base")
            return None
            
    except Exception as e:
        print(f"⚠️  Grad-CAM生成失败: {e}")
        return None


def create_comparison_visualization(rgb_image, baseline_cam, your_model_cam, output_dir):
    """创建对比可视化"""
    print("🎨 生成对比可视化...")
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison Visualization', fontsize=16, fontweight='bold')
    
    # 原始图像
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Baseline热力图
    if baseline_cam is not None:
        try:
            baseline_vis = show_cam_on_image(rgb_image, baseline_cam, use_rgb=True)
            axes[0, 1].imshow(baseline_vis)
            axes[0, 1].set_title('Baseline Model Attention', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
        except Exception as e:
            print(f"⚠️  Baseline热力图处理失败: {e}")
            axes[0, 1].text(0.5, 0.5, 'Baseline CAM\nProcessing Error', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'Baseline CAM\nNot Available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')
    
    # 您的模型热力图
    if your_model_cam is not None:
        try:
            your_model_vis = show_cam_on_image(rgb_image, your_model_cam, use_rgb=True)
            axes[0, 2].imshow(your_model_vis)
            axes[0, 2].set_title('Your Model Attention', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
        except Exception as e:
            print(f"⚠️  您的模型热力图处理失败: {e}")
            axes[0, 2].text(0.5, 0.5, 'Your Model CAM\nProcessing Error', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
    else:
        axes[0, 2].text(0.5, 0.5, 'Your Model CAM\nNot Available', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].axis('off')
    
    # 注意力差异图
    if baseline_cam is not None and your_model_cam is not None:
        try:
            diff_cam = your_model_cam - baseline_cam
            diff_vis = show_cam_on_image(rgb_image, diff_cam, use_rgb=True)
            axes[1, 0].imshow(diff_vis)
            axes[1, 0].set_title('Attention Difference\n(Your Model - Baseline)', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
        except Exception as e:
            print(f"⚠️  注意力差异图处理失败: {e}")
            axes[1, 0].text(0.5, 0.5, 'Attention Difference\nProcessing Error', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'Attention Difference\nCannot Generate', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
    
    # 保存图像
    output_path = os.path.join(output_dir, 'simple_model_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    
    print(f"✅ 对比可视化已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Simple Model Comparison Visualization")
    parser.add_argument("--baseline-cfg", type=str, required=True, help="Baseline config file")
    parser.add_argument("--your-model-cfg", type=str, required=True, help="Your model config file")
    parser.add_argument("--baseline-weight", type=str, required=True, help="Baseline model weight")
    parser.add_argument("--your-model-weight", type=str, required=True, help="Your model weight")
    parser.add_argument("--img-path", type=str, required=True, help="Input image path")
    parser.add_argument("--output-dir", type=str, default="simple_comparison_results", help="Output directory")
    parser.add_argument("--target-layer", type=str, default="transformer", help="Target layer name")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 开始简化模型对比可视化...")
    
    # 加载图像
    print("🖼️  加载测试图像...")
    input_tensor, rgb_image = load_image(args.img_path, (256, 128))
    input_tensor = input_tensor.cuda()
    print(f"✅ 图像加载完成，尺寸: {input_tensor.shape}")
    
    # 加载Baseline模型
    print("🔄 加载Baseline模型...")
    baseline_model, baseline_cfg = load_model(args.baseline_cfg, args.baseline_weight, False)
    print("✅ Baseline模型加载完成")
    
    # 加载您的模型
    print("🔄 加载您的模型...")
    your_model, your_model_cfg = load_model(args.your_model_cfg, args.your_model_weight, True)
    print("✅ 您的模型加载完成")
    
    # 生成Grad-CAM热力图
    print("🔥 生成Grad-CAM热力图...")
    
    # Baseline热力图
    print("🔄 生成Baseline热力图...")
    baseline_cam = get_gradcam_heatmap(baseline_model, input_tensor, args.target_layer)
    
    # 您的模型热力图
    print("🔄 生成您的模型热力图...")
    your_model_cam = get_gradcam_heatmap(your_model, input_tensor, args.target_layer)
    
    # 创建对比可视化
    create_comparison_visualization(rgb_image, baseline_cam, your_model_cam, args.output_dir)
    
    print("🎉 简化模型对比可视化完成！")
    print(f"📁 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
