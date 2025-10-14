"""
Grad-CAM小波特征可视化工具

该脚本用于生成基于Grad-CAM的小波变换特征热力图可视化，
帮助分析模型在小波域中的注意力分布和特征学习情况。

主要功能：
1. 加载训练好的ReID模型
2. 对指定的小波变换层进行Grad-CAM分析
3. 生成热力图可视化结果
4. 保存可视化图像

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

# 导入自定义的模型结构
from modeling.make_model import make_model


def load_config(cfg_path):
    """
    加载YAML配置文件
    
    Args:
        cfg_path (str): 配置文件路径
        
    Returns:
        CN: 配置对象，包含所有模型和训练参数
    """
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = CN(cfg_dict)
    return cfg


def load_image(image_path, input_size):
    """
    加载和预处理输入图像
    
    Args:
        image_path (str): 图像文件路径
        input_size (tuple): 目标图像尺寸 (height, width)
        
    Returns:
        tuple: (input_tensor, rgb_image)
            - input_tensor: 预处理后的张量，用于模型输入
            - rgb_image: 原始RGB图像数组，用于可视化叠加
    """
    # 构建图像预处理管道
    transform = transforms.Compose([
        transforms.Resize(input_size),                    # 调整图像尺寸
        transforms.ToTensor(),                           # 转换为张量 [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet标准化
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像并转换为RGB格式
    image = Image.open(image_path).convert("RGB")
    
    # 预处理为模型输入张量
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    # 保存原始图像用于可视化叠加（归一化到[0,1]）
    rgb_image = np.array(image.resize(input_size)) / 255.0
    
    return input_tensor, rgb_image


def get_target_layer(model, layer_name):
    """
    从模型中获取指定的目标层，用于Grad-CAM分析
    
    Args:
        model: 训练好的模型对象
        layer_name (str): 目标层的名称，如 "tokens_to_token.wavelet_conv.1"
        
    Returns:
        torch.nn.Module: 目标层对象
        
    Raises:
        ValueError: 当指定层不存在时抛出异常
    """
    # 检查模型是否包含指定的层
    if hasattr(model.base, layer_name):
        return getattr(model.base, layer_name)
    else:
        # 如果层不存在，提供详细的错误信息
        available_layers = [name for name, _ in model.base.named_modules()]
        raise ValueError(f"Layer '{layer_name}' not found in model.base. "
                        f"Available layers: {available_layers[:10]}...")


def main():
    """
    主函数：执行Grad-CAM小波特征可视化
    
    工作流程：
    1. 解析命令行参数
    2. 加载模型配置和权重
    3. 预处理输入图像
    4. 执行Grad-CAM分析
    5. 生成并保存热力图可视化
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="Grad-CAM Wavelet Feature Visualization")
    parser.add_argument("--cfg", type=str, required=True, 
                       help="Path to config.yaml file")
    parser.add_argument("--img-path", type=str, required=True, 
                       help="Path to input image for visualization")
    parser.add_argument("--target-layer", type=str, 
                       default="tokens_to_token.wavelet_conv.1", 
                       help="Layer name for Grad-CAM analysis")
    parser.add_argument("--output", type=str, 
                       default="cam_wavelet_output.jpg", 
                       help="Path to save CAM visualization result")
    args = parser.parse_args()

    # ========== 加载配置文件 ==========
    print("📦 加载配置文件...")
    cfg = load_config(args.cfg)
    print(f"✅ 配置文件加载完成: {args.cfg}")

    # ========== 模型初始化和加载 ==========
    print("🔄 初始化模型...")
    # 根据数据集设置类别数量（需要根据实际数据集调整）
    num_classes = 1051  # RGBNT201数据集的类别数
    camera_num, view_num = 6, 2  # 相机数量和视角数量
    
    # 创建模型实例
    model = make_model(cfg, num_classes=num_classes, camera_num=camera_num, view_num=view_num)
    model.eval()  # 设置为评估模式
    model.cuda()  # 移动到GPU
    print(f"✅ 模型初始化完成，类别数: {num_classes}, 相机数: {camera_num}")

    # ========== 加载预训练权重 ==========
    print("📥 加载模型权重...")
    if os.path.exists(cfg.TEST.WEIGHT):
        print(f"Loading weights from {cfg.TEST.WEIGHT}")
        model.load_param(cfg.TEST.WEIGHT)
        print("✅ 模型权重加载完成")
    else:
        raise FileNotFoundError(f"Weight file not found: {cfg.TEST.WEIGHT}")

    # ========== 图像预处理 ==========
    print("🖼️  加载和预处理图像...")
    input_tensor, rgb_image = load_image(args.img_path, tuple(cfg.INPUT.SIZE_TEST))
    input_tensor = input_tensor.cuda()  # 移动到GPU
    print(f"✅ 图像预处理完成，尺寸: {input_tensor.shape}")

    # ========== Grad-CAM 分析设置 ==========
    print("🎯 设置Grad-CAM分析...")
    print(f"目标层: {args.target_layer}")
    
    # 获取目标层对象
    target_layer = get_target_layer(model.base, args.target_layer)
    print(f"✅ 目标层获取成功: {type(target_layer).__name__}")

    # ========== 执行Grad-CAM分析 ==========
    print("🔥 执行Grad-CAM分析...")
    # 创建Grad-CAM对象
    cam = GradCAM(model=model.base, target_layers=[target_layer], use_cuda=True)
    
    # 计算梯度激活图
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    print(f"✅ Grad-CAM计算完成，激活图形状: {grayscale_cam.shape}")

    # ========== 生成可视化结果 ==========
    print("🎨 生成热力图可视化...")
    # 将热力图叠加到原始图像上
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    print("✅ 热力图可视化生成完成")

    # ========== 保存结果 ==========
    print("💾 保存可视化结果...")
    # 转换颜色空间并保存
    cv2.imwrite(args.output, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"🎉 CAM结果已保存到: {args.output}")
    
    # 输出分析信息
    print("\n📊 分析信息:")
    print(f"   输入图像: {args.img_path}")
    print(f"   目标层: {args.target_layer}")
    print(f"   输出文件: {args.output}")
    print(f"   激活图范围: [{grayscale_cam.min():.3f}, {grayscale_cam.max():.3f}]")


if __name__ == "__main__":
    main()
