# Category: dev_utils (开发调试)
# Description: 开发辅助工具，包括进程清理、层输出调试、环境诊断及后端 API

#!/usr/bin/env python
"""
EigenCAM 测试脚本

功能说明：
测试新添加的 EigenCAM 功能，生成热力图可视化。
EigenCAM 对 Transformer/Mamba 架构效果更好，能精准分离物体和背景。

使用方法：
  python test_eigencam.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --weight_path /path/to/model.pth \
    --query_id 000276 \
    --dataset_root /path/to/RGBNT201 \
    --output_dir outputs/Grad_CAM/eigencam_test

作者：MambaPro团队
日期：2024
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import cfg
from modeling import make_model
from visualize_Cam.grad_cam import EigenCAM, GradCAM


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


def load_image(image_path: str):
    """加载图像并返回原始图像和PIL图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    pil_image = Image.open(image_path).convert('RGB')
    original_image = np.array(pil_image)
    
    return original_image, pil_image


def detect_camera_num_from_weights(weight_path: str) -> int:
    """从权重文件中检测相机数量"""
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 查找相机嵌入层
        for key in state_dict.keys():
            if 'cv_embed' in key:
                # cv_embed 的形状通常是 [camera_num, feat_dim]
                cv_embed_shape = state_dict[key].shape
                if len(cv_embed_shape) >= 1:
                    return cv_embed_shape[0]
        
        # 如果找不到，返回默认值
        return 6
    except Exception as e:
        print(f"⚠️  无法从权重文件检测相机数量: {e}，使用默认值 6")
        return 6


def main():
    parser = argparse.ArgumentParser(
        description='EigenCAM 测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # RGBNT201 数据集（从配置文件读取路径）
  python test_eigencam.py \\
    --config_file ../MambaPro/configs/RGBNT201/MambaPro.yml \\
    --weight_path /path/to/MambaProbest.pth \\
    --query_id 000276

  # RGBNT100 数据集
  python test_eigencam.py \\
    --config_file ../MambaPro/configs/RGBNT100/MambaPro.yml \\
    --weight_path /path/to/model.pth \\
    --query_id 000001

  # MSVR310 数据集
  python test_eigencam.py \\
    --config_file ../MambaPro/configs/MSVR310/MambaPro.yml \\
    --weight_path /path/to/model.pth \\
    --query_id 000001

  # 禁用某些模块
  python test_eigencam.py \\
    --config_file ../MambaPro/configs/RGBNT201/MambaPro.yml \\
    --weight_path /path/to/model.pth \\
    --query_id 000276 \\
    --disable_mamba --disable_adapter

  # 启用某些模块（覆盖配置文件）
  python test_eigencam.py \\
    --config_file ../MambaPro/configs/RGBNT201/MambaPro.yml \\
    --weight_path /path/to/model.pth \\
    --query_id 000276 \\
    --enable_prompt --enable_mamba

  # 对比 EigenCAM 和 GradCAM
  python test_eigencam.py \\
    --config_file ../MambaPro/configs/RGBNT201/MambaPro.yml \\
    --weight_path /path/to/MambaProbest.pth \\
    --query_id 000276 \\
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \\
    --output_dir outputs/Grad_CAM/eigencam_test \\
    --compare
        """
    )
    
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
        '--query_id',
        type=str,
        required=True,
        help='要测试的人物ID（如 000276）'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201',
        help='数据集根目录（应包含 test/ 子目录）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/Grad_CAM/eigencam_test',
        help='输出目录路径'
    )
    parser.add_argument(
        '--target_layer',
        type=str,
        default='BACKBONE.base.ln_post',
        help='目标层路径（默认：ln_post，推荐使用。也可使用 resblocks.11）'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='同时生成 EigenCAM 和 GradCAM 进行对比'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='热力图透明度（0-1），默认 0.4'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型权重: {args.weight_path}")
    camera_num = detect_camera_num_from_weights(args.weight_path)
    num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model.load_param(args.weight_path)
    model.eval()
    print(f"✅ 模型加载完成 (camera_num={camera_num}, num_class={num_class})")
    
    # 构建变换
    transform = build_transforms()
    
    # 准备图像路径
    test_dir = os.path.join(args.dataset_root, 'test')
    rgb_path = os.path.join(test_dir, 'RGB', f"{args.query_id}_cam2_0_01.jpg")
    ni_path = os.path.join(test_dir, 'NI', f"{args.query_id}_cam2_0_01.jpg")
    ti_path = os.path.join(test_dir, 'TI', f"{args.query_id}_cam2_0_01.jpg")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==========================================
    # 🔥 新增：Mamba 专用的 Reshape 函数（根据解决方案文档）
    # ==========================================
    def mamba_reid_reshape_transform(tensor, height=16, width=8):
        """
        专门处理 Mamba/ViT ReID 模型的 1D output -> 2D feature map
        根据解决方案文档实现，解决"四角对称光斑"问题
        
        输入形状: [Batch, Sequence_Length, Channels] (e.g., [1, 129, 768])
        输出形状: [Batch, Channels, Height, Width] (e.g., [1, 768, 16, 8])
        """
        # 1. 硬编码目标尺寸 (ReID 256x128 / Patch 16 = 16x8)
        # 这是解决四角光斑最关键的一步，必须严格由高到宽
        target_h, target_w = height, width
        
        # 2. 剥离 CLS Token
        # 如果序列长度 (129) = H*W (128) + 1，说明有 CLS token
        if tensor.shape[1] == target_h * target_w + 1:
            tensor = tensor[:, 1:, :]
        
        # 3. 维度置换 [B, L, C] -> [B, C, L]
        result = tensor.transpose(1, 2)
        
        # 4. 强制 Reshape
        # ⚠️ 关键：必须是 (Batch, Channel, Height, Width)
        result = result.reshape(tensor.size(0), result.size(1), target_h, target_w)
        
        return result
    
    # 初始化 EigenCAM
    print(f"\n初始化 EigenCAM (目标层: {args.target_layer})")
    # 🔥 修改：传入 reshape_transform 参数（根据解决方案文档）
    eigencam = EigenCAM(
        model, 
        target_layer=args.target_layer, 
        use_cuda=torch.cuda.is_available(),
        reshape_transform=mamba_reid_reshape_transform  # 传入 reshape 函数
    )
    
    # 如果启用对比，也初始化 GradCAM
    gradcam = None
    if args.compare:
        print(f"初始化 GradCAM (目标层: {args.target_layer})")
        gradcam = GradCAM(model, target_layer=args.target_layer, use_cuda=torch.cuda.is_available())
    
    # 处理每个模态，收集所有数据
    modalities = [
        ('RGB', rgb_path, 'RGB'),
        ('NI', ni_path, 'NIR'),
        ('TI', ti_path, 'TIR')
    ]
    
    # 存储所有模态的原始图像和热力图
    original_images = []
    overlay_images = []
    modality_labels = []
    
    for modality, image_path, mod_name in modalities:
        if not os.path.exists(image_path):
            print(f"⚠️  跳过 {modality} 模态（文件不存在: {image_path}）")
            continue
        
        print(f"\n处理 {modality} 模态: {image_path}")
        
        # 加载图像
        original_image, pil_image = load_image(image_path)
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 构建输入字典
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),
            'NI': torch.zeros_like(img_tensor),
            'TI': torch.zeros_like(img_tensor)
        }
        input_dict[modality] = img_tensor
        
        # 准备标签
        cam_label = torch.tensor([0]).to(device)
        view_label = torch.tensor([0]).to(device)
        
        # 生成 EigenCAM 热力图
        print(f"  生成 EigenCAM 热力图...")
        try:
                heatmap_eigen = eigencam.generate_cam(
                    input_dict,
                    cam_label=cam_label,
                    view_label=view_label
                )
                # 注意：强制反转已在 grad_cam.py 的 EigenCAM.generate_cam() 中统一处理
                # 确保人体区域显示为高响应（红色），背景为低响应（蓝色）

                # ==========================================
                # 🔥 美化方案：实现"云雾感"效果
                # 解决"断层"现象和"马赛克"质感问题
                # ==========================================
                
                # 1. 阈值清理：去除低响应噪声
                threshold = 0.2
                heatmap_eigen[heatmap_eigen < threshold] = 0
                heatmap_eigen = (heatmap_eigen - threshold) / (1 - threshold + 1e-8)
                # 确保值域在 [0, 1]
                heatmap_eigen = np.clip(heatmap_eigen, 0, 1)
                
                # 2. 关键修改：先放大，再模糊！
                # 步骤 A: 双三次插值放大 (Bicubic) - 比双线性更平滑
                heatmap_resized = cv2.resize(
                    heatmap_eigen,
                    (original_image.shape[1], original_image.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )
                
                # 步骤 B: 🔥 高斯模糊 (Gaussian Blur) - 这是论文图"柔和感"的秘诀
                # kernel size (21, 21) 可以根据效果调整，越大越糊，越连贯
                # 这会把头部和腿部的高亮区域向中间扩散，填补躯干的"断层"
                heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (21, 21), 0)
                
                # 步骤 C: 重新归一化 (防止模糊后数值变小)
                heatmap_max = heatmap_blurred.max()
                if heatmap_max > 0:
                    heatmap_blurred = heatmap_blurred / heatmap_max
                else:
                    heatmap_blurred = np.zeros_like(heatmap_blurred)
                
                # 3. 渲染颜色：转换为 JET 颜色映射
                heatmap_colored = cv2.applyColorMap(
                    np.uint8(255 * heatmap_blurred), cv2.COLORMAP_JET
                )
                # 转换颜色格式：BGR -> RGB
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                # 4. 叠加：使用 alpha 参数控制热力图透明度
                # 推荐值：0.5 可以获得更好的视觉效果（云雾感更明显）
                overlay_alpha = args.alpha
                overlay_eigen = (
                    heatmap_colored * overlay_alpha + original_image.astype(np.float32) * (1 - overlay_alpha)
                ).astype(np.uint8)
                
                # 保存到列表
                original_images.append(original_image)
                overlay_images.append(overlay_eigen)
                modality_labels.append(mod_name)
                print(f"  ✅ {mod_name} 模态处理完成")
                
        except Exception as e:
            print(f"  ❌ EigenCAM 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果启用对比，也生成 GradCAM 版本
    if args.compare and gradcam is not None:
        print(f"\n生成 GradCAM 对比图...")
        original_images_grad = []
        overlay_images_grad = []
        modality_labels_grad = []
        
        for modality, image_path, mod_name in modalities:
            if not os.path.exists(image_path):
                continue
            
            original_image, pil_image = load_image(image_path)
            img_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            input_dict = {
                'RGB': torch.zeros_like(img_tensor),
                'NI': torch.zeros_like(img_tensor),
                'TI': torch.zeros_like(img_tensor)
            }
            input_dict[modality] = img_tensor
            
            cam_label = torch.tensor([0]).to(device)
            view_label = torch.tensor([0]).to(device)
            
            try:
                heatmap_grad = gradcam.generate_cam(
                    input_dict,
                    cam_label=cam_label,
                    view_label=view_label
                )
                
                overlay_grad = gradcam.overlay_heatmap(
                    original_image,
                    heatmap_grad,
                    alpha=args.alpha
                )
                
                original_images_grad.append(original_image)
                overlay_images_grad.append(overlay_grad)
                modality_labels_grad.append(mod_name)
                
            except Exception as e:
                print(f"  ❌ {mod_name} GradCAM 生成失败: {e}")
        
        # 生成 GradCAM 对比图
        if original_images_grad:
            num_modalities_grad = len(original_images_grad)
            fig_grad, axes_grad = plt.subplots(3, 2, figsize=(12, 18))
            for row, (orig_img, overlay_img, mod_label) in enumerate(zip(original_images_grad, overlay_images_grad, modality_labels_grad)):
                # 使用 aspect='auto' 让图片填满整个 subplot，消除空白
                axes_grad[row, 0].imshow(orig_img, aspect='auto')
                axes_grad[row, 0].set_title(mod_label, fontsize=12, fontweight='bold', pad=10)
                axes_grad[row, 0].axis('off')
                
                axes_grad[row, 1].imshow(overlay_img, aspect='auto')
                axes_grad[row, 1].set_title(mod_label, fontsize=12, fontweight='bold', pad=10)
                axes_grad[row, 1].axis('off')
            
            # 隐藏未使用的子图
            for row in range(num_modalities_grad, 3):
                axes_grad[row, 0].axis('off')
                axes_grad[row, 1].axis('off')
            
            # 调整子图间距：缩小左右两列之间的间隙
            plt.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.02, 
                               wspace=0.0, hspace=0.1)
            output_path_grad = os.path.join(args.output_dir, f"gradcam_{args.query_id}.png")
            plt.savefig(output_path_grad, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✅ GradCAM 对比图已保存: {output_path_grad}")
    
    # 生成论文格式的 EigenCAM 结果图（3行2列）
    if original_images:
        print(f"\n生成论文格式可视化结果...")
        num_modalities = len(original_images)
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        
        for row, (orig_img, overlay_img, mod_label) in enumerate(zip(original_images, overlay_images, modality_labels)):
            # 第一列：原始图像
            # 使用 aspect='auto' 让图片填满整个 subplot，消除空白
            axes[row, 0].imshow(orig_img, aspect='auto')
            axes[row, 0].set_title(mod_label, fontsize=12, fontweight='bold', pad=10)
            axes[row, 0].axis('off')
            
            # 第二列：热力图叠加图像
            # 使用 aspect='auto' 让图片填满整个 subplot，消除空白
            axes[row, 1].imshow(overlay_img, aspect='auto')
            axes[row, 1].set_title(mod_label, fontsize=12, fontweight='bold', pad=10)
            axes[row, 1].axis('off')
        
        # 隐藏未使用的子图
        for row in range(num_modalities, 3):
            axes[row, 0].axis('off')
            axes[row, 1].axis('off')
        
        # 调整子图间距：缩小左右两列之间的间隙
        # wspace: 列之间的间距，值越小间距越小
        plt.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.02, 
                           wspace=0.0, hspace=0.1)
        output_path = os.path.join(args.output_dir, f"eigencam_{args.query_id}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ EigenCAM 结果已保存: {output_path}")
    
    print(f"\n✅ 所有结果已保存到: {args.output_dir}")
    if args.compare:
        print(f"💡 提示: 对比 EigenCAM 和 GradCAM 的结果，EigenCAM 通常对 Transformer/Mamba 架构效果更好")


if __name__ == '__main__':
    main()
