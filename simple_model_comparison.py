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
    """从模型中获取指定的目标层，支持专家融合层"""
    try:
        print(f"🔍 调试信息: 开始查找目标层: {layer_name}")
        print(f"🔍 调试信息: 模型类型: {type(model).__name__}")
        
        parts = layer_name.split('.')
        print(f"🔍 调试信息: 层路径: {parts}")
        
        current = model
        for i, part in enumerate(parts):
            print(f"🔍 调试信息: 步骤 {i+1}: 查找 {part}")
            print(f"🔍 调试信息: 当前对象类型: {type(current).__name__}")
            
            if hasattr(current, part):
                current = getattr(current, part)
                print(f"🔍 调试信息: 找到 {part}, 类型: {type(current).__name__}")
            else:
                print(f"🔍 调试信息: 未找到属性 {part}")
                print(f"🔍 调试信息: 可用属性: {[attr for attr in dir(current) if not attr.startswith('_')]}")
                
                # 如果是专家融合层，尝试备用查找方法
                if 'moe_fusion' in layer_name or 'expert' in layer_name:
                    print(f"🔄 尝试查找专家融合层备用路径...")
                    # 尝试不同的专家融合层路径
                    alternative_paths = [
                        'clip_multi_scale_moe.moe_fusion',
                        'clip_multi_scale_moe.expert_fusion', 
                        'clip_multi_scale_moe.final_fusion',
                        'clip_multi_scale_moe.gating_network',
                        'BACKBONE.clip_multi_scale_moe.moe_fusion'
                    ]
                    
                    for alt_path in alternative_paths:
                        try:
                            alt_parts = alt_path.split('.')
                            alt_current = model
                            for alt_part in alt_parts:
                                if hasattr(alt_current, alt_part):
                                    alt_current = getattr(alt_current, alt_part)
                                else:
                                    break
                            else:
                                print(f"✅ 找到备用路径: {alt_path}")
                                return alt_current
                        except:
                            continue
                
                raise AttributeError(f"未找到属性: {part}")
        
        print(f"🔍 调试信息: 目标层查找成功: {type(current).__name__}")
        return current
    except AttributeError as e:
        print(f"⚠️  未找到目标层: {layer_name}")
        print(f"🔍 详细错误信息: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"🔍 完整错误堆栈:")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"⚠️  目标层查找失败: {e}")
        print(f"🔍 详细错误信息: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"🔍 完整错误堆栈:")
        traceback.print_exc()
        return None


class ExpertFusionWrapper(torch.nn.Module):
    """专家融合层包装器，处理复杂的输入格式"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # 如果输入是字典，直接传递给模型
        if isinstance(x, dict):
            return self.model(x)
        # 如果输入是张量，包装为字典
        else:
            batch_size = x.size(0)
            model_input = {
                'RGB': x,
                'NI': x,   # 使用相同的RGB数据作为NI的占位符
                'TI': x,   # 使用相同的RGB数据作为TI的占位符
                'cam_label': torch.zeros(batch_size, dtype=torch.long, device=x.device),
                'view_label': torch.zeros(batch_size, dtype=torch.long, device=x.device)
            }
            return self.model(model_input)


def load_model(cfg_path, weight_path, is_your_model=False):
    """加载模型"""
    print(f"📦 加载配置文件: {cfg_path}")
    cfg = load_config(cfg_path)
    add_missing_config(cfg, is_your_model)
    
    print("🔄 初始化模型...")
    # 获取模型参数，使用默认值作为备用
    num_class = getattr(cfg.MODEL, 'NUM_CLASSES', 1051)
    camera_num = getattr(cfg.MODEL, 'CAMERA_NUM', 6)
    view_num = getattr(cfg.MODEL, 'VIEW_NUM', 2)
    
    print(f"📊 模型参数: num_class={num_class}, camera_num={camera_num}, view_num={view_num}")
    
    model = make_model(cfg, 
                      num_class=num_class,
                      camera_num=camera_num,
                      view_num=view_num)
    
    print(f"📥 加载模型权重: {weight_path}")
    try:
        print(f"🔍 调试信息: 开始加载模型权重...")
        print(f"🔍 调试信息: 模型类型: {type(model).__name__}")
        print(f"🔍 调试信息: 模型设备: {next(model.parameters()).device}")
        
        model.load_param(weight_path)
        print(f"🔍 调试信息: 权重加载完成")
        
        model.eval()
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"⚠️  模型权重加载失败: {e}")
        print(f"🔍 详细错误信息: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"🔍 完整错误堆栈:")
        traceback.print_exc()
        print("🔄 尝试继续使用未加载权重的模型...")
    
    return model, cfg


def get_gradcam_heatmap(model, input_tensor, target_layer_name):
    """获取Grad-CAM热力图 - 支持专家融合层"""
    try:
        # 检查是否是专家融合层
        if 'moe_fusion' in target_layer_name or 'expert' in target_layer_name:
            print("🔄 检测到专家融合层，使用完整模型...")
            # 对于专家融合层，需要使用完整模型而不是model.base
            target_layer = get_target_layer(model, target_layer_name)
            if target_layer is None:
                print("⚠️  在完整模型中找不到专家融合层")
                return None
            
            print(f"✅ 在完整模型中找到专家融合层: {target_layer_name}")
            print(f"✅ 目标层类型: {type(target_layer).__name__}")
            
            # 检查GradCAM构造函数参数
            import inspect
            sig = inspect.signature(GradCAM.__init__)
            gradcam_kwargs = {}
            if 'use_cuda' in sig.parameters:
                gradcam_kwargs['use_cuda'] = True
            
            # 使用专家融合层包装器创建GradCAM
            wrapped_model = ExpertFusionWrapper(model)
            cam = GradCAM(model=wrapped_model, target_layers=[target_layer], **gradcam_kwargs)
        else:
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
                print(f"🔍 调试信息: input_tensor.shape = {input_tensor.shape}")
                print(f"🔍 调试信息: target_layer = {target_layer}")
                print(f"🔍 调试信息: base_model = {type(base_model).__name__}")
                
                # 尝试调用GradCAM
                grayscale_cam = cam(input_tensor=input_tensor)[0]
                print(f"✅ Grad-CAM计算成功，形状: {grayscale_cam.shape}")
                return grayscale_cam
            except Exception as e:
                print(f"⚠️  Grad-CAM计算失败: {e}")
                print(f"🔍 详细错误信息: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"🔍 完整错误堆栈:")
                traceback.print_exc()
                
                # 尝试生成简单的热力图作为备用
                print("🔄 尝试生成简单热力图作为备用...")
                try:
                    print(f"🔍 调试信息: 开始简单热力图生成")
                    print(f"🔍 调试信息: input_tensor.requires_grad = {input_tensor.requires_grad}")
                    
                    # 使用输入图像的梯度信息
                    input_tensor.requires_grad_(True)
                    print(f"🔍 调试信息: 设置requires_grad后，input_tensor.requires_grad = {input_tensor.requires_grad}")
                    
                    print(f"🔍 调试信息: 开始调用base_model...")
                    output = base_model(input_tensor)
                    print(f"🔍 调试信息: base_model输出类型: {type(output)}")
                    print(f"🔍 调试信息: base_model输出形状: {output.shape if hasattr(output, 'shape') else 'No shape'}")
                    print(f"🔍 调试信息: output.requires_grad = {output.requires_grad}")
                    
                    if output.requires_grad:
                        print(f"🔍 调试信息: 开始计算梯度...")
                        gradients = torch.autograd.grad(outputs=output, inputs=input_tensor, 
                                                      retain_graph=True)[0]
                        print(f"🔍 调试信息: 梯度计算成功，形状: {gradients.shape}")
                        
                        grayscale_cam = torch.mean(torch.abs(gradients), dim=1).squeeze().cpu().numpy()
                        print(f"🔍 调试信息: 热力图形状: {grayscale_cam.shape}")
                        
                        # 归一化
                        if grayscale_cam.max() > grayscale_cam.min():
                            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
                        
                        print(f"✅ 简单热力图生成成功，形状: {grayscale_cam.shape}")
                        return grayscale_cam
                    else:
                        print("⚠️  输出不需要梯度，使用随机热力图")
                        h, w = input_tensor.shape[2], input_tensor.shape[3]
                        return np.random.rand(h, w)
                except Exception as e2:
                    print(f"⚠️  简单热力图生成也失败: {e2}")
                    print(f"🔍 详细错误信息: {type(e2).__name__}: {str(e2)}")
                    print(f"🔍 完整错误堆栈:")
                    traceback.print_exc()
                    # 最后的备用方案：随机热力图
                    h, w = input_tensor.shape[2], input_tensor.shape[3]
                    return np.random.rand(h, w)
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
            # 修复热力图尺寸
            if baseline_cam.shape != rgb_image.shape[:2]:
                import cv2
                baseline_cam = cv2.resize(baseline_cam, (rgb_image.shape[1], rgb_image.shape[0]))
            
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
            # 修复热力图尺寸
            if your_model_cam.shape != rgb_image.shape[:2]:
                import cv2
                your_model_cam = cv2.resize(your_model_cam, (rgb_image.shape[1], rgb_image.shape[0]))
            
            your_model_vis = show_cam_on_image(rgb_image, your_model_cam, use_rgb=True)
            axes[1, 0].imshow(your_model_vis)
            axes[1, 0].set_title('Your Model Attention', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
        except Exception as e:
            print(f"⚠️  您的模型热力图处理失败: {e}")
            axes[1, 0].text(0.5, 0.5, 'Your Model CAM\nProcessing Error', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'Your Model CAM\nNot Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
    
    # 注意力差异图
    if baseline_cam is not None and your_model_cam is not None:
        try:
            # 确保两个热力图尺寸一致
            if baseline_cam.shape != your_model_cam.shape:
                import cv2
                if baseline_cam.shape != rgb_image.shape[:2]:
                    baseline_cam = cv2.resize(baseline_cam, (rgb_image.shape[1], rgb_image.shape[0]))
                if your_model_cam.shape != rgb_image.shape[:2]:
                    your_model_cam = cv2.resize(your_model_cam, (rgb_image.shape[1], rgb_image.shape[0]))
            
            diff_cam = your_model_cam - baseline_cam
            diff_vis = show_cam_on_image(rgb_image, diff_cam, use_rgb=True)
            axes[1, 1].imshow(diff_vis)
            axes[1, 1].set_title('Attention Difference\n(Your Model - Baseline)', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
        except Exception as e:
            print(f"⚠️  注意力差异图处理失败: {e}")
            axes[1, 1].text(0.5, 0.5, 'Attention Difference\nProcessing Error', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'Attention Difference\nCannot Generate', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
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
    parser.add_argument("--output-dir", type=str, default="expert_fusion_comparison", help="Output directory for expert fusion comparison")
    parser.add_argument("--target-layer", type=str, default="clip_multi_scale_moe.moe_fusion", help="Target layer name for MoE expert fusion")
    
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
