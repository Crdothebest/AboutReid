"""
多尺度滑动窗口+MoE特征可视化工具

该脚本用于生成基于Grad-CAM的多尺度滑动窗口和MoE特征热力图可视化，
帮助分析模型在多尺度特征提取和专家网络中的注意力分布。

主要功能：
1. 加载训练好的ReID模型（包含多尺度MoE模块）
2. 对多尺度滑动窗口层进行Grad-CAM分析
3. 对MoE专家网络进行注意力可视化
4. 生成多尺度特征热力图
5. 保存可视化结果

作者：MambaPro团队
日期：2024
"""

# 关于路径设置：
# 1. 模型权重路径：/home/zubuntu/workspace/yzy/MambaPro/mybest_model/experiment_20251013_110028/models/MambaProbest.pth
# 2. 输入图像路径：data/RGBNT201/test/RGB/000001_cam1_0_01.jpg
# 3. 配置文件路径：configs/RGBNT201/MambaPro_moe.yml
# 4. 输出目录：visualization_results/
# 5. 目标层名称：clip_multi_scale_moe.moe_fusion
# 6. 滑动窗口尺度：4, 8, 16

# 关于命令行参数：
# 使用您的配置文件和具体测试图像
# # 使用您的配置文件和具体测试图像
# python cam_multiscale_moe_visualize.py \
  # --cfg /home/zubuntu/workspace/yzy/MambaPro/mybest_model/experiment_20251015_132633/configs/experiment_config.yml \
  #--img-path /home/zubuntu/workspace/yzy/MambaPro/data/RGBNT201/test/RGB/000258_cam1_0_00.jpg \
  #--output-dir your_model_heatmaps \
  #--target-layer clip_multi_scale_moe.moe_fusion


import argparse
import os
import torch
import cv2
import numpy as np

# 设置matplotlib环境变量
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    import matplotlib.font_manager as fm
    
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
        'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC',
        'Source Han Sans SC', 'PingFang SC', 'Hiragino Sans GB'
    ]
    
    # 检查系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 选择第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
        print(f"✅ 使用中文字体: {selected_font}")
    else:
        # 如果没有找到中文字体，使用默认字体并禁用中文显示
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️  未找到中文字体，将使用英文标签")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 禁用字体警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    warnings.filterwarnings('ignore', message='.*Glyph.*missing from current font.*')
    warnings.filterwarnings('ignore', message='.*missing from current font.*')

# 初始化中文字体
setup_chinese_font()

# 定义英文标签映射
LABELS = {
    '原始图像': 'Original Image',
    'Grad-CAM热力图': 'Grad-CAM Heatmap',
    '多尺度特征': 'Multi-scale Features',
    'MoE专家权重': 'MoE Expert Weights',
    '尺度': 'Scale',
    '特征': 'Features',
    '权重': 'Weights',
    '激活值': 'Activations'
}
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
        layer_name (str): 目标层的名称
        
    Returns:
        torch.nn.Module: 目标层对象
        
    Raises:
        ValueError: 当指定层不存在时抛出异常
    """
    # 检查模型是否包含指定的层
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    else:
        # 如果层不存在，提供详细的错误信息
        available_layers = [name for name, _ in model.named_modules()]
        raise ValueError(f"Layer '{layer_name}' not found in model. "
                        f"Available layers: {available_layers[:10]}...")


def visualize_multiscale_features(model, input_tensor, scales=[4, 8, 16]):
    """
    可视化多尺度滑动窗口特征
    
    Args:
        model: 训练好的模型
        input_tensor: 输入图像张量
        scales: 滑动窗口尺度列表
        
    Returns:
        dict: 包含各尺度特征图的字典
    """
    model.eval()
    with torch.no_grad():
        # 获取多尺度特征
        multiscale_features = {}
        
        for scale in scales:
            # 这里需要根据实际模型结构调整
            # 假设模型有多尺度特征提取方法
            if hasattr(model, 'extract_multiscale_features'):
                features = model.extract_multiscale_features(input_tensor, scale)
                multiscale_features[f'scale_{scale}'] = features
            else:
                print(f"⚠️  模型不支持尺度 {scale} 的特征提取")
    
    return multiscale_features


def visualize_moe_expert_weights(model, input_tensor):
    """
    可视化MoE专家网络权重分布
    
    Args:
        model: 训练好的模型
        input_tensor: 输入图像张量
        
    Returns:
        dict: 包含专家权重信息的字典
    """
    model.eval()
    with torch.no_grad():
        # 获取MoE专家权重
        if hasattr(model, 'clip_multi_scale_moe'):
            moe_module = model.clip_multi_scale_moe
            if hasattr(moe_module, 'moe_fusion'):
                # 获取专家权重
                expert_weights = moe_module.moe_fusion.get_expert_weights(input_tensor)
                return {
                    'expert_weights': expert_weights,
                    'expert_names': ['4x4 Expert', '8x8 Expert', '16x16 Expert']
                }
    
    return None


def create_multiscale_visualization(rgb_image, multiscale_cams, output_dir):
    """
    创建多尺度特征可视化
    
    Args:
        rgb_image: 原始RGB图像
        multiscale_cams: 多尺度CAM结果
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始图像
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 多尺度特征可视化
    scales = [4, 8, 16]
    for i, scale in enumerate(scales):
        if f'scale_{scale}' in multiscale_cams:
            cam = multiscale_cams[f'scale_{scale}']
            visualization = show_cam_on_image(rgb_image, cam, use_rgb=True)
            
            row = (i + 1) // 2
            col = (i + 1) % 2
            axes[row, col].imshow(visualization)
            axes[row, col].set_title(f'Scale {scale}x{scale} Features')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiscale_features.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_moe_visualization(expert_weights, output_dir):
    """
    创建MoE专家权重可视化
    
    Args:
        expert_weights: 专家权重信息
        output_dir: 输出目录
    """
    if expert_weights is None:
        print("⚠️  无法获取MoE专家权重信息")
        return
    
    # 创建专家权重柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 专家权重分布
    weights = expert_weights['expert_weights'].cpu().numpy()
    expert_names = expert_weights['expert_names']
    
    ax1.bar(expert_names, weights, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('MoE Expert Weights Distribution')
    ax1.set_ylabel('Weight Value')
    ax1.set_ylim(0, 1)
    
    # 添加数值标签
    for i, v in enumerate(weights):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 专家权重饼图
    ax2.pie(weights, labels=expert_names, autopct='%1.1f%%', 
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('MoE Expert Weights Pie Chart')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moe_expert_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    主函数：执行多尺度MoE特征可视化
    
    工作流程：
    1. 解析命令行参数
    2. 加载模型配置和权重
    3. 预处理输入图像
    4. 执行多尺度特征分析
    5. 执行MoE专家权重分析
    6. 生成并保存可视化结果
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="Multi-Scale MoE Feature Visualization")
    parser.add_argument("--cfg", type=str, required=True, 
                       help="Path to config.yaml file")
    parser.add_argument("--img-path", type=str, required=True, 
                       help="Path to input image for visualization")
    parser.add_argument("--target-layer", type=str, 
                       default="clip_multi_scale_moe.moe_fusion", 
                       help="Layer name for Grad-CAM analysis")
    parser.add_argument("--output-dir", type=str, 
                       default="multiscale_moe_visualization", 
                       help="Directory to save visualization results")
    parser.add_argument("--scales", type=int, nargs='+', 
                       default=[4, 8, 16], 
                       help="Multi-scale window sizes")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 加载配置文件 ==========
    print("📦 加载配置文件...")
    cfg = load_config(args.cfg)
    print(f"✅ 配置文件加载完成: {args.cfg}")

    # ========== 模型初始化和加载 ==========
    print("🔄 初始化模型...")
    
    # 添加缺失的配置参数
    def add_missing_config(cfg):
        """为配置文件添加缺失的参数"""
        # 基础参数（不影响模型功能）
        if not hasattr(cfg.MODEL, 'FLOPS_TEST'):
            cfg.MODEL.FLOPS_TEST = False
        if not hasattr(cfg.MODEL, 'DIST_TRAIN'):
            cfg.MODEL.DIST_TRAIN = False
        if not hasattr(cfg.MODEL, 'DEVICE'):
            cfg.MODEL.DEVICE = 'cuda'
        if not hasattr(cfg.MODEL, 'DEVICE_ID'):
            cfg.MODEL.DEVICE_ID = '0'
        if not hasattr(cfg.MODEL, 'IF_LABELSMOOTH'):
            cfg.MODEL.IF_LABELSMOOTH = 'on'
        if not hasattr(cfg.MODEL, 'METRIC_LOSS_TYPE'):
            cfg.MODEL.METRIC_LOSS_TYPE = 'triplet'
        
        # 添加SIE相关参数
        if not hasattr(cfg.MODEL, 'SIE_CAMERA'):
            cfg.MODEL.SIE_CAMERA = True
        if not hasattr(cfg.MODEL, 'SIE_VIEW'):
            cfg.MODEL.SIE_VIEW = True
        if not hasattr(cfg.MODEL, 'SIE_COE'):
            cfg.MODEL.SIE_COE = 1.0
        if not hasattr(cfg.MODEL, 'DIRECT'):
            cfg.MODEL.DIRECT = 1
        
        # 添加其他可能缺失的参数
        if not hasattr(cfg.MODEL, 'ID_LOSS_WEIGHT'):
            cfg.MODEL.ID_LOSS_WEIGHT = 0.25
        if not hasattr(cfg.MODEL, 'TRIPLET_LOSS_WEIGHT'):
            cfg.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
        if not hasattr(cfg.MODEL, 'ID_LOSS_TYPE'):
            cfg.MODEL.ID_LOSS_TYPE = 'softmax'
        if not hasattr(cfg.MODEL, 'PROMPT'):
            cfg.MODEL.PROMPT = True
        if not hasattr(cfg.MODEL, 'ADAPTER'):
            cfg.MODEL.ADAPTER = True
        if not hasattr(cfg.MODEL, 'MAMBA'):
            cfg.MODEL.MAMBA = True
        if not hasattr(cfg.MODEL, 'FROZEN'):
            cfg.MODEL.FROZEN = True
        
        # 添加更多可能缺失的参数
        if not hasattr(cfg.MODEL, 'TRANSFORMER_TYPE'):
            cfg.MODEL.TRANSFORMER_TYPE = 'ViT-B-16'
        if not hasattr(cfg.MODEL, 'STRIDE_SIZE'):
            cfg.MODEL.STRIDE_SIZE = [16, 16]
        if not hasattr(cfg.MODEL, 'PRETRAIN_PATH_T'):
            cfg.MODEL.PRETRAIN_PATH_T = '/home/zubuntu/workspace/yzy/MambaPro/pths/ViT-B-16.pt'
        if not hasattr(cfg.MODEL, 'NECK'):
            cfg.MODEL.NECK = 'bnneck'
        if not hasattr(cfg.MODEL, 'NECK_FEAT'):
            cfg.MODEL.NECK_FEAT = 'after'
        if not hasattr(cfg.MODEL, 'JPM'):
            cfg.MODEL.JPM = False
        if not hasattr(cfg.MODEL, 'LAST_STRIDE'):
            cfg.MODEL.LAST_STRIDE = 1
        if not hasattr(cfg.MODEL, 'MAMBA_BI'):
            cfg.MODEL.MAMBA_BI = False
        if not hasattr(cfg.MODEL, 'MAMBA_BI_LAYER'):
            cfg.MODEL.MAMBA_BI_LAYER = 0
        if not hasattr(cfg.MODEL, 'MAMBA_BI_DIM'):
            cfg.MODEL.MAMBA_BI_DIM = 768
        
        # 关键参数：根据您的模型设置
        if not hasattr(cfg.MODEL, 'USE_CLIP_MULTI_SCALE'):
            cfg.MODEL.USE_CLIP_MULTI_SCALE = True  # 您的模型启用多尺度
        if not hasattr(cfg.MODEL, 'USE_MULTI_SCALE_MOE'):
            cfg.MODEL.USE_MULTI_SCALE_MOE = True   # 您的模型启用MoE
        if not hasattr(cfg.MODEL, 'USE_GATE_FUSION'):
            cfg.MODEL.USE_GATE_FUSION = False  # 默认禁用，根据实际需要调整
        
        return cfg
    
    # 为配置文件添加缺失参数
    cfg = add_missing_config(cfg)
    
    # 根据数据集设置类别数量
    num_classes = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    camera_num = getattr(cfg, 'CAMERA_NUM', 4)
    
    # 创建模型实例
    try:
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
        model.eval()  # 设置为评估模式
        model.cuda()  # 移动到GPU
        print(f"✅ 模型初始化完成，类别数: {num_classes}, 相机数: {camera_num}")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

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

    # ========== 多尺度特征分析 ==========
    print("🔍 分析多尺度滑动窗口特征...")
    multiscale_features = visualize_multiscale_features(model, input_tensor, args.scales)
    print(f"✅ 多尺度特征分析完成，尺度: {args.scales}")

    # ========== MoE专家权重分析 ==========
    print("🎯 分析MoE专家网络权重...")
    expert_weights = visualize_moe_expert_weights(model, input_tensor)
    if expert_weights:
        print("✅ MoE专家权重分析完成")
        print(f"   专家权重: {expert_weights['expert_weights'].cpu().numpy()}")
    else:
        print("⚠️  无法获取MoE专家权重信息")

    # ========== Grad-CAM 分析 ==========
    print("🔥 执行Grad-CAM分析...")
    try:
        target_layer = get_target_layer(model, args.target_layer)
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        print(f"✅ Grad-CAM计算完成，激活图形状: {grayscale_cam.shape}")
        
        # 生成热力图可视化
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        cv2.imwrite(os.path.join(args.output_dir, 'gradcam_heatmap.jpg'), 
                   cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print("✅ Grad-CAM热力图已保存")
        
    except Exception as e:
        print(f"⚠️  Grad-CAM分析失败: {e}")

    # ========== 生成可视化结果 ==========
    print("🎨 生成可视化结果...")
    
    # 多尺度特征可视化
    create_multiscale_visualization(rgb_image, multiscale_features, args.output_dir)
    print("✅ 多尺度特征可视化已保存")
    
    # MoE专家权重可视化
    create_moe_visualization(expert_weights, args.output_dir)
    print("✅ MoE专家权重可视化已保存")

    # ========== 输出分析信息 ==========
    print(f"\n🎉 可视化分析完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    print(f"🖼️  输入图像: {args.img_path}")
    print(f"🎯 目标层: {args.target_layer}")
    print(f"📏 分析尺度: {args.scales}")
    
    if expert_weights:
        weights = expert_weights['expert_weights'].cpu().numpy()
        print(f"⚖️  专家权重分布:")
        for i, (name, weight) in enumerate(zip(expert_weights['expert_names'], weights)):
            print(f"   {name}: {weight:.3f} ({weight*100:.1f}%)")


if __name__ == "__main__":
    main()
