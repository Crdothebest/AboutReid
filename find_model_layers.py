"""
查找模型中的层结构
用于确定合适的热力图目标层
"""

import argparse
import os
import torch
import yaml
from yacs.config import CfgNode as CN
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
    
    if is_your_model:
        cfg.MODEL.USE_CLIP_MULTI_SCALE = True
        cfg.MODEL.USE_MULTI_SCALE_MOE = True


def find_model_layers(model, prefix=""):
    """递归查找模型中的所有层"""
    layers = []
    
    for name, module in model.named_modules():
        if prefix:
            full_name = f"{prefix}.{name}" if name else prefix
        else:
            full_name = name
        
        # 只记录有实际参数的层
        if len(list(module.parameters())) > 0:
            layers.append((full_name, type(module).__name__))
    
    return layers


def main():
    parser = argparse.ArgumentParser(description="Find Model Layers")
    parser.add_argument("--cfg", type=str, required=True, help="Config file path")
    parser.add_argument("--weight", type=str, required=True, help="Model weight path")
    parser.add_argument("--is-your-model", action="store_true", help="Is your multi-scale MoE model")
    
    args = parser.parse_args()
    
    print(f"📦 加载配置文件: {args.cfg}")
    cfg = load_config(args.cfg)
    add_missing_config(cfg, args.is_your_model)
    
    print("🔄 初始化模型...")
    num_class = getattr(cfg.MODEL, 'NUM_CLASSES', 1051)
    camera_num = getattr(cfg.MODEL, 'CAMERA_NUM', 6)
    view_num = getattr(cfg.MODEL, 'VIEW_NUM', 2)
    
    model = make_model(cfg, 
                      num_class=num_class,
                      camera_num=camera_num,
                      view_num=view_num)
    
    print(f"📥 加载模型权重: {args.weight}")
    try:
        model.load_param(args.weight)
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"⚠️  模型权重加载失败: {e}")
        print("🔄 继续分析模型结构...")
    
    model.eval()
    
    print("\n🔍 查找模型中的所有层...")
    layers = find_model_layers(model)
    
    print(f"\n📊 找到 {len(layers)} 个有参数的层:")
    print("=" * 80)
    
    # 按类别分组显示
    multi_scale_layers = []
    moe_layers = []
    attention_layers = []
    other_layers = []
    
    for layer_name, layer_type in layers:
        if 'multi_scale' in layer_name.lower() or 'sliding' in layer_name.lower():
            multi_scale_layers.append((layer_name, layer_type))
        elif 'moe' in layer_name.lower() or 'expert' in layer_name.lower():
            moe_layers.append((layer_name, layer_type))
        elif 'attn' in layer_name.lower() or 'attention' in layer_name.lower():
            attention_layers.append((layer_name, layer_type))
        else:
            other_layers.append((layer_name, layer_type))
    
    # 显示多尺度相关层
    if multi_scale_layers:
        print("\n🎯 多尺度滑动窗口相关层 (推荐用于热力图):")
        for layer_name, layer_type in multi_scale_layers:
            print(f"  {layer_name:<50} {layer_type}")
    
    # 显示MoE相关层
    if moe_layers:
        print("\n🧠 MoE专家网络相关层 (推荐用于热力图):")
        for layer_name, layer_type in moe_layers:
            print(f"  {layer_name:<50} {layer_type}")
    
    # 显示注意力层
    if attention_layers:
        print("\n👁️  注意力相关层:")
        for layer_name, layer_type in attention_layers:
            print(f"  {layer_name:<50} {layer_type}")
    
    # 显示其他层
    if other_layers:
        print("\n📋 其他层:")
        for layer_name, layer_type in other_layers[:20]:  # 只显示前20个
            print(f"  {layer_name:<50} {layer_type}")
        if len(other_layers) > 20:
            print(f"  ... 还有 {len(other_layers) - 20} 个层")
    
    print("\n" + "=" * 80)
    print("💡 建议:")
    if multi_scale_layers:
        print("1. 使用多尺度滑动窗口层来证明多尺度特征提取的优越性")
    if moe_layers:
        print("2. 使用MoE专家网络层来证明专家选择和融合的优越性")
    print("3. 选择Baseline模型没有的层来突出您的模型优势")


if __name__ == "__main__":
    main()
