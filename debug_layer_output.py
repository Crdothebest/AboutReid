#!/usr/bin/env python
"""调试脚本：检查目标层的实际输出形状"""
import sys
sys.path.insert(0, '.')
import torch
from config import cfg
from modeling import make_model
from visualize_Cam.visualize_gradcam import build_transforms, load_image, detect_camera_num_from_weights

# 配置
cfg.merge_from_file('configs/RGBNT201/yzy_best_Mambapro_moe.yml')
cfg.freeze()

weight_path = '/home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = build_transforms()

# 创建模型
camera_num = detect_camera_num_from_weights(weight_path)
num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
model.load_param(weight_path)
model.eval()

# 测试图像
test_dir = '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201/test'
rgb_path = f"{test_dir}/RGB/000276_cam2_0_01.jpg"

# 加载图像
original_image, pil_image = load_image(rgb_path)
img_tensor = transform(pil_image).unsqueeze(0).to(device)

# 构建输入字典
input_dict = {
    'RGB': img_tensor,
    'NI': torch.zeros_like(img_tensor),
    'TI': torch.zeros_like(img_tensor)
}

cam_label = torch.tensor([0]).to(device)
view_label = torch.tensor([0]).to(device)

# 注册钩子检查 ln_post 的输出
activations_info = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations_info[name] = {
            'shape': output.shape,
            'dtype': output.dtype,
            'min': output.min().item(),
            'max': output.max().item(),
            'mean': output.mean().item()
        }
    return hook

# 注册钩子
target_layer = model.BACKBONE.base.ln_post
target_layer.register_forward_hook(hook_fn('ln_post'))

# 前向传播
with torch.no_grad():
    _ = model(input_dict, cam_label=cam_label, view_label=view_label)

# 打印信息
print("=" * 80)
print("目标层输出信息")
print("=" * 80)
for name, info in activations_info.items():
    print(f"\n{name}:")
    print(f"  形状: {info['shape']}")
    print(f"  数据类型: {info['dtype']}")
    print(f"  值域: [{info['min']:.6f}, {info['max']:.6f}]")
    print(f"  均值: {info['mean']:.6f}")

# 分析形状
if 'ln_post' in activations_info:
    shape = activations_info['ln_post']['shape']
    print(f"\n分析:")
    print(f"  批次大小: {shape[0]}")
    print(f"  序列长度: {shape[1]} (包含CLS token: {shape[1] == 129})")
    print(f"  特征维度: {shape[2]}")
    print(f"  预期patch数量: 16 * 8 = 128")
    print(f"  是否有CLS token: {shape[1] == 129}")
