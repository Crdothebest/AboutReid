#!/usr/bin/env python
"""调试 Grad-CAM：检查激活值和梯度"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from config import cfg
from modeling import make_model
from visualize_Cam.visualize_gradcam import build_transforms, load_image
from visualize_Cam.grad_cam import GradCAM

# 加载模型
cfg.merge_from_file('configs/RGBNT201/yzy_best_Mambapro_moe.yml')
cfg.freeze()

weight_path = '/home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = build_transforms()

# 创建模型
from visualize_Cam.visualize_gradcam import detect_camera_num_from_weights
camera_num = detect_camera_num_from_weights(weight_path)
num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
model.load_param(weight_path)
model.eval()

# 加载一张测试图像
image_path = '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201/test/RGB/000258_c1_1.jpg'
original_image, pil_image = load_image(image_path)
img_tensor = transform(pil_image).unsqueeze(0).to(device)

# 构建输入字典
input_dict = {
    'RGB': img_tensor,
    'NI': torch.zeros_like(img_tensor),
    'TI': torch.zeros_like(img_tensor)
}

# 创建 Grad-CAM
target_layer = 'BACKBONE.base.transformer.resblocks.11'
gradcam = GradCAM(model, target_layer=target_layer, use_cuda=True)

# 前向传播
with torch.enable_grad():
    output = model(input_dict, cam_label=torch.tensor([0]).to(device), view_label=torch.tensor([0]).to(device))

print(f"模型输出: {output.shape if hasattr(output, 'shape') else type(output)}")
print(f"激活值: {gradcam.activations.shape if gradcam.activations is not None else 'None'}")
if gradcam.activations is not None:
    print(f"  激活值统计: min={gradcam.activations.min():.6f}, max={gradcam.activations.max():.6f}, mean={gradcam.activations.mean():.6f}")

# 计算目标得分
if isinstance(output, (list, tuple)):
    output_tensor = output[0]
else:
    output_tensor = output

if output_tensor.dim() == 1:
    target_score = torch.norm(output_tensor, p=2)
else:
    target_score = output_tensor[0].sum() if output_tensor.dim() == 2 else torch.norm(output_tensor[0], p=2)

print(f"目标得分: {target_score.item():.6f}, requires_grad: {target_score.requires_grad}")

# 反向传播
model.zero_grad()
target_score.backward()

print(f"梯度: {gradcam.gradients.shape if gradcam.gradients is not None else 'None'}")
if gradcam.gradients is not None:
    print(f"  梯度统计: min={gradcam.gradients.min():.6f}, max={gradcam.gradients.max():.6f}, mean={gradcam.gradients.mean():.6f}")
    print(f"  非零梯度数量: {(gradcam.gradients != 0).sum().item()} / {gradcam.gradients.numel()}")

# 生成热力图
try:
    heatmap = gradcam.generate_cam(input_dict, cam_label=torch.tensor([0]).to(device), view_label=torch.tensor([0]).to(device))
    print(f"\n热力图: shape={heatmap.shape}, min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")
    print(f"非零值数量: {(heatmap > 0).sum()} / {heatmap.size}")
except Exception as e:
    print(f"生成热力图失败: {e}")
    import traceback
    traceback.print_exc()
