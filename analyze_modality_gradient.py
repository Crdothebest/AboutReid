#!/usr/bin/env python
"""分析不同模态的梯度传播"""
import sys
sys.path.insert(0, '.')
import torch
from config import cfg
from modeling import make_model
from visualize_gradcam import build_transforms, load_image
from grad_cam import GradCAM

# 加载模型
cfg.merge_from_file('configs/RGBNT201/yzy_best_Mambapro_moe.yml')
cfg.freeze()

weight_path = '/home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = build_transforms()

# 创建模型
from visualize_gradcam import detect_camera_num_from_weights
camera_num = detect_camera_num_from_weights(weight_path)
num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
model.load_param(weight_path)
model.eval()

# 测试图像路径
test_dir = '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201/test'
person_id = '000276'

rgb_path = f"{test_dir}/RGB/{person_id}_cam2_0_01.jpg"
ni_path = f"{test_dir}/NI/{person_id}_cam2_0_01.jpg"
ti_path = f"{test_dir}/TI/{person_id}_cam2_0_01.jpg"

import os
print("=" * 60)
print("检查图像路径")
print("=" * 60)
print(f"RGB路径存在: {os.path.exists(rgb_path)} - {rgb_path}")
print(f"NI路径存在: {os.path.exists(ni_path)} - {ni_path}")
print(f"TI路径存在: {os.path.exists(ti_path)} - {ti_path}")

# 测试不同模态
target_layer = 'BACKBONE.base.transformer.resblocks.11'
gradcam = GradCAM(model, target_layer=target_layer, use_cuda=True)

print("\n" + "=" * 60)
print("测试不同模态的热力图生成")
print("=" * 60)

for modality, image_path in [('RGB', rgb_path), ('NI', ni_path), ('TI', ti_path)]:
    if not os.path.exists(image_path):
        print(f"\n{modality}模态: 图像文件不存在，跳过")
        continue
    
    try:
        # 加载图像
        original_image, pil_image = load_image(image_path)
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 构建输入字典（只激活当前模态）
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),
            'NI': torch.zeros_like(img_tensor),
            'TI': torch.zeros_like(img_tensor)
        }
        input_dict[modality] = img_tensor
        
        print(f"\n{modality}模态:")
        print(f"  输入形状: {img_tensor.shape}")
        print(f"  输入值范围: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
        
        # 前向传播检查
        with torch.enable_grad():
            output = model(input_dict, 
                         cam_label=torch.tensor([0]).to(device), 
                         view_label=torch.tensor([0]).to(device))
            print(f"  模型输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")
            print(f"  激活值形状: {gradcam.activations.shape if gradcam.activations is not None else 'None'}")
            if gradcam.activations is not None:
                print(f"  激活值统计: min={gradcam.activations.min():.6f}, max={gradcam.activations.max():.6f}, mean={gradcam.activations.mean():.6f}")
        
        # 生成热力图
        heatmap = gradcam.generate_cam(
            input_dict,
            cam_label=torch.tensor([0]).to(device),
            view_label=torch.tensor([0]).to(device)
        )
        
        print(f"  热力图: shape={heatmap.shape}, min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")
        print(f"  非零值: {(heatmap > 0).sum()} / {heatmap.size} ({100*(heatmap>0).sum()/heatmap.size:.2f}%)")
        
        # 检查梯度
        if gradcam.gradients is not None:
            print(f"  梯度形状: {gradcam.gradients.shape}")
            print(f"  梯度统计: min={gradcam.gradients.min():.6f}, max={gradcam.gradients.max():.6f}, mean={gradcam.gradients.mean():.6f}")
            print(f"  非零梯度: {(gradcam.gradients != 0).sum().item()} / {gradcam.gradients.numel()}")
        else:
            print(f"  梯度: None")
            
    except Exception as e:
        print(f"\n{modality}模态: ❌ 失败 - {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)





