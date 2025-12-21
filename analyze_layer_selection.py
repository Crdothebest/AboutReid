#!/usr/bin/env python
"""分析不同层对热力图生成的影响"""
import sys
sys.path.insert(0, '.')
import torch
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

# 检查模型结构
print("=" * 60)
print("模型结构分析")
print("=" * 60)

# 检查BACKBONE结构
if hasattr(model, 'BACKBONE'):
    backbone = model.BACKBONE
    print(f"\nBACKBONE类型: {type(backbone)}")
    
    if hasattr(backbone, 'base'):
        base = backbone.base
        print(f"BACKBONE.base类型: {type(base)}")
        
        if hasattr(base, 'transformer'):
            transformer = base.transformer
            print(f"BACKBONE.base.transformer类型: {type(transformer)}")
            
            if hasattr(transformer, 'resblocks'):
                resblocks = transformer.resblocks
                print(f"Transformer resblocks数量: {len(resblocks)}")
                print(f"可用层索引: 0-{len(resblocks)-1}")
                
                # 检查不同层的结构
                print("\n各层结构:")
                for i in range(min(3, len(resblocks))):
                    print(f"  resblocks[{i}]: {type(resblocks[i])}")
                if len(resblocks) > 3:
                    print(f"  ...")
                    print(f"  resblocks[{len(resblocks)-1}]: {type(resblocks[-1])}")

# 测试不同层
print("\n" + "=" * 60)
print("测试不同层对热力图生成的影响")
print("=" * 60)

# 加载一张测试图像
test_dir = '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201/test/RGB'
import os
rgb_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
if rgb_files:
    image_path = os.path.join(test_dir, rgb_files[0])
    print(f"\n使用测试图像: {image_path}")
    
    original_image, pil_image = load_image(image_path)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # 构建输入字典（只激活RGB模态）
    input_dict = {
        'RGB': img_tensor,
        'NI': torch.zeros_like(img_tensor),
        'TI': torch.zeros_like(img_tensor)
    }
    
    # 测试不同层
    test_layers = [
        'BACKBONE.base.transformer.resblocks.5',   # 中间层
        'BACKBONE.base.transformer.resblocks.8',   # 中后层
        'BACKBONE.base.transformer.resblocks.11', # 最后一层（当前使用）
    ]
    
    for layer_name in test_layers:
        try:
            print(f"\n测试层: {layer_name}")
            gradcam = GradCAM(model, target_layer=layer_name, use_cuda=True)
            
            # 生成热力图
            heatmap = gradcam.generate_cam(
                input_dict,
                cam_label=torch.tensor([0]).to(device),
                view_label=torch.tensor([0]).to(device)
            )
            
            print(f"  热力图: shape={heatmap.shape}, min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")
            print(f"  非零值: {(heatmap > 0).sum()} / {heatmap.size} ({100*(heatmap>0).sum()/heatmap.size:.2f}%)")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)








