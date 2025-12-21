#!/usr/bin/env python
"""测试不同层对NI和TI模态梯度的影响"""
import sys
sys.path.insert(0, '.')
import torch
from config import cfg
from modeling import make_model
from visualize_gradcam import build_transforms, load_image, detect_camera_num_from_weights
from grad_cam import GradCAM

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
person_id = '000276'

rgb_path = f"{test_dir}/RGB/{person_id}_cam2_0_01.jpg"
ni_path = f"{test_dir}/NI/{person_id}_cam2_0_01.jpg"
ti_path = f"{test_dir}/TI/{person_id}_cam2_0_01.jpg"

# 测试不同层
test_layers = [
    ('resblocks.5', 'BACKBONE.base.transformer.resblocks.5'),
    ('resblocks.8', 'BACKBONE.base.transformer.resblocks.8'),
    ('resblocks.9', 'BACKBONE.base.transformer.resblocks.9'),
    ('resblocks.10', 'BACKBONE.base.transformer.resblocks.10'),
    ('resblocks.11', 'BACKBONE.base.transformer.resblocks.11'),  # 当前使用的层
]

print("=" * 80)
print("测试不同层对NI和TI模态梯度的影响")
print("=" * 80)

results = {}

for layer_name, layer_path in test_layers:
    print(f"\n{'='*80}")
    print(f"测试层: {layer_name} ({layer_path})")
    print(f"{'='*80}")
    
    try:
        gradcam = GradCAM(model, target_layer=layer_path, use_cuda=True)
        
        layer_results = {}
        
        for modality, image_path in [('RGB', rgb_path), ('NI', ni_path), ('TI', ti_path)]:
            try:
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
                
                # 生成热力图
                heatmap = gradcam.generate_cam(
                    input_dict,
                    cam_label=torch.tensor([0]).to(device),
                    view_label=torch.tensor([0]).to(device)
                )
                
                # 检查梯度
                gradients = gradcam.gradients
                if gradients is not None:
                    grad_min = gradients.min().item()
                    grad_max = gradients.max().item()
                    grad_mean = gradients.mean().item()
                    grad_nonzero = (gradients != 0).sum().item()
                    grad_total = gradients.numel()
                else:
                    grad_min = grad_max = grad_mean = 0
                    grad_nonzero = grad_total = 0
                
                layer_results[modality] = {
                    'heatmap_max': heatmap.max(),
                    'heatmap_mean': heatmap.mean(),
                    'heatmap_nonzero': (heatmap > 0).sum(),
                    'grad_min': grad_min,
                    'grad_max': grad_max,
                    'grad_mean': grad_mean,
                    'grad_nonzero': grad_nonzero,
                    'grad_total': grad_total,
                }
                
                print(f"\n{modality}模态:")
                print(f"  热力图: max={heatmap.max():.6f}, mean={heatmap.mean():.6f}, non-zero={(heatmap>0).sum()}/{heatmap.size}")
                print(f"  梯度: min={grad_min:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}, non-zero={grad_nonzero}/{grad_total}")
                
            except Exception as e:
                print(f"\n{modality}模态: ❌ 失败 - {e}")
                layer_results[modality] = None
        
        results[layer_name] = layer_results
        
    except Exception as e:
        print(f"❌ 层 {layer_name} 测试失败: {e}")
        results[layer_name] = None

# 总结
print("\n" + "=" * 80)
print("测试结果总结")
print("=" * 80)

print("\n推荐层选择（基于NI和TI梯度非零）：")
for layer_name, layer_results in results.items():
    if layer_results:
        ni_grad = layer_results.get('NI', {}).get('grad_nonzero', 0)
        ti_grad = layer_results.get('TI', {}).get('grad_nonzero', 0)
        ni_heatmap = layer_results.get('NI', {}).get('heatmap_max', 0)
        ti_heatmap = layer_results.get('TI', {}).get('heatmap_max', 0)
        
        score = 0
        if ni_grad > 0:
            score += 1
        if ti_grad > 0:
            score += 1
        if ni_heatmap > 0:
            score += 1
        if ti_heatmap > 0:
            score += 1
        
        status = "✅" if score >= 3 else "⚠️" if score >= 1 else "❌"
        print(f"{status} {layer_name}: NI梯度={ni_grad}, TI梯度={ti_grad}, NI热力图={ni_heatmap:.4f}, TI热力图={ti_heatmap:.4f} (得分={score}/4)")

print("\n" + "=" * 80)
print("建议：选择NI和TI梯度都非零的层")
print("=" * 80)
