#!/usr/bin/env python
"""
基于训练权重测试热力图效果脚本

功能说明：
每次运行基于指定的 .pth 权重文件，从测试集（test 目录）随机选择10张图像，
生成多模态热力图（RGB/NI/TI），并进行量化评估。

重要说明：
- ✅ 所有图像均来自测试集（test 目录），不涉及训练集
- ✅ 数据集结构: {dataset_root}/test/RGB/, {dataset_root}/test/NI/, {dataset_root}/test/TI/
- ✅ 从测试集随机选择人员ID，确保评估的客观性

使用方法：
python test_heatmap_from_weight.py \
    --weight_path path/to/model.pth \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --num_images 10 \
    --output_dir outputs/Grad_CAM/test_results

作者：MambaPro团队
日期：2024
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import cfg
from modeling import make_model
from visualize_gradcam import (
    build_transforms,
    load_image,
    detect_camera_num_from_weights,
    get_target_layer_name,
    visualize_multimodal
)
from grad_cam import GradCAM
from heatmap_evaluator import (
    evaluate_heatmap_quality,
    evaluate_cross_modal_alignment,
    diagnose_model_issues,
    create_simple_human_mask
)


def get_random_test_images(dataset_root: str, num_images: int = 10) -> list:
    """
    从测试集中随机选择指定数量的图像（按人员ID选择）
    
    功能说明：
    - 从测试集（test 目录）的 RGB 模态图像中随机选择人员ID
    - 确保热力图测试使用的是测试集，而非训练集
    
    Args:
        dataset_root (str): 数据集根目录（应包含 test/ 子目录）
        num_images (int): 需要选择的图像数量（实际是人员ID数量）
    
    Returns:
        list: 人员ID列表，如 ['000123', '000456', ...]
    
    注意：
        - 数据集结构应该是: {dataset_root}/test/RGB/, {dataset_root}/test/NI/, {dataset_root}/test/TI/
        - 只从测试集选择，不涉及训练集
    """
    # 从测试集的 RGB 模态目录选择图像
    test_rgb_dir = os.path.join(dataset_root, 'test', 'RGB')
    
    if not os.path.exists(test_rgb_dir):
        raise FileNotFoundError(f"测试集路径不存在: {test_rgb_dir}")
    
    # 获取所有测试图像
    image_files = [f for f in os.listdir(test_rgb_dir) if f.endswith('.jpg')]
    
    # 提取所有唯一的人员ID（从文件名前6位）
    person_ids = list(set([f[:6] for f in image_files]))
    
    # 随机挑选指定数量的人员ID
    selected_ids = random.sample(person_ids, min(num_images, len(person_ids)))
    selected_ids.sort()  # 排序以便于查看
    
    return selected_ids


def find_image_paths(dataset_root: str, person_id: str, modality: str) -> list:
    """
    查找指定人员ID和模态的图像路径（从测试集）
    
    功能说明：
    - 在测试集（test 目录）中查找指定人员ID和模态的图像
    - 确保热力图测试使用的是测试集数据
    
    Args:
        dataset_root (str): 数据集根目录（应包含 test/ 子目录）
        person_id (str): 人员ID，如 '000123'
        modality (str): 模态类型，'RGB'、'NI' 或 'TI'
    
    Returns:
        list: 图像路径列表（测试集中的图像）
    
    注意：
        - 路径格式: {dataset_root}/test/{modality}/{person_id}_*.jpg
        - 只从测试集查找，不涉及训练集
    """
    modality_map = {'RGB': 'RGB', 'NI': 'NI', 'TI': 'TI'}
    # 从测试集目录查找图像
    test_dir = os.path.join(dataset_root, 'test', modality_map[modality])
    
    if not os.path.exists(test_dir):
        return []
    
    # 查找匹配的图像
    matching_files = [
        os.path.join(test_dir, f) 
        for f in os.listdir(test_dir) 
        if f.startswith(person_id) and f.endswith('.jpg')
    ]
    
    return sorted(matching_files)


def generate_heatmap_for_person(
    model: nn.Module,
    person_id: str,
    dataset_root: str,
    transform,
    device: torch.device,
    target_layer: str,
    output_dir: str,
    alpha: float = 0.4
) -> dict:
    """
    为指定人员ID生成多模态热力图并评估
    
    Args:
        model: 训练好的模型
        person_id: 人员ID
        dataset_root: 数据集根目录
        transform: 图像预处理变换
        device: 计算设备
        target_layer: 目标层路径
        output_dir: 输出目录
        alpha: 热力图透明度
    
    Returns:
        dict: 评估结果
    """
    # 查找三种模态的图像路径
    rgb_paths = find_image_paths(dataset_root, person_id, 'RGB')
    ni_paths = find_image_paths(dataset_root, person_id, 'NI')
    ti_paths = find_image_paths(dataset_root, person_id, 'TI')
    
    if not rgb_paths or not ni_paths or not ti_paths:
        print(f"⚠️  人员ID {person_id} 缺少某些模态的图像，跳过")
        return None
    
    # 选择第一张图像（每个模态）
    rgb_path = rgb_paths[0]
    ni_path = ni_paths[0]
    ti_path = ti_paths[0]
    
    # 创建 Grad-CAM 对象
    try:
        gradcam = GradCAM(model, target_layer=target_layer, use_cuda=device.type == 'cuda')
    except Exception as e:
        print(f"⚠️  创建 Grad-CAM 失败: {e}")
        return None
    
    # 为三种模态生成热力图
    heatmaps = {}
    overlays = {}
    
    for modality, image_path in [('RGB', rgb_path), ('NI', ni_path), ('TI', ti_path)]:
        try:
            # 加载图像
            original_image, pil_image = load_image(image_path)
            
            # 预处理
            img_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # 构建多模态输入字典
            input_dict = {
                'RGB': torch.zeros_like(img_tensor),
                'NI': torch.zeros_like(img_tensor),
                'TI': torch.zeros_like(img_tensor)
            }
            
            # 激活当前模态
            if modality == 'RGB':
                input_dict['RGB'] = img_tensor
            elif modality == 'NI':
                input_dict['NI'] = img_tensor
            else:  # TI
                input_dict['TI'] = img_tensor
            
            # 生成热力图
            heatmap, overlay = gradcam.generate_gradcam(
                input_dict,
                original_image,
                target_class=None,
                alpha=alpha
            )
            
            heatmaps[modality] = heatmap
            overlays[modality] = overlay
            
        except Exception as e:
            print(f"⚠️  处理 {modality} 模态失败: {e}")
            return None
    
    # 评估热力图质量
    H, W = heatmaps['RGB'].shape
    human_mask = create_simple_human_mask((H, W))
    background_mask = ~human_mask
    
    metrics_rgb = evaluate_heatmap_quality(heatmaps['RGB'], human_mask, background_mask)
    metrics_ni = evaluate_heatmap_quality(heatmaps['NI'], human_mask, background_mask)
    metrics_ti = evaluate_heatmap_quality(heatmaps['TI'], human_mask, background_mask)
    
    # 评估跨模态对齐
    alignment = evaluate_cross_modal_alignment(
        heatmaps['RGB'], heatmaps['NI'], heatmaps['TI']
    )
    
    # 诊断问题
    avg_metrics = {
        'human_response': np.mean([metrics_rgb['human_response'], 
                                   metrics_ni['human_response'], 
                                   metrics_ti['human_response']]),
        'background_response': np.mean([metrics_rgb['background_response'], 
                                        metrics_ni['background_response'], 
                                        metrics_ti['background_response']]),
        'suppression_ratio': np.mean([metrics_rgb['suppression_ratio'], 
                                     metrics_ni['suppression_ratio'], 
                                     metrics_ti['suppression_ratio']]),
        'focus_score': np.mean([metrics_rgb['focus_score'], 
                               metrics_ni['focus_score'], 
                               metrics_ti['focus_score']])
    }
    
    diagnosis = diagnose_model_issues(avg_metrics, alignment)
    
    # 保存可视化结果（使用 visualize_multimodal 函数）
    output_path = os.path.join(output_dir, f"heatmap_{person_id}.png")
    try:
        # 直接调用 visualize_multimodal 生成并保存可视化图像
        visualize_multimodal(
            model, person_id, dataset_root, transform,
            device, target_layer, output_path, alpha
        )
    except Exception as e:
        print(f"⚠️  保存可视化结果失败: {e}")
        # 即使保存失败，也继续使用已生成的热力图数据进行评估
    
    return {
        'person_id': person_id,
        'metrics_rgb': metrics_rgb,
        'metrics_ni': metrics_ni,
        'metrics_ti': metrics_ti,
        'alignment': alignment,
        'diagnosis': diagnosis,
        'heatmaps': heatmaps,  # 保存热力图数据用于后续分析
        'output_path': output_path
    }


def generate_report(results: list, output_dir: str, weight_path: str):
    """
    生成评估报告
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
        weight_path: 权重文件路径
    """
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 热力图评估报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**权重文件**: `{weight_path}`\n\n")
        f.write(f"**测试样本数**: {len(results)}\n\n")
        f.write("---\n\n")
        
        # 统计信息
        alignment_scores = [r['alignment']['alignment_score'] for r in results]
        avg_alignment = np.mean(alignment_scores)
        
        background_responses = []
        for r in results:
            background_responses.extend([
                r['metrics_rgb']['background_response'],
                r['metrics_ni']['background_response'],
                r['metrics_ti']['background_response']
            ])
        avg_background = np.mean(background_responses)
        
        f.write("## 总体统计\n\n")
        f.write(f"- **平均跨模态对齐度**: {avg_alignment:.3f}\n")
        f.write(f"  - 优秀样本数 (>0.8): {sum(1 for s in alignment_scores if s > 0.8)}\n")
        f.write(f"  - 一般样本数 (0.5-0.8): {sum(1 for s in alignment_scores if 0.5 <= s <= 0.8)}\n")
        f.write(f"  - 较差样本数 (<0.5): {sum(1 for s in alignment_scores if s < 0.5)}\n")
        f.write(f"- **平均背景响应**: {avg_background:.3f} (目标 < 0.3)\n\n")
        f.write("---\n\n")
        
        # 详细结果
        f.write("## 详细结果\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"### 样本 {i}: 人员ID {result['person_id']}\n\n")
            
            # 单模态质量
            f.write("#### 单模态热力图质量\n\n")
            for modality, metrics in [('RGB', result['metrics_rgb']),
                                      ('NI', result['metrics_ni']),
                                      ('TI', result['metrics_ti'])]:
                f.write(f"**{modality} 模态**:\n")
                f.write(f"- 人体响应: {metrics['human_response']:.3f} (目标 > 0.6)\n")
                f.write(f"- 背景响应: {metrics['background_response']:.3f} (目标 < 0.3)\n")
                f.write(f"- 抑制比: {metrics['suppression_ratio']:.3f} (目标 > 2.0)\n")
                f.write(f"- 聚焦分数: {metrics['focus_score']:.3f} (目标 > 0.5)\n")
                f.write(f"- 质量等级: {metrics['quality_level']}\n\n")
            
            # 跨模态对齐
            f.write("#### 跨模态对齐评估 ⭐\n\n")
            alignment = result['alignment']
            f.write(f"- **对齐分数**: {alignment['alignment_score']:.3f}\n")
            f.write(f"  - RGB-NI IoU: {alignment['iou_rgb_nir']:.3f}\n")
            f.write(f"  - RGB-TI IoU: {alignment['iou_rgb_tir']:.3f}\n")
            f.write(f"  - NI-TI IoU: {alignment['iou_nir_tir']:.3f}\n")
            f.write(f"- **对齐等级**: {alignment['alignment_level']}\n\n")
            
            # 判断
            if alignment['alignment_score'] > 0.8:
                f.write("✅ **优秀**：模型学到了模态不变性特征\n\n")
            elif alignment['alignment_score'] >= 0.5:
                f.write("⚠️ **一般**：需要进一步优化跨模态对齐\n\n")
            else:
                f.write("❌ **较差**：跨模态对齐失败，需要重新设计模型\n\n")
            
            # 问题诊断
            diagnosis = result['diagnosis']
            if diagnosis['issues']:
                f.write("#### 问题诊断\n\n")
                f.write(diagnosis['summary'] + "\n\n")
            
            f.write("---\n\n")
        
        # 总结和建议
        f.write("## 总结\n\n")
        f.write("### 关键判断标准\n\n")
        f.write("| 指标 | 优秀 | 一般 | 较差 |\n")
        f.write("|------|------|------|------|\n")
        f.write("| 背景响应 | < 0.3 | 0.3-0.5 | > 0.5 |\n")
        f.write("| 跨模态对齐度 | > 0.8 | 0.5-0.8 | < 0.5 |\n")
        f.write("| 人体响应 | > 0.6 | 0.4-0.6 | < 0.4 |\n")
        f.write("| 聚焦分数 | > 0.5 | 0.3-0.5 | < 0.3 |\n\n")
        
        # 保存热力图数据
        heatmap_data_path = os.path.join(output_dir, 'heatmap_data.npz')
        heatmap_data = {}
        for r in results:
            pid = r['person_id']
            heatmap_data[f'{pid}_rgb'] = r['heatmaps']['RGB']
            heatmap_data[f'{pid}_ni'] = r['heatmaps']['NI']
            heatmap_data[f'{pid}_ti'] = r['heatmaps']['TI']
        np.savez(heatmap_data_path, **heatmap_data)
        f.write(f"### 热力图数据\n\n")
        f.write(f"热力图数据已保存到: `{heatmap_data_path}`\n")
        f.write(f"可以使用以下代码加载:\n")
        f.write(f"```python\n")
        f.write(f"import numpy as np\n")
        f.write(f"data = np.load('{heatmap_data_path}')\n")
        f.write(f"heatmap_rgb = data['{results[0]['person_id']}_rgb']\n")
        f.write(f"```\n\n")
    
    print(f"\n✅ 评估报告已保存: {report_path}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='基于训练权重测试热力图效果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python test_heatmap_from_weight.py \\
    --weight_path outputs/best_model.pth \\
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \\
    --num_images 10

  # 指定输出目录
  python test_heatmap_from_weight.py \\
    --weight_path outputs/best_model.pth \\
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \\
    --num_images 10 \\
    --output_dir outputs/Grad_CAM/test_20241217

  # 指定数据集路径
  python test_heatmap_from_weight.py \\
    --weight_path outputs/best_model.pth \\
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \\
    --dataset_root /path/to/RGBNT201 \\
    --num_images 10
        """
    )
    
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='模型权重文件路径（.pth 文件）'
    )
    
    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='配置文件路径（YAML 格式）'
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201',
        help='数据集根目录（应包含 test/ 子目录），默认: /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201。'
             '注意：热力图测试使用的是测试集（test 目录），而非训练集。'
    )
    
    parser.add_argument(
        '--num_images',
        type=int,
        default=10,
        help='测试图像数量（实际是人员ID数量），从测试集（test 目录）随机选择，默认: 10'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录，如果不指定则自动生成（基于时间戳）'
    )
    
    parser.add_argument(
        '--target_layer',
        type=str,
        default=None,
        help='目标层路径（用于 Grad-CAM），如果不指定则自动检测'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='热力图透明度（0.0-1.0），默认: 0.4'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（用于可复现性），如果不指定则使用随机值'
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print("="*60)
    print("基于训练权重测试热力图效果")
    print("="*60)
    
    # 检查路径
    if not os.path.exists(args.weight_path):
        print(f"❌ 权重文件不存在: {args.weight_path}")
        return
    
    if not os.path.exists(args.config_file):
        print(f"❌ 配置文件不存在: {args.config_file}")
        return
    
    if not os.path.exists(args.dataset_root):
        print(f"❌ 数据集路径不存在: {args.dataset_root}")
        return
    
    # 验证测试集目录存在
    test_dir = os.path.join(args.dataset_root, 'test')
    if not os.path.exists(test_dir):
        print(f"❌ 测试集目录不存在: {test_dir}")
        print(f"   请确保数据集根目录包含 test/ 子目录")
        return
    
    # 验证测试集的三种模态目录
    test_rgb_dir = os.path.join(test_dir, 'RGB')
    test_ni_dir = os.path.join(test_dir, 'NI')
    test_ti_dir = os.path.join(test_dir, 'TI')
    
    missing_modalities = []
    if not os.path.exists(test_rgb_dir):
        missing_modalities.append('RGB')
    if not os.path.exists(test_ni_dir):
        missing_modalities.append('NI')
    if not os.path.exists(test_ti_dir):
        missing_modalities.append('TI')
    
    if missing_modalities:
        print(f"⚠️  警告: 测试集缺少以下模态目录: {', '.join(missing_modalities)}")
        print(f"   测试集路径: {test_dir}")
        print(f"   这可能会影响多模态热力图的生成")
    else:
        print(f"✅ 测试集验证通过: {test_dir}")
        print(f"   - RGB: {test_rgb_dir}")
        print(f"   - NI: {test_ni_dir}")
        print(f"   - TI: {test_ti_dir}")
    
    # 创建输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        weight_name = os.path.basename(args.weight_path).replace('.pth', '')
        args.output_dir = os.path.join(
            script_dir,
            'outputs',
            'Grad_CAM',
            f'test_{weight_name}_{timestamp}'
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {args.output_dir}")
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms()
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    print("\n📦 加载模型配置和权重...")
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    camera_num = detect_camera_num_from_weights(args.weight_path)
    num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    
    model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model.load_param(args.weight_path)
    model.eval()
    print("✅ 模型加载完成")
    
    # 确定目标层
    if args.target_layer:
        target_layer = args.target_layer
    else:
        print("🔍 自动检测目标层...")
        target_layer = get_target_layer_name(model)
        print(f"   使用目标层: {target_layer}")
    
    # 从测试集随机选择图像
    print(f"\n🎲 从测试集（test 目录）随机选择 {args.num_images} 个人员ID...")
    print(f"   测试集路径: {os.path.join(args.dataset_root, 'test')}")
    person_ids = get_random_test_images(args.dataset_root, args.num_images)
    print(f"✅ 选择的人员ID: {', '.join(person_ids)}")
    print(f"   注意: 所有图像均来自测试集，不涉及训练集数据")
    
    # 为每个人员ID生成热力图并评估
    print(f"\n🖼️  生成热力图并评估...")
    results = []
    
    for i, person_id in enumerate(tqdm(person_ids, desc="处理图像")):
        try:
            result = generate_heatmap_for_person(
                model, person_id, args.dataset_root,
                transform, device, target_layer,
                args.output_dir, args.alpha
            )
            
            if result:
                results.append(result)
                print(f"  [{i+1}/{len(person_ids)}] {person_id}: "
                      f"对齐度={result['alignment']['alignment_score']:.3f} "
                      f"({result['alignment']['alignment_level']})")
        except Exception as e:
            print(f"  ⚠️  处理 {person_id} 失败: {e}")
            continue
    
    if not results:
        print("❌ 没有成功处理任何图像")
        return
    
    # 生成评估报告
    print(f"\n📊 生成评估报告...")
    generate_report(results, args.output_dir, args.weight_path)
    
    # 打印摘要
    print("\n" + "="*60)
    print("🎉 测试完成！")
    print("="*60)
    print(f"\n📋 结果摘要:")
    print(f"  - 成功处理: {len(results)}/{len(person_ids)} 个样本")
    
    alignment_scores = [r['alignment']['alignment_score'] for r in results]
    avg_alignment = np.mean(alignment_scores)
    print(f"  - 平均跨模态对齐度: {avg_alignment:.3f}")
    print(f"    - 优秀 (>0.8): {sum(1 for s in alignment_scores if s > 0.8)}")
    print(f"    - 一般 (0.5-0.8): {sum(1 for s in alignment_scores if 0.5 <= s <= 0.8)}")
    print(f"    - 较差 (<0.5): {sum(1 for s in alignment_scores if s < 0.5)}")
    
    print(f"\n📁 结果保存在: {args.output_dir}")
    print(f"  - 可视化图像: {len(results)} 张")
    print(f"  - 评估报告: evaluation_report.md")
    print(f"  - 热力图数据: heatmap_data.npz")


if __name__ == '__main__':
    main()
