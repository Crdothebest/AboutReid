# Category: vis_utils (可视化工具)
# Description: 提供热力图 (CAM)、t-SNE 降维、检索结果展示等模型可视化功能

#!/usr/bin/env python
"""
热力图评估脚本

功能说明：
基于热力图介绍文档，对生成的热力图进行量化评估，包括：
1. 单模态热力图质量评估
2. 跨模态对齐度评估
3. 模型问题诊断和改进建议

使用方法：
python evaluate_heatmaps.py \
    --heatmap_dir outputs/Grad_CAM/batch_visualization \
    --output_report outputs/Grad_CAM/evaluation_report.md

作者：MambaPro团队
日期：2024
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import json

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from heatmap_evaluator import (
    evaluate_heatmap_quality,
    evaluate_cross_modal_alignment,
    diagnose_model_issues,
    create_simple_human_mask
)


def load_heatmap_from_image(image_path: str) -> np.ndarray:
    """
    从图像文件中加载热力图
    
    注意：这是一个简化版本。实际使用时，应该从保存的热力图数据文件中加载。
    如果热力图是保存在图像中的，需要提取热力图通道。
    
    Args:
        image_path (str): 热力图图像路径
    
    Returns:
        np.ndarray: 热力图，形状为 [H, W]（值域 [0, 1]）
    """
    # 这里假设热力图是单独保存的灰度图
    # 实际实现可能需要从可视化图像中提取热力图
    if os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.astype(np.float32) / 255.0
    return None


def extract_heatmap_from_visualization(vis_image_path: str) -> np.ndarray:
    """
    从可视化图像中提取热力图
    
    功能说明：
    可视化图像通常包含三部分：原始图像、热力图、叠加图像。
    这里假设热力图在中间位置。
    
    Args:
        vis_image_path (str): 可视化图像路径
    
    Returns:
        np.ndarray: 提取的热力图
    """
    img = cv2.imread(vis_image_path)
    if img is None:
        return None
    
    H, W = img.shape[:2]
    # 假设热力图在中间 1/3 区域
    heatmap_region = img[:, W//3:2*W//3]
    
    # 转换为灰度图并归一化
    heatmap_gray = cv2.cvtColor(heatmap_region, cv2.COLOR_BGR2GRAY)
    heatmap = heatmap_gray.astype(np.float32) / 255.0
    
    return heatmap


def evaluate_multimodal_heatmaps(
    heatmap_rgb_path: str,
    heatmap_nir_path: str,
    heatmap_tir_path: str,
    human_mask: np.ndarray = None
) -> Dict:
    """
    评估多模态热力图
    
    Args:
        heatmap_rgb_path (str): RGB 热力图路径
        heatmap_nir_path (str): NIR 热力图路径
        heatmap_tir_path (str): TIR 热力图路径
        human_mask (np.ndarray, optional): 人体区域掩码
    
    Returns:
        dict: 评估结果
    """
    # 加载热力图
    heatmap_rgb = load_heatmap_from_image(heatmap_rgb_path)
    heatmap_nir = load_heatmap_from_image(heatmap_nir_path)
    heatmap_tir = load_heatmap_from_image(heatmap_tir_path)
    
    if heatmap_rgb is None or heatmap_nir is None or heatmap_tir is None:
        print(f"⚠️  无法加载热力图文件")
        return None
    
    # 确保尺寸一致
    H, W = heatmap_rgb.shape
    if heatmap_nir.shape != (H, W) or heatmap_tir.shape != (H, W):
        heatmap_nir = cv2.resize(heatmap_nir, (W, H))
        heatmap_tir = cv2.resize(heatmap_tir, (W, H))
    
    # 创建人体掩码（如果没有提供）
    if human_mask is None:
        human_mask = create_simple_human_mask((H, W))
    background_mask = ~human_mask
    
    # 评估单模态质量
    metrics_rgb = evaluate_heatmap_quality(heatmap_rgb, human_mask, background_mask)
    metrics_nir = evaluate_heatmap_quality(heatmap_nir, human_mask, background_mask)
    metrics_tir = evaluate_heatmap_quality(heatmap_tir, human_mask, background_mask)
    
    # 评估跨模态对齐
    alignment_metrics = evaluate_cross_modal_alignment(heatmap_rgb, heatmap_nir, heatmap_tir)
    
    # 诊断问题
    # 使用平均指标进行诊断
    avg_metrics = {
        'human_response': np.mean([metrics_rgb['human_response'], 
                                   metrics_nir['human_response'], 
                                   metrics_tir['human_response']]),
        'background_response': np.mean([metrics_rgb['background_response'], 
                                        metrics_nir['background_response'], 
                                        metrics_tir['background_response']]),
        'suppression_ratio': np.mean([metrics_rgb['suppression_ratio'], 
                                     metrics_nir['suppression_ratio'], 
                                     metrics_tir['suppression_ratio']]),
        'focus_score': np.mean([metrics_rgb['focus_score'], 
                               metrics_nir['focus_score'], 
                               metrics_tir['focus_score']])
    }
    
    diagnosis = diagnose_model_issues(avg_metrics, alignment_metrics)
    
    return {
        'metrics_rgb': metrics_rgb,
        'metrics_nir': metrics_nir,
        'metrics_tir': metrics_tir,
        'alignment': alignment_metrics,
        'diagnosis': diagnosis
    }


def generate_evaluation_report(
    results: List[Dict],
    output_path: str
):
    """
    生成评估报告（Markdown 格式）
    
    Args:
        results (list): 评估结果列表
        output_path (str): 输出报告路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 热力图评估报告\n\n")
        f.write("基于热力图介绍文档的量化评估结果\n\n")
        f.write("---\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"## 样本 {i}\n\n")
            
            # 单模态质量评估
            f.write("### 单模态热力图质量\n\n")
            for modality in ['rgb', 'nir', 'tir']:
                metrics = result.get(f'metrics_{modality}', {})
                f.write(f"#### {modality.upper()} 模态\n\n")
                f.write(f"- 人体响应: {metrics.get('human_response', 0):.3f} (目标 > 0.6)\n")
                f.write(f"- 背景响应: {metrics.get('background_response', 0):.3f} (目标 < 0.3)\n")
                f.write(f"- 抑制比: {metrics.get('suppression_ratio', 0):.3f} (目标 > 2.0)\n")
                f.write(f"- 聚焦分数: {metrics.get('focus_score', 0):.3f} (目标 > 0.5)\n")
                f.write(f"- 质量等级: {metrics.get('quality_level', 'N/A')}\n\n")
            
            # 跨模态对齐评估
            f.write("### 跨模态对齐评估 ⭐\n\n")
            alignment = result.get('alignment', {})
            f.write(f"- **对齐分数**: {alignment.get('alignment_score', 0):.3f}\n")
            f.write(f"  - RGB-NIR IoU: {alignment.get('iou_rgb_nir', 0):.3f}\n")
            f.write(f"  - RGB-TIR IoU: {alignment.get('iou_rgb_tir', 0):.3f}\n")
            f.write(f"  - NIR-TIR IoU: {alignment.get('iou_nir_tir', 0):.3f}\n")
            f.write(f"- **对齐等级**: {alignment.get('alignment_level', 'N/A')}\n\n")
            
            # 判断标准
            alignment_score = alignment.get('alignment_score', 0)
            if alignment_score > 0.8:
                f.write("✅ **优秀**：模型学到了模态不变性特征\n\n")
            elif alignment_score >= 0.5:
                f.write("⚠️ **一般**：需要进一步优化跨模态对齐\n\n")
            else:
                f.write("❌ **较差**：跨模态对齐失败，需要重新设计模型\n\n")
            
            # 问题诊断
            f.write("### 问题诊断\n\n")
            diagnosis = result.get('diagnosis', {})
            f.write(diagnosis.get('summary', '无问题') + "\n\n")
            
            if diagnosis.get('recommendations'):
                f.write("#### 改进建议\n\n")
                for j, rec in enumerate(diagnosis['recommendations'], 1):
                    f.write(f"{j}. **[{rec['priority']}]** {rec['action']}\n")
                    f.write(f"   ```python\n")
                    f.write(f"   {rec['code']}\n")
                    f.write(f"   ```\n\n")
            
            f.write("---\n\n")
        
        # 总结
        f.write("## 总结\n\n")
        f.write("### 关键判断标准\n\n")
        f.write("| 指标 | 优秀 | 一般 | 较差 |\n")
        f.write("|------|------|------|------|\n")
        f.write("| 背景响应 | < 0.3 | 0.3-0.5 | > 0.5 |\n")
        f.write("| 跨模态对齐度 | > 0.8 | 0.5-0.8 | < 0.5 |\n")
        f.write("| 人体响应 | > 0.6 | 0.4-0.6 | < 0.4 |\n")
        f.write("| 聚焦分数 | > 0.5 | 0.3-0.5 | < 0.3 |\n\n")
        
        f.write("### 快速检查清单\n\n")
        f.write("- [ ] 背景响应是否 < 0.3？\n")
        f.write("- [ ] 跨模态对齐度是否 > 0.8？\n")
        f.write("- [ ] 三种模态下高亮区域位置是否一致？\n")
        f.write("- [ ] 热力图是否集中在人体关键部位？\n\n")
    
    print(f"✅ 评估报告已保存: {output_path}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='热力图评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--heatmap_dir',
        type=str,
        required=True,
        help='热力图目录路径（包含多模态热力图文件）'
    )
    
    parser.add_argument(
        '--output_report',
        type=str,
        default='outputs/Grad_CAM/evaluation_report.md',
        help='输出报告路径（Markdown 格式）'
    )
    
    parser.add_argument(
        '--human_mask_path',
        type=str,
        default=None,
        help='人体区域掩码路径（可选，如果不提供将使用简单掩码）'
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    print("="*60)
    print("热力图评估工具")
    print("="*60)
    
    # 检查目录
    if not os.path.exists(args.heatmap_dir):
        print(f"❌ 热力图目录不存在: {args.heatmap_dir}")
        return
    
    # 查找热力图文件
    # 这里需要根据实际文件命名规则来查找
    # 假设文件命名格式为: model_name_person_XXXXX.png
    
    print(f"\n📁 扫描热力图目录: {args.heatmap_dir}")
    print("⚠️  注意：当前实现需要根据实际文件结构调整")
    print("   建议：直接使用 evaluate_heatmaps() 函数评估单个样本")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    
    print(f"\n✅ 评估工具已准备就绪")
    print(f"   使用 heatmap_evaluator 模块中的函数进行评估")
    print(f"   示例代码请参考 heatmap_evaluator.py 的 __main__ 部分")


if __name__ == '__main__':
    main()
