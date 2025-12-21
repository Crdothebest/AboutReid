#!/usr/bin/env python
"""
热力图评估工具

功能说明：
基于热力图介绍文档，实现热力图的量化评估指标，包括：
1. 热力图质量评估（背景抑制能力、特征学习有效性）
2. 跨模态对齐度评估（多模态热力图位置一致性）
3. 模型诊断工具（问题识别和改进建议）

主要功能：
- evaluate_heatmap_quality: 评估单模态热力图质量
- evaluate_cross_modal_alignment: 评估跨模态对齐度
- diagnose_model_issues: 诊断模型问题并给出改进建议

作者：MambaPro团队
日期：2024
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV (cv2) 未安装，某些功能可能不可用")


def evaluate_heatmap_quality(
    heatmap: np.ndarray,
    human_mask: np.ndarray,
    background_mask: np.ndarray
) -> Dict[str, float]:
    """
    评估热力图质量
    
    功能说明：
    根据热力图介绍文档中的标准，评估热力图的质量，包括：
    - 人体区域平均响应（应该 > 0.6）
    - 背景区域平均响应（应该 < 0.3）
    - 抑制比（人体/背景，应该 > 2.0）
    - 聚焦分数（高响应区域集中度，应该 > 0.5）
    
    Args:
        heatmap (np.ndarray): 热力图，形状为 [H, W]（值域 [0, 1]）
        human_mask (np.ndarray): 人体区域掩码，形状为 [H, W]（布尔数组）
        background_mask (np.ndarray): 背景区域掩码，形状为 [H, W]（布尔数组）
    
    Returns:
        dict: 评估指标字典
            - 'human_response': 人体区域平均响应
            - 'background_response': 背景区域平均响应
            - 'suppression_ratio': 抑制比（人体/背景）
            - 'focus_score': 聚焦分数（高响应区域集中度）
            - 'quality_level': 质量等级（'优秀'/'一般'/'较差'）
    
    示例:
        >>> heatmap = np.random.rand(256, 128)  # 示例热力图
        >>> human_mask = np.zeros((256, 128), dtype=bool)
        >>> human_mask[50:200, 30:100] = True  # 人体区域
        >>> background_mask = ~human_mask  # 背景区域
        >>> metrics = evaluate_heatmap_quality(heatmap, human_mask, background_mask)
        >>> print(f"背景响应: {metrics['background_response']:.3f}")
    """
    # 确保热力图和掩码尺寸匹配
    assert heatmap.shape == human_mask.shape == background_mask.shape, \
        f"尺寸不匹配: heatmap={heatmap.shape}, human_mask={human_mask.shape}, background_mask={background_mask.shape}"
    
    # 确保掩码是布尔类型
    human_mask = human_mask.astype(bool)
    background_mask = background_mask.astype(bool)
    
    # 1. 计算人体区域平均响应
    if human_mask.sum() > 0:
        human_response = heatmap[human_mask].mean()
    else:
        warnings.warn("人体掩码为空，无法计算人体响应")
        human_response = 0.0
    
    # 2. 计算背景区域平均响应
    if background_mask.sum() > 0:
        background_response = heatmap[background_mask].mean()
    else:
        warnings.warn("背景掩码为空，无法计算背景响应")
        background_response = 0.0
    
    # 3. 计算抑制比（人体响应 / 背景响应）
    suppression_ratio = human_response / (background_response + 1e-6)
    
    # 4. 计算聚焦分数：高响应区域（>0.7）占人体区域的比例
    if human_mask.sum() > 0:
        high_response_mask = (heatmap > 0.7) & human_mask
        focus_score = high_response_mask.sum() / human_mask.sum()
    else:
        focus_score = 0.0
    
    # 5. 评估质量等级
    quality_level = _assess_quality_level(
        human_response, background_response, suppression_ratio, focus_score
    )
    
    return {
        'human_response': float(human_response),
        'background_response': float(background_response),
        'suppression_ratio': float(suppression_ratio),
        'focus_score': float(focus_score),
        'quality_level': quality_level
    }


def _assess_quality_level(
    human_response: float,
    background_response: float,
    suppression_ratio: float,
    focus_score: float
) -> str:
    """
    评估热力图质量等级
    
    判断标准（基于文档）：
    - 优秀：人体响应 > 0.6, 背景响应 < 0.3, 抑制比 > 2.0, 聚焦分数 > 0.5
    - 一般：部分指标满足
    - 较差：大部分指标不满足
    """
    excellent_count = 0
    if human_response > 0.6:
        excellent_count += 1
    if background_response < 0.3:
        excellent_count += 1
    if suppression_ratio > 2.0:
        excellent_count += 1
    if focus_score > 0.5:
        excellent_count += 1
    
    if excellent_count >= 3:
        return '优秀'
    elif excellent_count >= 2:
        return '一般'
    else:
        return '较差'


def evaluate_cross_modal_alignment(
    heatmap_rgb: np.ndarray,
    heatmap_nir: np.ndarray,
    heatmap_tir: np.ndarray,
    threshold: float = None  # 自动计算阈值
) -> Dict[str, float]:
    """
    评估跨模态热力图对齐度
    
    功能说明：
    根据热力图介绍文档，计算三种模态（RGB、NIR、TIR）热力图的空间对齐度。
    这是跨模态 ReID 的核心评估指标。
    
    算法：
    1. 将热力图二值化为高响应区域（>threshold）
    2. 计算两两之间的 IoU（交并比）
    3. 对齐度 = 三个 IoU 的平均值
    
    判断标准（基于文档）：
    - 对齐度 > 0.8：✅ 优秀，模型学到了模态不变性特征
    - 对齐度 0.5-0.8：⚠️ 一般，需要进一步优化
    - 对齐度 < 0.5：❌ 较差，跨模态对齐失败
    
    Args:
        heatmap_rgb (np.ndarray): RGB 模态热力图，形状为 [H, W]（值域 [0, 1]）
        heatmap_nir (np.ndarray): NIR 模态热力图，形状为 [H, W]（值域 [0, 1]）
        heatmap_tir (np.ndarray): TIR 模态热力图，形状为 [H, W]（值域 [0, 1]）
        threshold (float): 高响应阈值，默认 0.7
    
    Returns:
        dict: 对齐度评估结果
            - 'alignment_score': 对齐分数（0-1）
            - 'iou_rgb_nir': RGB 和 NIR 的 IoU
            - 'iou_rgb_tir': RGB 和 TIR 的 IoU
            - 'iou_nir_tir': NIR 和 TIR 的 IoU
            - 'alignment_level': 对齐等级（'优秀'/'一般'/'较差'）
    
    示例:
        >>> heatmap_rgb = np.random.rand(256, 128)
        >>> heatmap_nir = np.random.rand(256, 128)
        >>> heatmap_tir = np.random.rand(256, 128)
        >>> result = evaluate_cross_modal_alignment(heatmap_rgb, heatmap_nir, heatmap_tir)
        >>> print(f"对齐度: {result['alignment_score']:.3f}")
    """
    # 确保所有热力图尺寸一致
    assert heatmap_rgb.shape == heatmap_nir.shape == heatmap_tir.shape, \
        f"热力图尺寸不一致: RGB={heatmap_rgb.shape}, NIR={heatmap_nir.shape}, TIR={heatmap_tir.shape}"
    
    # 1. 自动计算阈值（如果未指定）
    # 使用每个热力图的最大值的70%作为阈值，确保至少有一些高响应区域
    if threshold is None:
        max_rgb = heatmap_rgb.max()
        max_nir = heatmap_nir.max()
        max_tir = heatmap_tir.max()
        # 使用三个模态最大值的70%作为阈值，但至少为0.1
        threshold = max(0.1, min(max_rgb, max_nir, max_tir) * 0.7)
    
    # 1. 将热力图二值化为高响应区域
    high_response_rgb = heatmap_rgb > threshold
    high_response_nir = heatmap_nir > threshold
    high_response_tir = heatmap_tir > threshold
    
    # 2. 计算两两之间的 IoU（交并比）
    # IoU = (A ∩ B) / (A ∪ B)
    
    # RGB 和 NIR 的 IoU
    intersection_rgb_nir = (high_response_rgb & high_response_nir).sum()
    union_rgb_nir = (high_response_rgb | high_response_nir).sum()
    iou_rgb_nir = intersection_rgb_nir / (union_rgb_nir + 1e-6)
    
    # RGB 和 TIR 的 IoU
    intersection_rgb_tir = (high_response_rgb & high_response_tir).sum()
    union_rgb_tir = (high_response_rgb | high_response_tir).sum()
    iou_rgb_tir = intersection_rgb_tir / (union_rgb_tir + 1e-6)
    
    # NIR 和 TIR 的 IoU
    intersection_nir_tir = (high_response_nir & high_response_tir).sum()
    union_nir_tir = (high_response_nir | high_response_tir).sum()
    iou_nir_tir = intersection_nir_tir / (union_nir_tir + 1e-6)
    
    # 3. 计算平均对齐度
    alignment_score = (iou_rgb_nir + iou_rgb_tir + iou_nir_tir) / 3
    
    # 4. 评估对齐等级
    if alignment_score > 0.8:
        alignment_level = '优秀'
    elif alignment_score >= 0.5:
        alignment_level = '一般'
    else:
        alignment_level = '较差'
    
    return {
        'alignment_score': float(alignment_score),
        'iou_rgb_nir': float(iou_rgb_nir),
        'iou_rgb_tir': float(iou_rgb_tir),
        'iou_nir_tir': float(iou_nir_tir),
        'alignment_level': alignment_level
    }


def diagnose_model_issues(
    heatmap_metrics: Dict[str, float],
    alignment_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, any]:
    """
    诊断模型问题并给出改进建议
    
    功能说明：
    根据热力图评估指标，诊断模型存在的问题，并基于文档提供改进建议。
    
    诊断的问题类型（基于文档）：
    1. 背景抑制不足（背景响应 > 0.5）
    2. 跨模态对齐失败（对齐度 < 0.5）
    3. 特征提取不充分（聚焦分数 < 0.3）
    4. 特征学习不稳定（人体响应 < 0.4）
    
    Args:
        heatmap_metrics (dict): 热力图质量评估结果（来自 evaluate_heatmap_quality）
        alignment_metrics (dict, optional): 跨模态对齐评估结果（来自 evaluate_cross_modal_alignment）
    
    Returns:
        dict: 诊断结果
            - 'issues': 问题列表
            - 'severity': 严重程度（'P0'/'P1'）
            - 'recommendations': 改进建议列表
    
    示例:
        >>> metrics = evaluate_heatmap_quality(heatmap, human_mask, background_mask)
        >>> alignment = evaluate_cross_modal_alignment(heatmap_rgb, heatmap_nir, heatmap_tir)
        >>> diagnosis = diagnose_model_issues(metrics, alignment)
        >>> print(diagnosis['issues'])
    """
    issues = []
    recommendations = []
    severity = 'P1'  # 默认优先级
    
    # 1. 检查背景抑制能力
    background_response = heatmap_metrics.get('background_response', 0.0)
    if background_response > 0.5:
        issues.append({
            'type': '背景抑制不足',
            'symptom': f'背景响应值过高: {background_response:.3f} (应该 < 0.3)',
            'severity': 'P0'
        })
        recommendations.append({
            'priority': 'P0',
            'action': '增加背景抑制损失函数',
            'code': 'loss_background = -log(1 - background_response)'
        })
        recommendations.append({
            'priority': 'P0',
            'action': '使用更强的数据增强（背景替换、随机裁剪）',
            'code': 'transforms.RandomBackgroundReplacement()'
        })
        recommendations.append({
            'priority': 'P0',
            'action': '增加困难负样本挖掘',
            'code': 'hard_negative_mining(background_samples)'
        })
        severity = 'P0'  # 最高优先级
    
    # 2. 检查跨模态对齐（如果提供了对齐指标）
    if alignment_metrics:
        alignment_score = alignment_metrics.get('alignment_score', 1.0)
        if alignment_score < 0.5:
            issues.append({
                'type': '跨模态对齐失败',
                'symptom': f'对齐度过低: {alignment_score:.3f} (应该 > 0.8)',
                'severity': 'P0'
            })
            recommendations.append({
                'priority': 'P0',
                'action': '添加跨模态对比学习损失',
                'code': 'loss_cross_modal = contrastive_loss(feat_rgb, feat_nir, feat_tir)'
            })
            recommendations.append({
                'priority': 'P0',
                'action': '增加模态不变性约束',
                'code': 'loss_invariant = ||feat_rgb - feat_nir|| + ||feat_rgb - feat_tir||'
            })
            recommendations.append({
                'priority': 'P0',
                'action': '检查 MoE 融合策略，确保不同模态使用相似的专家',
                'code': 'check_moe_expert_weights(modality_rgb, modality_nir, modality_tir)'
            })
            severity = 'P0'  # 最高优先级
        elif alignment_score < 0.8:
            issues.append({
                'type': '跨模态对齐一般',
                'symptom': f'对齐度: {alignment_score:.3f} (目标 > 0.8)',
                'severity': 'P1'
            })
            recommendations.append({
                'priority': 'P1',
                'action': '优化跨模态对齐损失权重',
                'code': 'lambda_alignment = 0.5  # 调整权重'
            })
    
    # 3. 检查特征提取充分性
    focus_score = heatmap_metrics.get('focus_score', 0.0)
    if focus_score < 0.3:
        issues.append({
            'type': '特征提取不充分',
            'symptom': f'聚焦分数过低: {focus_score:.3f} (应该 > 0.5)',
            'severity': 'P1'
        })
        recommendations.append({
            'priority': 'P1',
            'action': '检查多尺度特征提取（4x4、8x8、16x16 窗口）',
            'code': 'verify_multiscale_features(scales=[4, 8, 16])'
        })
        recommendations.append({
            'priority': 'P1',
            'action': '增加部分特征损失，鼓励学习身体各部位特征',
            'code': 'loss_part = part_based_loss(features, labels)'
        })
    
    # 4. 检查特征学习有效性
    human_response = heatmap_metrics.get('human_response', 0.0)
    if human_response < 0.4:
        issues.append({
            'type': '特征学习不稳定',
            'symptom': f'人体响应过低: {human_response:.3f} (应该 > 0.6)',
            'severity': 'P1'
        })
        recommendations.append({
            'priority': 'P1',
            'action': '检查模型架构和训练策略',
            'code': 'review_model_architecture_and_training()'
        })
        recommendations.append({
            'priority': 'P1',
            'action': '延长训练时间或增加模型容量',
            'code': 'increase_training_epochs() or increase_model_capacity()'
        })
    
    return {
        'issues': issues,
        'severity': severity,
        'recommendations': recommendations,
        'summary': _generate_diagnosis_summary(issues, severity)
    }


def _generate_diagnosis_summary(issues: List[Dict], severity: str) -> str:
    """
    生成诊断摘要
    """
    if not issues:
        return "✅ 未发现明显问题，模型性能优秀"
    
    summary = f"发现 {len(issues)} 个问题（严重程度: {severity}）:\n"
    for i, issue in enumerate(issues, 1):
        summary += f"  {i}. [{issue['severity']}] {issue['type']}: {issue['symptom']}\n"
    
    return summary


def compute_heatmap_alignment(
    heatmap_rgb: np.ndarray,
    heatmap_nir: np.ndarray,
    heatmap_tir: np.ndarray
) -> float:
    """
    计算热力图的空间对齐度（便捷函数）
    
    这是 evaluate_cross_modal_alignment 的简化版本，只返回对齐分数。
    
    Args:
        heatmap_rgb (np.ndarray): RGB 模态热力图
        heatmap_nir (np.ndarray): NIR 模态热力图
        heatmap_tir (np.ndarray): TIR 模态热力图
    
    Returns:
        float: 对齐分数（0-1）
    """
    result = evaluate_cross_modal_alignment(heatmap_rgb, heatmap_nir, heatmap_tir)
    return result['alignment_score']


def create_simple_human_mask(
    image_shape: Tuple[int, int],
    center_ratio: Tuple[float, float] = (0.5, 0.4),
    size_ratio: Tuple[float, float] = (0.3, 0.6)
) -> np.ndarray:
    """
    创建简单的人体区域掩码（用于测试）
    
    功能说明：
    这是一个简化的掩码生成函数，用于测试评估工具。
    实际使用时，应该使用更精确的人体分割方法（如 DeepLab、Mask R-CNN 等）。
    
    Args:
        image_shape (tuple): 图像尺寸 (H, W)
        center_ratio (tuple): 人体中心位置比例 (x_ratio, y_ratio)
        size_ratio (tuple): 人体区域大小比例 (width_ratio, height_ratio)
    
    Returns:
        np.ndarray: 人体区域掩码，形状为 [H, W]（布尔数组）
    """
    H, W = image_shape
    center_x = int(W * center_ratio[0])
    center_y = int(H * center_ratio[1])
    width = int(W * size_ratio[0])
    height = int(H * size_ratio[1])
    
    mask = np.zeros((H, W), dtype=bool)
    x1 = max(0, center_x - width // 2)
    x2 = min(W, center_x + width // 2)
    y1 = max(0, center_y - height // 2)
    y2 = min(H, center_y + height // 2)
    
    mask[y1:y2, x1:x2] = True
    return mask


if __name__ == '__main__':
    """
    测试代码
    """
    print("="*60)
    print("热力图评估工具测试")
    print("="*60)
    
    # 创建示例热力图
    H, W = 256, 128
    heatmap = np.random.rand(H, W)
    
    # 创建简单的人体和背景掩码
    human_mask = create_simple_human_mask((H, W))
    background_mask = ~human_mask
    
    # 评估热力图质量
    print("\n1. 评估热力图质量:")
    metrics = evaluate_heatmap_quality(heatmap, human_mask, background_mask)
    for key, value in metrics.items():
        if key != 'quality_level':
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # 评估跨模态对齐
    print("\n2. 评估跨模态对齐:")
    heatmap_rgb = np.random.rand(H, W)
    heatmap_nir = np.random.rand(H, W)
    heatmap_tir = np.random.rand(H, W)
    alignment = evaluate_cross_modal_alignment(heatmap_rgb, heatmap_nir, heatmap_tir)
    for key, value in alignment.items():
        if key != 'alignment_level':
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # 诊断模型问题
    print("\n3. 诊断模型问题:")
    diagnosis = diagnose_model_issues(metrics, alignment)
    print(diagnosis['summary'])
    print(f"\n改进建议数量: {len(diagnosis['recommendations'])}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
