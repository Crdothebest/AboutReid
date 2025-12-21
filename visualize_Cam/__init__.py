"""
可视化 CAM 工具包

包含 Grad-CAM 和 EigenCAM 热力图生成工具
"""

from .grad_cam import GradCAM, EigenCAM, find_target_layers
from .visualize_gradcam import (
    build_transforms,
    load_image,
    detect_camera_num_from_weights,
    get_target_layer_name
)

__all__ = [
    'GradCAM',
    'EigenCAM',
    'find_target_layers',
    'build_transforms',
    'load_image',
    'detect_camera_num_from_weights',
    'get_target_layer_name',
]



