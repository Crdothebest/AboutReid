#!/usr/bin/env python
"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 热力图生成工具

功能说明：
Grad-CAM 是一种可视化技术，用于展示深度神经网络在做出判断时关注图像的哪些区域。
通过计算目标输出对卷积层特征图的梯度，生成热力图，红色/暖色区域表示高响应（模型关注度高），
蓝色/冷色区域表示低响应（模型关注度低）。

核心原理：
1. 前向传播：将图像输入模型，获取目标层的特征图
2. 反向传播：计算目标输出（如分类得分）对特征图的梯度
3. 权重计算：对梯度进行全局平均池化，得到每个通道的重要性权重
4. 热力图生成：将权重与特征图加权求和，得到热力图
5. 可视化：将热力图叠加到原始图像上

适用场景：
- 模型可解释性分析：理解模型关注哪些区域
- 模型调试：检查模型是否关注了正确的区域（如人体而非背景）
- 论文可视化：展示模型的可解释性，增强论文说服力

作者：MambaPro团队
日期：2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional, Tuple, Union
import warnings


class GradCAM:
    """
    Grad-CAM 热力图生成类
    
    功能：
    - 自动注册目标层的梯度钩子（hook）
    - 计算梯度加权类激活映射
    - 生成热力图并叠加到原始图像上
    
    使用示例：
        >>> model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
        >>> model.eval()
        >>> gradcam = GradCAM(model, target_layer='BACKBONE.image_encoder.transformer.resblocks.11')
        >>> heatmap = gradcam.generate_cam(image_tensor, target_class=None)
        >>> overlay = gradcam.overlay_heatmap(original_image, heatmap)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: Union[str, nn.Module],
        use_cuda: bool = True
    ):
        """
        初始化 Grad-CAM
        
        Args:
            model (nn.Module): 要分析的模型（必须是训练好的模型）
            target_layer (str or nn.Module): 目标层名称或层对象
                - 字符串格式：如 'BACKBONE.image_encoder.transformer.resblocks.11'
                - 层对象：直接传入 nn.Module 对象
                - 注意：目标层应该是卷积层或 Transformer 的最后一层
            use_cuda (bool): 是否使用 GPU，默认 True
        
        说明：
            - 目标层选择：通常选择模型的最后一层卷积层或 Transformer 的最后一层
            - 对于 CLIP ViT：可以选择 'BACKBONE.image_encoder.transformer.resblocks.11'（最后一层）
            - 对于 ResNet：可以选择 'BACKBONE.base.layer4'（最后一层卷积）
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 注册目标层
        self.target_layer = self._get_target_layer(target_layer)
        self.gradients = None  # 存储梯度
        self.activations = None  # 存储激活值
        
        # 注册前向和反向钩子
        self._register_hooks()
    
    def _get_target_layer(self, target_layer: Union[str, nn.Module]) -> nn.Module:
        """
        获取目标层对象
        
        Args:
            target_layer: 层名称（字符串）或层对象
            
        Returns:
            nn.Module: 目标层对象
            
        Raises:
            ValueError: 如果找不到指定的层
        """
        if isinstance(target_layer, nn.Module):
            # 如果直接传入层对象，直接返回
            return target_layer
        
        # 如果是字符串，按路径查找
        parts = target_layer.split('.')
        layer = self.model
        
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                raise ValueError(
                    f"找不到目标层: {target_layer}\n"
                    f"在 '{'.'.join(parts[:parts.index(part)])}' 中找不到 '{part}'\n"
                    f"可用属性: {dir(layer) if hasattr(layer, '__dict__') else 'N/A'}"
                )
        
        return layer
    
    def _register_hooks(self):
        """
        注册前向和反向传播钩子
        
        功能：
        - 前向钩子：捕获目标层的激活值（特征图）
        - 反向钩子：捕获目标层的梯度
        
        说明：
        - 钩子（hook）是 PyTorch 提供的机制，可以在前向/反向传播时执行自定义函数
        - 前向钩子：在 forward 时执行，用于保存激活值
        - 反向钩子：在 backward 时执行，用于保存梯度
        """
        def forward_hook(module, input, output):
            """
            前向传播钩子：保存激活值（特征图）
            
            Args:
                module: 目标层模块
                input: 输入（元组）
                output: 输出（特征图）
            """
            # 保存激活值（特征图）
            # 注意：如果 output 是 tuple，取第一个元素
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """
            反向传播钩子：保存梯度
            
            Args:
                module: 目标层模块
                grad_input: 输入梯度（通常不使用）
                grad_output: 输出梯度（元组，取第一个元素）
            """
            # 保存梯度
            # grad_output 是元组，取第一个元素（对应输出的梯度）
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
            else:
                warnings.warn(f"目标层 {module} 的梯度为 None，可能该层未参与反向传播")
        
        # 注册钩子
        # register_forward_hook: 在前向传播时调用 forward_hook
        # register_full_backward_hook: 在反向传播时调用 backward_hook
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        retain_graph: bool = False
    ) -> np.ndarray:
        """
        生成 Grad-CAM 热力图
        
        功能说明：
        1. 前向传播：获取目标层的特征图
        2. 反向传播：计算目标输出对特征图的梯度
        3. 权重计算：对梯度进行全局平均池化，得到通道权重
        4. 热力图生成：将权重与特征图加权求和
        
        算法流程：
        - 特征图形状：[B, C, H, W] 或 [B, N, D]（Transformer）
        - 梯度形状：与特征图相同
        - 权重计算：alpha_c = mean(gradient_c)  # 对空间维度求平均
        - 热力图：CAM = sum(alpha_c * feature_map_c)  # 加权求和
        
        Args:
            input_tensor (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]
            target_class (int, optional): 目标类别索引
                - None: 使用模型预测的类别（最大得分对应的类别）
                - int: 使用指定的类别（用于分析特定类别的关注区域）
            retain_graph (bool): 是否保留计算图，默认 False
        
        Returns:
            np.ndarray: 热力图，形状为 [H, W]（单通道，值域 [0, 1]）
        
        示例:
            >>> image_tensor = torch.randn(1, 3, 256, 128).to(device)
            >>> heatmap = gradcam.generate_cam(image_tensor, target_class=None)
            >>> # heatmap 形状: (256, 128)，值域 [0, 1]
        """
        # 清空之前的梯度和激活值
        self.gradients = None
        self.activations = None
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 检查激活值和输出
        if self.activations is None:
            raise RuntimeError(
                "未捕获到目标层的激活值。"
                "请检查目标层路径是否正确，或目标层是否在前向传播中被调用。"
            )
        
        # 确定目标类别
        if target_class is None:
            # 如果未指定目标类别，使用模型预测的类别（最大得分）
            if isinstance(output, (list, tuple)):
                # 如果输出是列表/元组（多分支模型），取第一个分支
                target_class = output[0].argmax(dim=1)
            else:
                # 如果输出是单个张量
                target_class = output.argmax(dim=1)
        
        # 如果 target_class 是张量，取第一个元素（batch 中的第一个样本）
        if isinstance(target_class, torch.Tensor):
            target_class = target_class[0].item() if target_class.dim() > 0 else target_class.item()
        
        # 获取目标类别的得分
        if isinstance(output, (list, tuple)):
            target_score = output[0][0, target_class]  # [0] 取第一个分支，[0] 取第一个样本
        else:
            target_score = output[0, target_class]  # [0] 取第一个样本
        
        # 反向传播：计算梯度
        self.model.zero_grad()  # 清零梯度
        target_score.backward(retain_graph=retain_graph)  # 反向传播
        
        # 检查梯度
        if self.gradients is None:
            raise RuntimeError(
                "未捕获到目标层的梯度。"
                "可能原因：目标层未参与反向传播，或梯度被截断。"
            )
        
        # 处理不同的特征图格式
        # 情况1：卷积层特征图 [B, C, H, W]
        # 情况2：Transformer 特征图 [B, N, D]（需要 reshape）
        if self.activations.dim() == 4:
            # 卷积层：[B, C, H, W]
            activations = self.activations[0]  # 取第一个样本：[C, H, W]
            gradients = self.gradients[0]      # 取第一个样本：[C, H, W]
            
            # 计算通道权重：对空间维度求平均
            # weights: [C] = mean(gradients, dim=(1, 2))
            weights = torch.mean(gradients, dim=(1, 2))  # 全局平均池化
            
            # 生成热力图：加权求和
            # cam: [H, W] = sum(weights[c] * activations[c, :, :])
            cam = torch.zeros(activations.shape[1:], device=activations.device)  # [H, W]
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
        elif self.activations.dim() == 3:
            # Transformer：[B, N, D] 或 [B, N, C]
            # 需要 reshape 为 [B, C, H, W] 格式才能生成空间热力图
            B, N, D = self.activations.shape
            
            # 假设特征图可以 reshape 为 2D 空间（如 16×8 = 128）
            # 对于 CLIP ViT-B-16，patch 数量通常是 16×8 = 128（256×128 输入）
            # 这里需要根据实际情况调整
            h = int(np.sqrt(N))  # 尝试正方形
            w = N // h
            
            if h * w != N:
                # 如果不是完全平方数，尝试其他组合
                # 对于 256×128 输入，patch 数量是 16×8 = 128
                # 可以根据输入尺寸计算：h = H/patch_size, w = W/patch_size
                h = 16  # 假设高度方向有 16 个 patch
                w = 8   # 假设宽度方向有 8 个 patch
            
            # Reshape 为 [B, D, H, W]
            activations = self.activations[0].permute(1, 0).reshape(D, h, w)  # [D, H, W]
            gradients = self.gradients[0].permute(1, 0).reshape(D, h, w)    # [D, H, W]
            
            # 计算通道权重
            weights = torch.mean(gradients, dim=(1, 2))  # [D]
            
            # 生成热力图
            cam = torch.zeros((h, w), device=activations.device)  # [H, W]
            for i, w in enumerate(weights):
                cam += w * activations[i]
        else:
            raise ValueError(
                f"不支持的特征图维度: {self.activations.dim()}\n"
                f"期望维度: 3 (Transformer) 或 4 (CNN)，实际维度: {self.activations.dim()}"
            )
        
        # 应用 ReLU：只保留正向激活（关注正向贡献）
        cam = F.relu(cam)
        
        # 归一化到 [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        # 转换为 numpy 数组
        cam = cam.cpu().numpy()
        
        return cam
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        将热力图叠加到原始图像上
        
        功能说明：
        - 将热力图（单通道，值域 [0, 1]）转换为彩色热力图（使用颜色映射）
        - 将彩色热力图与原始图像叠加（透明度混合）
        - 红色/暖色表示高响应区域，蓝色/冷色表示低响应区域
        
        Args:
            original_image (np.ndarray): 原始图像，形状为 [H, W, 3]（RGB 格式，值域 [0, 255]）
            heatmap (np.ndarray): 热力图，形状为 [H, W]（值域 [0, 1]）
            alpha (float): 热力图透明度，范围 [0, 1]
                - 0.0: 完全透明（只显示原始图像）
                - 1.0: 完全不透明（热力图完全覆盖）
                - 0.4: 推荐值，40% 热力图 + 60% 原始图像
            colormap (int): OpenCV 颜色映射类型
                - cv2.COLORMAP_JET: 蓝-绿-黄-红（默认，常用）
                - cv2.COLORMAP_HOT: 黑-红-黄-白
                - cv2.COLORMAP_VIRIDIS: 紫-蓝-绿-黄（对色盲友好）
        
        Returns:
            np.ndarray: 叠加后的图像，形状为 [H, W, 3]（RGB 格式，值域 [0, 255]）
        
        示例:
            >>> original = cv2.imread('image.jpg')  # BGR 格式
            >>> original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  # 转为 RGB
            >>> overlay = gradcam.overlay_heatmap(original_rgb, heatmap, alpha=0.4)
        """
        # 确保热力图和图像尺寸匹配
        if heatmap.shape != original_image.shape[:2]:
            # 如果尺寸不匹配，将热力图 resize 到图像尺寸
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # 将热力图转换为 8 位整数（值域 [0, 255]）
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        # 应用颜色映射：将单通道热力图转换为 3 通道彩色热力图
        # COLORMAP_JET: 蓝色(低值) -> 绿色 -> 黄色 -> 红色(高值)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # 转换颜色格式：BGR -> RGB（OpenCV 使用 BGR，matplotlib 使用 RGB）
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加：alpha 混合
        # overlay = alpha * heatmap + (1 - alpha) * original
        overlay = (alpha * colored_heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlay
    
    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        一键生成 Grad-CAM 热力图和叠加图像
        
        功能说明：
        这是 generate_cam() 和 overlay_heatmap() 的便捷封装，
        一次性生成热力图和叠加图像。
        
        Args:
            input_tensor (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]
            original_image (np.ndarray): 原始图像，形状为 [H, W, 3]（RGB，值域 [0, 255]）
            target_class (int, optional): 目标类别索引，None 表示使用预测类别
            alpha (float): 热力图透明度，默认 0.4
        
        Returns:
            tuple: (heatmap, overlay)
                - heatmap: 热力图，形状为 [H, W]（值域 [0, 1]）
                - overlay: 叠加图像，形状为 [H, W, 3]（RGB，值域 [0, 255]）
        
        示例:
            >>> heatmap, overlay = gradcam.generate_gradcam(
            ...     image_tensor, original_image, target_class=None, alpha=0.4
            ... )
        """
        # 生成热力图
        heatmap = self.generate_cam(input_tensor, target_class=target_class)
        
        # 叠加到原始图像
        overlay = self.overlay_heatmap(original_image, heatmap, alpha=alpha)
        
        return heatmap, overlay


def find_target_layers(model: nn.Module, layer_type: type = nn.Conv2d) -> List[Tuple[str, nn.Module]]:
    """
    查找模型中指定类型的所有层
    
    功能说明：
    自动遍历模型的所有层，找出指定类型（如 Conv2d、Transformer 层）的层，
    用于帮助用户选择合适的目标层进行 Grad-CAM 分析。
    
    Args:
        model (nn.Module): 要搜索的模型
        layer_type (type): 要查找的层类型，如 nn.Conv2d、nn.Linear 等
    
    Returns:
        List[Tuple[str, nn.Module]]: 找到的层列表，每个元素为 (层路径, 层对象)
    
    示例:
        >>> # 查找所有卷积层
        >>> conv_layers = find_target_layers(model, nn.Conv2d)
        >>> for name, layer in conv_layers:
        ...     print(f"{name}: {layer}")
        >>> 
        >>> # 查找所有 Transformer 层
        >>> transformer_layers = find_target_layers(model, type(model.BACKBONE.image_encoder.transformer.resblocks[0]))
    """
    layers = []
    
    def _find_layers(module, prefix=""):
        """递归查找层"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, layer_type):
                layers.append((full_name, child))
            _find_layers(child, full_name)
    
    _find_layers(model)
    return layers
