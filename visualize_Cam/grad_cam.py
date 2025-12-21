#!/usr/bin/env python
"""
Grad-CAM 和 EigenCAM 热力图生成工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional, Tuple, Union
import warnings


class GradCAM:
    """Grad-CAM 热力图生成类"""
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: Union[str, nn.Module],
        use_cuda: bool = True
    ):
        """初始化 Grad-CAM"""
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.target_layer = self._get_target_layer(target_layer)
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _get_target_layer(self, target_layer: Union[str, nn.Module]) -> nn.Module:
        """获取目标层对象"""
        if isinstance(target_layer, nn.Module):
            return target_layer
        
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
        """注册前向和反向传播钩子"""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
            else:
                warnings.warn(f"目标层 {module} 的梯度为 None，可能该层未参与反向传播")
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self, 
        input_tensor: Union[torch.Tensor, dict],
        target_class: Optional[int] = None,
        retain_graph: bool = False,
        cam_label: Optional[torch.Tensor] = None,
        view_label: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """生成 Grad-CAM 热力图"""
        self.gradients = None
        self.activations = None
        
        if isinstance(input_tensor, dict):
            if cam_label is not None and view_label is not None:
                output = self.model(input_tensor, cam_label=cam_label, view_label=view_label)
            else:
                device = next(self.model.parameters()).device
                cam_label = torch.tensor([0]).to(device) if cam_label is None else cam_label
                view_label = torch.tensor([0]).to(device) if view_label is None else view_label
                output = self.model(input_tensor, cam_label=cam_label, view_label=view_label)
        else:
            output = self.model(input_tensor)
        
        if self.activations is None:
            raise RuntimeError(
                "未捕获到目标层的激活值。"
                "请检查目标层路径是否正确，或目标层是否在前向传播中被调用。"
            )
        
        if target_class is None:
            if isinstance(output, (list, tuple)):
                target_class = output[0].argmax(dim=1)
            else:
                target_class = output.argmax(dim=1)
        
        if isinstance(target_class, torch.Tensor):
            target_class = target_class[0].item() if target_class.dim() > 0 else target_class.item()
        
        if isinstance(output, (list, tuple)):
            target_score = output[0][0, target_class]
        else:
            target_score = output[0, target_class]
        
        self.model.zero_grad()
        target_score.backward(retain_graph=retain_graph)
        
        if self.gradients is None:
            raise RuntimeError(
                "未捕获到目标层的梯度。"
                "可能原因：目标层未参与反向传播，或梯度被截断。"
            )
        
        if self.activations.dim() == 4:
            activations = self.activations[0]
            gradients = self.gradients[0]
            weights = torch.mean(gradients, dim=(1, 2))
            cam = torch.zeros(activations.shape[1:], device=activations.device)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
        elif self.activations.dim() == 3:
            B, N, D = self.activations.shape
            h = int(np.sqrt(N))
            w = N // h
            
            if h * w != N:
                h = 16
                w = 8
            
            activations = self.activations[0].permute(1, 0).reshape(D, h, w)
            gradients = self.gradients[0].permute(1, 0).reshape(D, h, w)
            weights = torch.mean(gradients, dim=(1, 2))
            cam = torch.zeros((h, w), device=activations.device)
            for i, w in enumerate(weights):
                cam += w * activations[i]
        else:
            raise ValueError(
                f"不支持的特征图维度: {self.activations.dim()}\n"
                f"期望维度: 3 (Transformer) 或 4 (CNN)，实际维度: {self.activations.dim()}"
            )
        
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        return cam
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """将热力图叠加到原始图像上"""
        if heatmap.shape != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        overlay = (alpha * colored_heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlay
    
    def generate_gradcam(
        self,
        input_tensor: Union[torch.Tensor, dict],
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        cam_label: Optional[torch.Tensor] = None,
        view_label: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """一键生成 Grad-CAM 热力图和叠加图像"""
        heatmap_raw = self.generate_cam(
            input_tensor, 
            target_class=target_class,
            cam_label=cam_label,
            view_label=view_label
        )
        
        heatmap_min = heatmap_raw.min()
        heatmap_max = heatmap_raw.max()
        if heatmap_max > heatmap_min:
            heatmap = (heatmap_raw - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap_raw)
        
        overlay = self.overlay_heatmap(original_image, heatmap, alpha=alpha)
        
        return heatmap, overlay


class EigenCAM:
    """EigenCAM 热力图生成类，使用 PCA 计算主成分，不需要梯度"""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module],
        use_cuda: bool = True,
        reshape_transform: Optional[callable] = None
    ):
        """初始化 EigenCAM"""
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.reshape_transform = reshape_transform
        self.target_layer = self._get_target_layer(target_layer)
        self.activations = None
        
        self._register_hooks()
    
    def _get_target_layer(self, target_layer: Union[str, nn.Module]) -> nn.Module:
        """获取目标层对象"""
        if isinstance(target_layer, nn.Module):
            return target_layer
        
        parts = target_layer.split('.')
        layer = self.model
        
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                raise ValueError(f"找不到目标层: {target_layer}")
        
        return layer
    
    def _register_hooks(self):
        """注册前向传播钩子"""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
    
    def generate_cam(
        self,
        input_tensor: Union[torch.Tensor, dict],
        cam_label: Optional[torch.Tensor] = None,
        view_label: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """生成 EigenCAM 热力图，使用 PCA 找到最重要的特征方向"""
        self.activations = None
        
        if isinstance(input_tensor, dict):
            device = next(self.model.parameters()).device
            cam_label = torch.tensor([0]).to(device) if cam_label is None else cam_label
            view_label = torch.tensor([0]).to(device) if view_label is None else view_label
            output = self.model(input_tensor, cam_label=cam_label, view_label=view_label)
        else:
            output = self.model(input_tensor)
        
        if self.activations is None:
            raise RuntimeError("未捕获到目标层的激活值")
        
        if self.reshape_transform is not None:
            activations = self.reshape_transform(self.activations)
        else:
            activations = self.activations
        
        if activations.dim() == 4:
            activations = activations[0]
            C, H, W = activations.shape[0], activations.shape[1], activations.shape[2]
        elif activations.dim() == 3:
            B, N, D = activations.shape
            h = 16
            w = 8
            expected_patches = h * w
            
            if N == expected_patches + 1:
                activations_no_cls = activations[:, 1:, :]
            else:
                activations_no_cls = activations
            
            activations_permuted = activations_no_cls[0].permute(1, 0)
            if activations_permuted.numel() != D * h * w:
                total_patches = activations_permuted.shape[1]
                h = int(np.sqrt(total_patches))
                w = total_patches // h
                if h * w != total_patches:
                    h, w = 16, 8
                    if activations_permuted.shape[1] > h * w:
                        activations_permuted = activations_permuted[:, :h*w]
                    else:
                        padding = torch.zeros(D, h*w - activations_permuted.shape[1], 
                                            device=activations_permuted.device)
                        activations_permuted = torch.cat([activations_permuted, padding], dim=1)
            
            activations = activations_permuted.reshape(D, h, w)
            C, H, W = D, h, w
        else:
            raise ValueError(f"不支持的特征图维度: {activations.dim()}")
        
        features = activations.view(C, H * W)
        features_mean = features.mean(dim=1, keepdim=True)
        features_centered = features - features_mean
        covariance = torch.matmul(features_centered, features_centered.t()) / (H * W - 1)
        
        try:
            U, S, V = torch.svd(covariance)
            principal_component = U[:, 0]
        except:
            principal_component = features_mean.squeeze(1)
        
        cam = torch.matmul(principal_component.unsqueeze(0), features)
        cam = cam.squeeze(0)
        cam = cam.view(H, W)
        
        cam = torch.abs(cam)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # 论文级后处理：增强红色区域，提升对比度，减少青色
        # 1. 鲁棒归一化：使用更高的百分位数，让更多区域变红
        v_min = cam.min()
        v_max = np.percentile(cam, 95)  # 降低到 95 百分位数，让更多高激活值达到 1.0
        if v_max > v_min:
            cam = (cam - v_min) / (v_max - v_min + 1e-8)
        else:
            cam = np.zeros_like(cam)
        cam = np.clip(cam, 0, 1)
        
        # 2. 提高阈值截断，过滤掉中等激活值（这些会产生青色）
        # 将阈值提高到 0.15，只保留最强的激活区域
        cam[cam < 0.15] = 0  # 过滤掉会产生青色的中等值
        
        # 3. 关键修改：进一步提高幂次，压缩中等值
        # 使用 6.0 或更高，让中等值（0.4-0.6）被压缩到接近 0
        # 原理：0.4^6 ≈ 0.004 (几乎为0，变深蓝)，0.6^6 ≈ 0.047 (变蓝)，1.0^6 = 1.0 (依然红)
        # 这会强制让只有"最强"的点保持红色，中等值被压缩到蓝色
        cam = np.power(cam, 6.0)
        
        return cam
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        论文级渲染：使用动态掩码叠加，确保背景纯净
        实现"只有重点区域亮起，背景完全保持原图色调"的效果
        """
        # 1. 减小模糊半径，让边界更清晰，不至于漫散
        if heatmap.shape != original_image.shape[:2]:
            heatmap_res = cv2.resize(
                heatmap, 
                (original_image.shape[1], original_image.shape[0]), 
                interpolation=cv2.INTER_CUBIC
            )
        else:
            heatmap_res = heatmap.copy()
        
        # 将 (25, 25) 缩小为 (11, 11)，防止颜色漫散
        heatmap_smoothed = cv2.GaussianBlur(heatmap_res, (11, 11), 0)

        # 2. 生成彩色图
        heatmap_uint8 = np.uint8(255 * heatmap_smoothed)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # 3. 论文级叠加逻辑：底色保蓝 + 核心聚拢
        mask = heatmap_smoothed[:, :, np.newaxis]
        
        # 让 Alpha 的增长变得更"陡峭"
        # 基础 alpha 提高到 0.3 (保证背景有足够的蓝色)，随着 mask 增加，alpha 迅速增加
        dynamic_alpha = 0.3 + (alpha - 0.3) * np.power(mask, 2.0) 
        
        overlay = original_image.astype(np.float32) * (1 - dynamic_alpha) + \
                  colored_heatmap.astype(np.float32) * dynamic_alpha

        return np.clip(overlay, 0, 255).astype(np.uint8)


def find_target_layers(model: nn.Module, layer_type: type = nn.Conv2d) -> List[Tuple[str, nn.Module]]:
    """查找模型中指定类型的所有层"""
    layers = []
    
    def _find_layers(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, layer_type):
                layers.append((full_name, child))
            _find_layers(child, full_name)
    
    _find_layers(model)
    return layers
