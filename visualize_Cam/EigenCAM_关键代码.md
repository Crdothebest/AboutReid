# EigenCAM 热力图生成关键代码

本文档包含生成 `eigencam_000275.png` 的所有关键代码，用于基于效果进行进一步优化。

## 1. EigenCAM 核心生成逻辑 (`grad_cam.py`)

### 1.1 PCA 主成分提取

```python
def generate_cam(
    self,
    input_tensor: Union[torch.Tensor, dict],
    cam_label: Optional[torch.Tensor] = None,
    view_label: Optional[torch.Tensor] = None
) -> np.ndarray:
    """生成 EigenCAM 热力图，使用 PCA 找到最重要的特征方向"""
    # ... 前向传播获取激活值 ...
    
    # 将特征图 reshape 为 [C, H*W]
    features = activations.view(C, H * W)
    
    # 计算协方差矩阵
    features_mean = features.mean(dim=1, keepdim=True)
    features_centered = features - features_mean
    covariance = torch.matmul(features_centered, features_centered.t()) / (H * W - 1)
    
    # 计算主特征向量（使用 SVD）
    try:
        U, S, V = torch.svd(covariance)
        principal_component = U[:, 0]  # 第一个主成分
    except:
        principal_component = features_mean.squeeze(1)
    
    # 投影到主成分方向
    cam = torch.matmul(principal_component.unsqueeze(0), features)
    cam = cam.squeeze(0)
    cam = cam.view(H, W)
    
    # 取绝对值和 ReLU
    cam = torch.abs(cam)
    cam = F.relu(cam)
    cam = cam.cpu().numpy()
    
    # 🔥 论文级后处理（在 grad_cam.py 中）
    # 1. 最小最大归一化
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    else:
        cam = np.zeros_like(cam)
    
    # 2. 阈值处理：将低激活区域（背景噪声）直接设为 0
    cam[cam < 0.2] = 0  # ⚠️ 可调参数：阈值 0.2
    
    # 3. 幂次变换：让红色更红，蓝色更深，聚焦核心区域
    cam = np.power(cam, 2.0)  # ⚠️ 可调参数：gamma=2.0
    
    return cam
```

**关键参数：**
- `阈值`: 0.2 (第343行) - 控制背景噪声过滤
- `gamma`: 2.0 (第346行) - 控制对比度增强

---

## 2. 多模态热力图后处理 (`generate_heatmap_visualization.py`)

### 2.1 EigenCAM 美化处理流程

```python
# 在 generate_multimodal_heatmap() 函数中，针对 EigenCAM 的处理
if method.lower() == 'eigencam':
    # 输入：heatmap (来自 generate_cam，已经过基础处理)
    
    # 1. 先归一化：将当前模态缩放到 [0, 1]
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)
    heatmap = np.clip(heatmap, 0, 1)
    
    # 2. Gamma 校正：模态特定的 gamma 值
    if mod == 'RGB':
        gamma = 0.85  # ⚠️ 可调参数
    elif mod == 'NI':
        gamma = 0.85  # ⚠️ 可调参数
    else:  # TI
        gamma = 0.9   # ⚠️ 可调参数
    heatmap = np.power(heatmap, gamma)
    
    # 3. 阈值过滤：过滤低激活区域
    threshold_ratio = 0.1  # ⚠️ 可调参数：10% 阈值
    threshold_base = heatmap.max() * threshold_ratio
    heatmap[heatmap < threshold_base] = 0
    
    # 4. 边缘裁剪（当前已禁用）
    heatmap_edge_cleaned = heatmap.copy()
    
    # 5. 放大：使用双三次插值
    heatmap_resized = cv2.resize(
        heatmap_edge_cleaned,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_CUBIC  # ⚠️ 可调：INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
    )
    
    # 6. 高斯模糊：消除格点感
    heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (45, 45), 0)  # ⚠️ 可调参数：核大小 (45, 45)
    
    # 7. 重新归一化
    heatmap_max = heatmap_blurred.max()
    heatmap_min = heatmap_blurred.min()
    if heatmap_max > heatmap_min:
        heatmap_blurred = (heatmap_blurred - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    elif heatmap_max > 0:
        heatmap_blurred = heatmap_blurred / heatmap_max
    else:
        heatmap_blurred = np.zeros_like(heatmap_blurred)
    
    # 8. 全局亮度对比：模态间对比度调整
    if global_max > 0:
        if mod == 'TI':
            global_contrast = 1.0  # TIR 保持 100%
        else:
            mod_max = raw_heatmaps[mod].max()
            if mod_max >= global_max * 0.95:
                global_contrast = 0.65  # ⚠️ 可调参数
            else:
                global_contrast = min(0.75, 0.5 + (mod_max / global_max) * 0.25)  # ⚠️ 可调参数
        heatmap_blurred = heatmap_blurred * global_contrast
    
    # 9. 颜色映射
    heatmap_normalized = np.clip(heatmap_blurred, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # ⚠️ 可调：COLORMAP_HOT, COLORMAP_VIRIDIS
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 10. 叠加：使用 alpha 透明度
    overlay = (
        heatmap_colored * alpha + original_image.astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)  # ⚠️ 可调参数：alpha (默认 0.5)
```

**关键可调参数：**

| 参数 | 位置 | 当前值 | 说明 |
|------|------|--------|------|
| `gamma (RGB/NIR)` | 第267-269行 | 0.85 | 对比度拉伸，值越大越聚焦 |
| `gamma (TIR)` | 第271行 | 0.9 | TIR 模态的对比度 |
| `threshold_ratio` | 第276行 | 0.1 | 阈值过滤比例（10%） |
| `插值方法` | 第302行 | INTER_CUBIC | 放大插值方法 |
| `高斯核大小` | 第307行 | (45, 45) | 模糊核大小，越大越平滑 |
| `global_contrast (RGB/NIR)` | 第340行 | 0.65 | RGB/NIR 亮度缩放 |
| `alpha` | 第369行 | 0.5 | 热力图透明度 |
| `colormap` | 第363行 | COLORMAP_JET | 颜色映射方案 |

---

## 3. 叠加函数 (`grad_cam.py` - 当前未使用)

**注意：** `EigenCAM.overlay_heatmap()` 方法在当前实现中**未被调用**，实际使用的是 `generate_multimodal_heatmap()` 中的叠加逻辑（第368-370行）。

如果希望使用 `overlay_heatmap()` 的动态叠加逻辑，需要修改代码。

```python
def overlay_heatmap(
    self,
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """优化后的叠加函数：使用 CUBIC 插值、高斯模糊和动态叠加"""
    # 1. CUBIC 插值放大
    if heatmap.shape != original_image.shape[:2]:
        heatmap_resized = cv2.resize(
            heatmap, 
            (original_image.shape[1], original_image.shape[0]), 
            interpolation=cv2.INTER_CUBIC
        )
    else:
        heatmap_resized = heatmap.copy()
    
    # 2. 高斯模糊
    kernel_size = 21  # ⚠️ 可调参数
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (kernel_size, kernel_size), 0)
    
    # 3. 颜色映射
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # 4. 动态叠加（基于热力图值的 mask）
    heatmap_float = heatmap_resized[:, :, np.newaxis]
    overlay = heatmap_float * colored_heatmap + (1 - heatmap_float * alpha) * original_image
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
```

---

## 4. 优化建议

### 4.1 如果背景噪声过多
- **降低阈值**：`cam[cam < 0.2] = 0` → `cam[cam < 0.15] = 0` 或 `cam[cam < 0.1] = 0`
- **提高 gamma**：`np.power(cam, 2.0)` → `np.power(cam, 2.5)` 或 `np.power(cam, 3.0)`
- **增加高斯模糊**：`(45, 45)` → `(61, 61)` 或 `(81, 81)`

### 4.2 如果红色区域不够明显
- **降低 gamma**：`0.85` → `0.7` 或 `0.75`
- **降低阈值过滤**：`threshold_ratio = 0.1` → `0.05` 或 `0.0`
- **提高 alpha**：`0.5` → `0.6` 或 `0.7`

### 4.3 如果边缘不够平滑
- **使用更好的插值**：`INTER_CUBIC` → `INTER_LANCZOS4`
- **增加高斯模糊核**：`(45, 45)` → `(61, 61)`

### 4.4 如果模态间对比度不够
- **调整 global_contrast**：RGB/NIR 的 `0.65` → `0.5` 或 `0.55`（让 TIR 更突出）
- **调整 TIR 的 gamma**：`0.9` → `0.85`（让 TIR 更聚焦）

---

## 5. 文件位置

- **EigenCAM 核心逻辑**：`visualize_Cam/grad_cam.py` 第256-348行
- **多模态后处理**：`visualize_Cam/generate_heatmap_visualization.py` 第245-370行
- **叠加函数**：`visualize_Cam/grad_cam.py` 第350-384行（当前未使用）

---

## 6. 调试输出

代码中包含了详细的调试输出，可以通过以下信息判断效果：

```
🔍 {mod_name} 原始热力图: min=..., max=..., mean=..., non_zero=...
🔍 {mod_name} 归一化后: min=..., max=..., mean=..., >0.5=..., >0.8=...
🔍 {mod_name} 阈值过滤后: min=..., max=..., >0.5=...
🔍 {mod_name} 模糊后: min=..., max=..., >0.5=...
🔍 {mod_name} 重新归一化后: min=..., max=..., >0.5=...
🔍 {mod_name} 全局对比系数: ...
🔍 {mod_name} 颜色映射前: min=..., max=..., mean=..., >0.5=..., >0.8=...
🔍 {mod_name} 高值像素(>200): .../... (...%)
```

通过这些输出可以判断：
- **高值像素比例**：如果 >200 的像素很少，说明红色区域不够明显
- **>0.5 的像素数**：判断热力图的覆盖范围
- **全局对比系数**：判断模态间的亮度差异
