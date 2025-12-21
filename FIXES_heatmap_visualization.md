# 热力图可视化脚本修复说明

## 🔧 修复的问题

### 1. **GradCAM 类不支持字典输入**

**问题**：
- `GradCAM.generate_cam()` 和 `generate_gradcam()` 方法只接受 `torch.Tensor` 输入
- 但模型的前向传播需要字典输入 `{'RGB': ..., 'NI': ..., 'TI': ...}`
- 导致调用失败

**修复**：
- 修改了 `grad_cam.py` 中的 `generate_cam()` 方法，支持字典输入
- 添加了 `cam_label` 和 `view_label` 参数支持
- 修改了 `generate_gradcam()` 方法，传递这些参数

**修改的文件**：
- `/home/zhanghaoyang/Desktop/yzy/AboutReid/grad_cam.py`

### 2. **缺少 conda 环境激活**

**问题**：
- 直接运行 Python 脚本时，没有激活 conda 环境
- 导致 `ModuleNotFoundError: No module named 'torch'`

**修复**：
- 创建了 `run_heatmap_visualization.sh` shell 脚本
- 脚本自动激活 `MambaPro` conda 环境
- 然后运行 Python 脚本

**创建的文件**：
- `/home/zhanghaoyang/Desktop/yzy/AboutReid/run_heatmap_visualization.sh`

### 3. **脚本中缺少 cam_label 和 view_label**

**问题**：
- `generate_heatmap_visualization.py` 调用 `generate_gradcam()` 时没有传递必需的标签参数

**修复**：
- 在脚本中添加了 `cam_label` 和 `view_label` 的创建和传递

**修改的文件**：
- `/home/zhanghaoyang/Desktop/yzy/AboutReid/generate_heatmap_visualization.py`

## 🚀 使用方法

### 方法1：使用 shell 脚本（推荐）

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid
bash run_heatmap_visualization.sh
```

### 方法2：手动运行

```bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate MambaPro

# 切换到脚本目录
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

# 运行脚本
python generate_heatmap_visualization.py \
    --weight_path /home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt \
    --config_file /home/zhanghaoyang/Desktop/yzy/MambaPro/configs/RGBNT201/MambaPro.yml \
    --query_id 000274 \
    --dataset_root /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201 \
    --output_path heatmap_000274.png
```

## 📝 修改详情

### grad_cam.py 修改

1. **`generate_cam()` 方法签名更新**：
   ```python
   def generate_cam(
       self, 
       input_tensor: Union[torch.Tensor, dict],  # 支持字典输入
       target_class: Optional[int] = None,
       retain_graph: bool = False,
       cam_label: Optional[torch.Tensor] = None,  # 新增
       view_label: Optional[torch.Tensor] = None   # 新增
   ) -> np.ndarray:
   ```

2. **前向传播逻辑更新**：
   ```python
   # 支持字典输入
   if isinstance(input_tensor, dict):
       if cam_label is not None and view_label is not None:
           output = self.model(input_tensor, cam_label=cam_label, view_label=view_label)
       else:
           # 使用默认值
           device = next(self.model.parameters()).device
           cam_label = torch.tensor([0]).to(device) if cam_label is None else cam_label
           view_label = torch.tensor([0]).to(device) if view_label is None else view_label
           output = self.model(input_tensor, cam_label=cam_label, view_label=view_label)
   else:
       output = self.model(input_tensor)
   ```

3. **`generate_gradcam()` 方法更新**：
   - 添加了 `cam_label` 和 `view_label` 参数
   - 将这些参数传递给 `generate_cam()`

### generate_heatmap_visualization.py 修改

在生成热力图之前添加了标签创建：

```python
# 准备标签
cam_label = torch.tensor([0]).to(device)
view_label = torch.tensor([0]).to(device)

# 生成热力图
heatmap, overlay = gradcam.generate_gradcam(
    input_dict, original_image, target_class=None, alpha=alpha,
    cam_label=cam_label, view_label=view_label
)
```

## ✅ 验证

修复后，脚本应该能够：
1. ✅ 正确加载模型权重
2. ✅ 处理多模态输入（RGB、NIR、TIR）
3. ✅ 生成 Grad-CAM 热力图
4. ✅ 保存可视化结果到指定路径

## 📋 注意事项

1. **环境要求**：
   - 需要激活 `MambaPro` conda 环境
   - 确保安装了所有必需的依赖（torch, numpy, cv2, matplotlib 等）

2. **权重文件**：
   - 确保权重文件路径正确
   - 权重文件应该是训练好的模型权重（.pth 或 .pt 文件）

3. **配置文件**：
   - 确保配置文件路径正确
   - 配置文件应该与训练时使用的配置一致

4. **数据集路径**：
   - 确保数据集目录结构正确：`{dataset_root}/test/RGB/`, `{dataset_root}/test/NI/`, `{dataset_root}/test/TI/`
   - 图像文件命名格式：`{query_id}_*.jpg`
