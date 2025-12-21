# TorchScript 模型支持修复说明

## 🔧 修复的问题

### 问题：TorchScript 模型（.pt 文件）无法加载

**错误信息**：
```
NotImplementedError
File "visualize_gradcam.py", line 154, in detect_camera_num_from_weights
    for key in checkpoint:
```

**原因**：
- `ViT-B-16.pt` 是一个 TorchScript 模型（编译后的模型）
- TorchScript 模型不能像普通字典那样迭代
- `detect_camera_num_from_weights` 函数尝试迭代 TorchScript 模型对象，导致错误

## ✅ 修复方案

### 1. 修复 `detect_camera_num_from_weights` 函数

**文件**：`visualize_gradcam.py`

**修改内容**：
- 添加了 TorchScript 模型检测
- 如果是 TorchScript 模型，返回默认相机数量（4）
- 添加了异常处理，避免崩溃

**代码**：
```python
def detect_camera_num_from_weights(weight_path: str) -> int:
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # 检查是否是 TorchScript 模型
        if isinstance(checkpoint, torch.jit.ScriptModule) or isinstance(checkpoint, torch.jit.ScriptFunction):
            print(f"⚠️  检测到 TorchScript 模型，无法自动检测相机数量，使用默认值 4")
            return 4
        
        # 处理普通权重文件...
    except Exception as e:
        print(f"⚠️  加载权重文件时出错: {e}，使用默认相机数量 4")
        return 4
```

### 2. 在脚本中添加 TorchScript 模型检测和错误提示

**文件**：`generate_heatmap_visualization.py`

**修改内容**：
- 在加载模型之前检测是否是 TorchScript 模型
- 如果是 TorchScript 模型，给出清晰的错误提示和解决方案
- 说明 TorchScript 模型不支持 Grad-CAM（因为无法访问内部层结构）

**代码**：
```python
# 检查是否是 TorchScript 模型
is_torchscript = False
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        is_torchscript = isinstance(checkpoint, torch.jit.ScriptModule) or isinstance(checkpoint, torch.jit.ScriptFunction)
except Exception as e:
    print(f"⚠️  检查权重文件格式时出错: {e}")
    is_torchscript = False

if is_torchscript:
    print("❌ 错误：检测到 TorchScript 模型（.pt 文件）")
    print("   TorchScript 模型不支持 Grad-CAM 热力图生成")
    print("   原因：TorchScript 模型是编译后的模型，无法访问内部层结构")
    print("\n💡 解决方案：")
    print("   1. 使用训练好的 PyTorch 权重文件（.pth 文件）")
    print("   2. 或者使用 torch.save() 保存的完整模型权重")
    return
```

## 📋 重要说明

### TorchScript 模型的限制

1. **不支持 Grad-CAM**：
   - TorchScript 模型是编译后的模型
   - 无法访问内部层结构（无法注册 hook）
   - 无法计算梯度（用于 Grad-CAM）

2. **需要 PyTorch 权重文件**：
   - 使用 `.pth` 文件（PyTorch 权重字典）
   - 或者使用 `torch.save(model.state_dict(), 'model.pth')` 保存的权重

### 解决方案

**选项1：使用 PyTorch 权重文件（.pth）**
```bash
# 如果只有 TorchScript 模型，需要重新保存为 PyTorch 权重
# 在训练脚本中使用：
torch.save(model.state_dict(), 'model.pth')

# 然后使用 .pth 文件：
python generate_heatmap_visualization.py \
    --weight_path model.pth \
    --config_file config.yml \
    --query_id 000274 \
    --dataset_root /path/to/RGBNT201 \
    --output_path heatmap_000274.png
```

**选项2：从训练检查点提取权重**
```python
# 如果训练时保存了检查点，可以提取权重：
checkpoint = torch.load('checkpoint.pth')
torch.save(checkpoint['model'], 'model_weights.pth')
```

## ✅ 验证

修复后，脚本会：
1. ✅ 正确检测 TorchScript 模型
2. ✅ 给出清晰的错误提示
3. ✅ 提供解决方案建议
4. ✅ 避免崩溃，优雅退出

## 📝 相关文件

- `visualize_gradcam.py` - 修复了 `detect_camera_num_from_weights` 函数
- `generate_heatmap_visualization.py` - 添加了 TorchScript 模型检测和错误处理
