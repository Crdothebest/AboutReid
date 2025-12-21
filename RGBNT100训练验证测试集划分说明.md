# RGBNT100 训练集、验证集、测试集划分说明

## ✅ 数据集划分完成

### 📊 数据集统计

| 集合 | 车辆数 | 图像数 | 摄像头数 | 用途 |
|------|--------|--------|----------|------|
| **训练集** | 50 | 7815 | 8 | 用于训练模型 |
| **验证集** | 50 | 860 | 8 | 用于训练监控、模型选择、早停 |
| **查询集** | 50 | 1715 | 8 | 用于最终测试评估 |
| **图库集** | 50 | 8575 | 8 | 用于最终测试评估 |
| **总计** | 100 | 18965 | 8 | - |

### 📁 目录结构

```
/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/rgbir/
├── bounding_box_train/  # 训练集（7815 张图像）
├── bounding_box_val/     # 验证集（860 张图像，从训练集中划分 10%）
├── query/                # 查询集（1715 张图像）
└── bounding_box_test/    # 图库集（8575 张图像）
```

---

## 🔧 划分方法

### 验证集划分策略
- **从训练集中划分**: 从每个车辆的图像中随机选择 10% 作为验证集
- **随机种子**: 42（确保可重复性）
- **划分比例**: 10%（可调整）

### 划分逻辑
1. 按车辆 ID 分组所有训练图像
2. 对每个车辆的图像进行随机打乱
3. 从每个车辆中选择 10% 的图像移动到验证集
4. 确保每个车辆至少保留 1 张图像在训练集中

---

## 📋 数据集用途

### 训练集（Train Set）
- **用途**: 训练模型参数
- **数量**: 7815 张图像
- **车辆**: 前 50 个车辆（PID: 501-550）

### 验证集（Validation Set）
- **用途**: 
  - 训练过程中监控模型性能
  - 模型选择（选择最佳 epoch）
  - 早停（Early Stopping）
  - 超参数调优
- **数量**: 860 张图像（从训练集中划分）
- **车辆**: 与训练集相同的 50 个车辆

### 测试集（Test Set = Query + Gallery）
- **用途**: 
  - 最终模型性能评估
  - 论文报告结果
  - 与其他方法对比
- **查询集**: 1715 张图像
- **图库集**: 8575 张图像
- **车辆**: 后 50 个车辆（PID: 551-600）

---

## 🚀 使用方法

### 1. 创建验证集（如果还没有）

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python create_train_val_test_split.py --val_split 0.1
```

### 2. 验证数据集加载

数据集加载器会自动：
- ✅ 检测验证集是否存在
- ✅ 如果存在，使用验证集进行训练监控
- ✅ 如果不存在，使用测试集作为验证集（向后兼容）

### 3. 训练模型

```bash
python train_net.py \
    --config_file configs/RGBNT100/jzb_baseline_optimize.yml \
    --use_moe \
    MODEL.USE_CLIP_MULTI_SCALE True \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]"
```

训练过程中：
- **训练集**: 用于更新模型参数
- **验证集**: 用于监控性能，选择最佳模型
- **测试集**: 用于最终评估（在训练完成后）

---

## 📊 数据加载验证

运行以下命令验证数据集加载：

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python -c "
from config import cfg
from data import make_dataloader

cfg.merge_from_file('configs/RGBNT100/jzb_baseline_optimize.yml')
cfg.freeze()

train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

print(f'训练集批次: {len(train_loader)}')
print(f'验证集批次: {len(val_loader)}')
print(f'查询集图像数: {num_query}')
print(f'类别数: {num_classes}')
"
```

**预期输出**:
```
✅ 使用验证集进行训练监控（从训练集中划分）
   验证集: 860 张图像
训练集批次: 118
验证集批次: 14
查询集图像数: 1715
类别数: 50
```

---

## 🔄 重新划分验证集

如果需要重新划分验证集（例如改变比例），运行：

```bash
python create_train_val_test_split.py --val_split 0.15  # 15% 验证集
```

**注意**: 脚本会自动删除现有验证集并重新创建。

---

## 📝 代码修改说明

### 1. 数据集类（RGBNT100.py）
- ✅ 添加了 `val_dir` 和 `val` 属性
- ✅ 支持加载验证集（如果存在）
- ✅ 更新了统计信息打印

### 2. 数据加载器（make_dataloader.py）
- ✅ 优先使用验证集（如果存在）
- ✅ 如果验证集不存在，使用测试集（向后兼容）
- ✅ 打印验证集使用情况

---

## ✅ 验证清单

- [x] 训练集存在（7815 张图像）
- [x] 验证集已创建（860 张图像）
- [x] 查询集存在（1715 张图像）
- [x] 图库集存在（8575 张图像）
- [x] 数据集可以正常加载
- [x] 验证集被正确使用

---

## 🎯 总结

现在 RGBNT100 数据集已经具备完整的训练集、验证集、测试集划分：

1. **训练集**: 7815 张图像，用于训练
2. **验证集**: 860 张图像，用于训练监控
3. **测试集**: 10290 张图像（1715 + 8575），用于最终评估

数据集已准备好用于模型训练！

---

**最后更新**: 2025-12-22
