# RGBNT100 数据集分析与调理报告

## 📊 数据集概况

### 基本信息
- **数据集名称**: RGBNT100
- **数据类型**: RGB-IR 双模态车辆重识别数据集
- **车辆数量**: 100 个车辆
- **训练集**: 50 个车辆
- **测试集**: 50 个车辆

### 数据集结构

#### 原始数据目录结构
```
/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/
├── R/          # RGB 图像目录（100 个车辆ID子目录）
├── N/          # NIR 图像目录（100 个车辆ID子目录）
├── T/          # TIR 图像目录（100 个车辆ID子目录）
└── rgbir/      # 整理后的三元组图像目录
    ├── bounding_box_train/  # 训练集（8675 张图像）
    ├── query/               # 查询集（1715 张图像）
    └── bounding_box_test/   # 图库集（8575 张图像）
```

#### 整理后的数据集统计
- **训练集**: 8675 张图像（50 个车辆）
- **查询集**: 1715 张图像（50 个车辆）
- **图库集**: 8575 张图像（50 个车辆）
- **总计**: 18965 张图像

---

## 🔍 数据集特点

### 图像格式
- **尺寸**: 768×128 像素
- **格式**: 水平拼接的三元组图像
  - **RGB 部分**: [0:256, 0:128] - 左侧 256×128
  - **NI 部分**: [256:512, 0:128] - 中间 256×128
  - **TI 部分**: [512:768, 0:128] - 右侧 256×128

### 文件命名格式
- **格式**: `PID_cCAMID_*.jpg`
- **示例**: `0501_c0001_000.jpg`
  - `0501`: 车辆 ID（Person ID）
  - `c0001`: 摄像头 ID（Camera ID，从 1 开始）
  - `000`: 图像索引

### 数据加载逻辑
- RGBNT100 是 **RGB-IR 双模态**数据集
- 代码中为了兼容三模态模型，创建了虚拟的 TI 模态
- 实际使用中，TI 模态会被忽略，只使用 RGB 和 IR（NI）模态

---

## ⚠️ 当前问题

### 路径配置问题

**配置文件路径**: `configs/RGBNT100/jzb_baseline_optimize.yml`
```yaml
DATASETS:
  ROOT_DIR: '/home/zubuntu/workspace/MambaPro/MambaPro/data/'
```

**实际数据集位置**: 
```
/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/rgbir
```

**问题**: 配置文件中的路径与实际数据集位置不匹配

---

## 🔧 解决方案

### 方案 1: 创建符号链接（推荐）

```bash
# 创建期望的根目录（如果不存在）
mkdir -p /home/zubuntu/workspace/MambaPro/MambaPro/data

# 创建符号链接
ln -s /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100 \
      /home/zubuntu/workspace/MambaPro/MambaPro/data/RGBNT100
```

### 方案 2: 修改配置文件

修改 `configs/RGBNT100/jzb_baseline_optimize.yml`:
```yaml
DATASETS:
  ROOT_DIR: '/home/zhanghaoyang/Desktop/yzy/MambaPro/data'
```

注意：需要修改 `RGBNT100.py` 中的 `dataset_dir` 为 `'RGBNT100/rgbir'` 或直接使用完整路径。

### 方案 3: 使用备用路径（代码已支持）

代码中已经有备用路径机制，如果主路径不存在，会自动尝试备用路径。

---

## 📋 数据集验证清单

- [x] 原始数据目录存在（R、N、T 目录）
- [x] 整理后的数据集存在（rgbir 目录）
- [x] 训练集图像数量正确（8675 张）
- [x] 查询集图像数量正确（1715 张）
- [x] 图库集图像数量正确（8575 张）
- [ ] 配置文件路径匹配（需要修复）
- [ ] 数据加载测试通过（需要修复路径后测试）

---

## 🚀 快速修复命令

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

# 创建符号链接
mkdir -p /home/zubuntu/workspace/MambaPro/MambaPro/data
ln -s /home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100 \
      /home/zubuntu/workspace/MambaPro/MambaPro/data/RGBNT100

# 验证
ls -la /home/zubuntu/workspace/MambaPro/MambaPro/data/RGBNT100/rgbir
```

---

## 📝 数据集使用说明

### 1. 数据加载
数据集加载器会自动：
- 读取 768×128 的拼接图像
- 裁剪成三个 256×128 的图像（RGB、NI、TI）
- 对于 RGBNT100，TI 是虚拟的（使用 NI 图像）

### 2. 训练配置
使用配置文件 `configs/RGBNT100/jzb_baseline_optimize.yml`:
```bash
python train_net.py --config_file configs/RGBNT100/jzb_baseline_optimize.yml
```

### 3. 多尺度 + MOE 训练
```bash
python train_net.py \
    --config_file configs/RGBNT100/jzb_baseline_optimize.yml \
    --use_moe \
    MODEL.USE_CLIP_MULTI_SCALE True \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]"
```

---

## 📊 数据集统计（官方标准）

| 集合 | 车辆数 | 图像数 | 说明 |
|------|--------|--------|------|
| 训练集 | 50 | ~8675 | 前 50 个车辆 |
| 查询集 | 50 | 1715 | 后 50 个车辆，每个车辆每个摄像头的第一张 |
| 图库集 | 50 | ~8575 | 后 50 个车辆，除查询集外的所有图像 |

---

## ✅ 总结

1. **数据集已整理**: ✅ 数据集已经整理完成，位于 `/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/rgbir`
2. **数据完整性**: ✅ 所有图像文件都存在，数量符合官方标准
3. **路径配置**: ⚠️ 需要修复配置文件路径或创建符号链接
4. **数据加载**: ⚠️ 修复路径后需要测试数据加载功能

**下一步操作**:
1. 创建符号链接（推荐）或修改配置文件路径
2. 运行 `python analyze_rgbnt100_dataset.py` 验证数据集
3. 开始训练实验

---

**最后更新**: 2025-12-22
