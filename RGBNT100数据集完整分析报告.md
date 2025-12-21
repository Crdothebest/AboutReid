# RGBNT100 数据集完整分析报告

## 📊 数据集概况

### 基本信息
- **数据集名称**: RGBNT100
- **数据类型**: RGB-IR 双模态车辆重识别数据集
- **车辆数量**: 100 个车辆
- **训练集**: 50 个车辆（501-550）
- **测试集**: 50 个车辆（551-600）
- **摄像头数量**: 8 个

### 数据集统计

| 集合 | 车辆数 | 图像数 | 摄像头数 | 说明 |
|------|--------|--------|----------|------|
| **训练集** | 50 | 8675 | 8 | 前 50 个车辆（PID: 501-550） |
| **查询集** | 50 | 1715 | 8 | 后 50 个车辆（PID: 551-600） |
| **图库集** | 50 | 8575 | 8 | 后 50 个车辆（PID: 551-600） |
| **总计** | 100 | 18965 | 8 | - |

---

## 📁 数据集结构

### 实际数据位置
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

### 文件命名格式
- **格式**: `PID_cCAMID_INDEX.jpg`
- **示例**: `0577_c0008_007.jpg`
  - `0577`: 车辆 ID（Person ID）
  - `c0008`: 摄像头 ID（Camera ID，从 1 开始）
  - `007`: 图像索引

---

## 🔍 数据集特点

### 图像格式
- **原始尺寸**: 768×128 像素
- **格式**: 水平拼接的三元组图像
  - **RGB 部分**: [0:256, 0:128] - 左侧 256×128
  - **NI 部分**: [256:512, 0:128] - 中间 256×128
  - **TI 部分**: [512:768, 0:128] - 右侧 256×128

### 模态说明
- **RGBNT100 是 RGB-IR 双模态数据集**
- 代码中为了兼容三模态模型，创建了虚拟的 TI 模态
- 实际使用中，TI 模态会被忽略，只使用 RGB 和 IR（NI）模态
- 图像读取后会自动裁剪成三个 256×128 的图像

### 数据划分
- **训练集**: 前 50 个车辆（PID: 501-550）
- **测试集**: 后 50 个车辆（PID: 551-600）
- **查询集**: 从测试集中选择，每个车辆每个摄像头的第一张图像

---

## ✅ 数据集验证结果

### 目录结构检查
- ✅ **训练集目录存在**: 8675 张图像
- ✅ **查询集目录存在**: 1715 张图像
- ✅ **图库集目录存在**: 8575 张图像

### 文件命名检查
- ✅ **所有文件命名格式正确**: 符合 `PID_cCAMID_*.jpg` 格式
- ✅ **PID 范围正确**: 训练集 501-550，测试集 551-600
- ✅ **Camera ID 范围正确**: 1-8

### 图像完整性检查
- ✅ **图像尺寸正确**: 所有图像为 768×128
- ✅ **图像可正常读取**: 采样检查全部通过
- ✅ **数据加载正常**: 可以成功加载并裁剪成三个模态

---

## 🔧 已完成的修复

### 1. 配置文件路径修复
**修改前**:
```yaml
DATASETS:
  ROOT_DIR: '/home/zubuntu/workspace/MambaPro/MambaPro/data/'
```

**修改后**:
```yaml
DATASETS:
  ROOT_DIR: '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets'
```

### 2. 数据集验证
- ✅ 数据集可以正常加载
- ✅ 数据统计信息正确
- ✅ 图像读取功能正常

---

## 📋 数据集使用说明

### 1. 配置文件
使用配置文件: `configs/RGBNT100/jzb_baseline_optimize.yml`

### 2. 基本训练命令
```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python train_net.py \
    --config_file configs/RGBNT100/jzb_baseline_optimize.yml
```

### 3. 多尺度 + MOE 训练
```bash
python train_net.py \
    --config_file configs/RGBNT100/jzb_baseline_optimize.yml \
    --use_moe \
    MODEL.USE_CLIP_MULTI_SCALE True \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]"
```

### 4. 连续多尺度实验
```bash
./run_rgbnt100_baseline_continuous.sh
```

---

## 📊 数据集分析工具

### 分析脚本
- **`analyze_rgbnt100_dataset.py`**: 完整的数据集分析脚本
  - 检查目录结构
  - 验证文件命名
  - 检查图像完整性
  - 加载数据集并统计
  - 测试数据加载逻辑
  - 生成分析报告

### 使用方法
```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python analyze_rgbnt100_dataset.py
```

---

## 🎯 数据集特点总结

### 优势
1. ✅ **数据完整**: 所有图像文件都存在，数量符合官方标准
2. ✅ **格式规范**: 文件命名格式统一，易于解析
3. ✅ **划分合理**: 训练集和测试集划分清晰
4. ✅ **多模态**: RGB 和 IR 两种模态，适合跨模态检索研究

### 注意事项
1. ⚠️ **双模态数据集**: 虽然代码支持三模态，但实际只有 RGB 和 IR
2. ⚠️ **虚拟 TI**: TI 模态是虚拟的，使用 NI 图像
3. ⚠️ **图像尺寸**: 原始图像是 768×128，需要裁剪成三个 256×128

---

## 📝 相关文件

### 数据集相关
- **数据集位置**: `/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100/`
- **整理脚本**: `/home/zhanghaoyang/Desktop/yzy/organize_rgbnt100.py`
- **数据集类**: `AboutReid/data/datasets/RGBNT100.py`

### 配置文件
- **配置文件**: `configs/RGBNT100/jzb_baseline_optimize.yml`
- **已修复路径**: ✅ `ROOT_DIR: '/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets'`

### 分析工具
- **分析脚本**: `analyze_rgbnt100_dataset.py`
- **修复脚本**: `fix_rgbnt100_dataset.py`
- **分析报告**: `outputs/rgbnt100_dataset_analysis_report.txt`

---

## ✅ 总结

### 数据集状态
- ✅ **数据集已整理**: 所有图像已整理到 `rgbir/` 目录
- ✅ **路径已修复**: 配置文件路径已更新为实际路径
- ✅ **验证通过**: 数据集可以正常加载和使用
- ✅ **数据完整**: 所有图像文件存在且可读

### 可以开始使用
数据集已经准备好，可以开始训练实验：
1. ✅ 使用 `jzb_baseline_optimize.yml` 配置文件
2. ✅ 运行 `run_rgbnt100_baseline_continuous.sh` 进行多尺度实验
3. ✅ 数据集会自动加载并处理

---

**最后更新**: 2025-12-22
