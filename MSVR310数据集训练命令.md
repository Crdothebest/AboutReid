# MSVR310数据集训练命令指南

## 📋 基本训练命令

### 1. 基础训练（使用默认配置）

```bash
# 使用MambaPro基础配置
python train_net.py --config_file configs/MSVR310/MambaPro.yml
```

### 2. 使用MoE配置训练

```bash
# 使用MoE优化配置（推荐）
python train_net.py --config_file configs/MSVR310/MambaPro_moe.yml
```

### 3. 基线实验（禁用MoE）

```bash
# 基线实验：禁用MoE，仅使用多尺度滑动窗口
python train_net.py --config_file configs/MSVR310/MambaPro.yml --disable_moe --use_multi_scale

# 完全基线：禁用MoE和多尺度
python train_net.py --config_file configs/MSVR310/MambaPro.yml --disable_moe --no_multi_scale
```

### 4. 启用MoE训练

```bash
# 启用MoE特征融合
python train_net.py --config_file configs/MSVR310/MambaPro.yml --use_moe
```

## 🔧 不同模型配置

### T2T-ViT模型

```bash
# T2T-ViT基线
python train_net.py --config_file configs/MSVR310/MambaPro_T2T_Baseline.yml

# T2T-ViT + 多尺度
python train_net.py --config_file configs/MSVR310/MambaPro_T2T_MultiScale.yml
```

### ViT模型

```bash
# ViT基线
python train_net.py --config_file configs/MSVR310/MambaPro_ViT_Baseline.yml

# ViT + 多尺度
python train_net.py --config_file configs/MSVR310/MambaPro_ViT_MultiScale.yml
```

## ⚙️ 命令行参数控制

### 多尺度滑动窗口控制

```bash
# 启用多尺度滑动窗口
python train_net.py --config_file configs/MSVR310/MambaPro.yml --use_multi_scale

# 禁用多尺度滑动窗口
python train_net.py --config_file configs/MSVR310/MambaPro.yml --no_multi_scale
```

### MoE控制

```bash
# 启用MoE（即使配置文件中禁用）
python train_net.py --config_file configs/MSVR310/MambaPro.yml --use_moe

# 禁用MoE（即使配置文件中启用）
python train_net.py --config_file configs/MSVR310/MambaPro_moe.yml --disable_moe
```

### 门控融合控制

```bash
# 启用门控融合机制
python train_net.py --config_file configs/MSVR310/MambaPro_moe.yml --use_attention

# 禁用门控融合机制
python train_net.py --config_file configs/MSVR310/MambaPro_moe.yml --disable_attention

# 自定义门控网络参数
python train_net.py --config_file configs/MSVR310/MambaPro_moe.yml --use_attention --attention_heads 8 --attention_dropout 0.1
```

## 📝 配置文件参数覆盖

可以通过命令行直接覆盖配置文件中的参数：

```bash
# 修改学习率
python train_net.py --config_file configs/MSVR310/MambaPro.yml SOLVER.BASE_LR 0.0005

# 修改训练轮数
python train_net.py --config_file configs/MSVR310/MambaPro.yml SOLVER.MAX_EPOCHS 80

# 修改批次大小
python train_net.py --config_file configs/MSVR310/MambaPro.yml SOLVER.IMS_PER_BATCH 64

# 修改数据集路径
python train_net.py --config_file configs/MSVR310/MambaPro.yml DATASETS.ROOT_DIR '/your/path/to/msvr310'
```

## 🎯 完整示例

### 示例1：完整MoE训练（推荐）

```bash
python train_net.py \
  --config_file configs/MSVR310/MambaPro_moe.yml \
  --use_moe
```

### 示例2：基线对比实验

```bash
# 基线：无MoE，无多尺度
python train_net.py \
  --config_file configs/MSVR310/MambaPro.yml \
  --disable_moe \
  --no_multi_scale

# 仅多尺度
python train_net.py \
  --config_file configs/MSVR310/MambaPro.yml \
  --disable_moe \
  --use_multi_scale

# 多尺度 + MoE
python train_net.py \
  --config_file configs/MSVR310/MambaPro_moe.yml \
  --use_moe
```

### 示例3：自定义参数训练

```bash
python train_net.py \
  --config_file configs/MSVR310/MambaPro_moe.yml \
  --use_moe \
  SOLVER.BASE_LR 0.0005 \
  SOLVER.MAX_EPOCHS 80 \
  SOLVER.IMS_PER_BATCH 32 \
  DATASETS.ROOT_DIR '/home/zubuntu/workspace/yzy/MambaPro/data/msvr310'
```

## ⚠️ 重要注意事项

### 1. 数据集路径配置

确保配置文件中的数据集路径正确：

```yaml
DATASETS:
  NAMES: ('MSVR310')
  ROOT_DIR: '/home/zubuntu/workspace/yzy/MambaPro/data/msvr310'
```

如果路径不同，可以通过命令行覆盖：

```bash
python train_net.py --config_file configs/MSVR310/MambaPro.yml \
  DATASETS.ROOT_DIR '/your/custom/path/to/msvr310'
```

### 2. 数据集目录结构

MSVR310数据集需要按照以下结构组织：

```
msvr310/
├── bounding_box_train/    # 训练集
│   ├── 0001/              # 车辆ID目录
│   │   ├── vis/           # 可见光图像
│   │   ├── ni/            # 近红外图像
│   │   └── th/            # 热红外图像
│   └── ...
├── query3/                # 查询集
│   └── ...
└── bounding_box_test/     # 图库集
    └── ...
```

### 3. 预训练权重路径

确保配置文件中的预训练权重路径正确：

```yaml
MODEL:
  PRETRAIN_PATH_T: '/home/zubuntu/workspace/yzy/MambaPro/pths/ViT-B-16.pt'
```

### 4. 输出目录

训练结果会保存到配置文件中指定的 `OUTPUT_DIR`：

```yaml
OUTPUT_DIR: '/home/zubuntu/workspace/yzy/MambaPro/outputs/msvr310_mamba_experiment'
```

## 🔍 参考RGBNT201的命令格式

RGBNT201的训练命令格式与MSVR310完全相同，只需替换配置文件路径：

```bash
# RGBNT201示例
python train_net.py --config_file configs/RGBNT201/MambaPro.yml --use_moe

# MSVR310示例（格式相同）
python train_net.py --config_file configs/MSVR310/MambaPro_moe.yml --use_moe
```

## 📊 训练日志

训练日志会保存在输出目录的 `train_log.txt` 文件中，可以使用之前创建的 `plot_training_curves.py` 脚本可视化：

```bash
python plot_training_curves.py --log_file outputs/msvr310_mamba_experiment/train_log.txt
```

## 🚀 快速开始

最简单的启动方式：

```bash
# 1. 确保数据集路径正确
# 2. 运行训练
python train_net.py --config_file configs/MSVR310/MambaPro_moe.yml
```

