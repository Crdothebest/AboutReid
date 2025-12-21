# 多尺度（4、8、16）无 MOE 实验命令

## 🎯 实验目标

使用多尺度滑动窗口（4×4、8×8、16×16）提取特征，但**不使用 MOE 融合**，而是使用简单的 MLP 融合。

---

## 📋 方法 1: 使用命令行参数（推荐）

### 基本命令

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python train_net.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --use_multi_scale \
    --disable_moe \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]"
```

### 完整命令（包含输出目录）

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python train_net.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --use_multi_scale \
    --disable_moe \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]" \
    OUTPUT_DIR "outputs/ablation/multiscale_no_moe_$(date +%Y%m%d_%H%M%S)"
```

---

## 📋 方法 2: 使用配置文件 + 命令行参数

### 步骤 1: 创建配置文件

创建新配置文件：`configs/RGBNT201/multiscale_no_moe.yml`

```yaml
MODEL:
  PRETRAIN_PATH_T: '/home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt'
  TRANSFORMER_TYPE: 'ViT-B-16'
  STRIDE_SIZE: [ 16, 16 ]
  SIE_CAMERA: True
  DIRECT: 1
  SIE_COE: 1.0
  ID_LOSS_WEIGHT: 0.25
  TRIPLET_LOSS_WEIGHT: 1.0
  PROMPT: True
  ADAPTER: True
  MAMBA: True
  FROZEN: True
  
  # ========== 多尺度滑动窗口配置：启用多尺度特征提取 ==========
  USE_CLIP_MULTI_SCALE: True   # ✅ 启用多尺度滑动窗口
  CLIP_MULTI_SCALE_SCALES: [4, 8, 16]  # ✅ 使用 4×4、8×8、16×16 三个尺度
  
  # ========== MoE配置：禁用MOE，使用简单MLP融合 ==========
  USE_MULTI_SCALE_MOE: False   # ❌ 禁用 MOE 融合
  # 注意：当 USE_MULTI_SCALE_MOE = False 时，会使用 CLIPMultiScaleSlidingWindow
  # 该模块使用简单的两层 MLP 进行特征融合

INPUT:
  SIZE_TRAIN: [ 256, 128 ]
  SIZE_TEST: [ 256, 128 ]
  PROB: 0.5
  RE: 0.5
  PIXEL_MEAN: [0.485, 0.456, 0.406]
  PIXEL_STD: [0.229, 0.224, 0.225]
  PADDING: 10

DATASETS:
  NAMES: ('RGBNT201',)
  ROOT_DIR: ('/home/zhanghaoyang/Desktop/yzy/AboutReid/datasets',)

SOLVER:
  OPTIMIZER_NAME: 'AdamW'
  MAX_EPOCHS: 60
  BASE_LR: 0.00035
  WEIGHT_DECAY: 0.0005
  BIAS_LR_FACTOR: 1.0
  WEIGHT_DECAY_BIAS: 0.0005
  MOMENTUM: 0.9
  SCHEDULER: 'cosine'
  WARMUP_EPOCHS: 5
  WARMUP_FACTOR: 0.01
  WARMUP_METHOD: 'linear'
  WARMUP_ITERS: 500
  FREEZE_EPOCHS: 0
  GAMMA: 0.1
  STEPSIZE: [20, 40]
  LR_SCHEDULER: 'cosine'
  ETA_MIN_LR: 0.0

TEST:
  IMS_PER_BATCH: 128
  WEIGHT: ''
  EVAL_PERIOD: 5

OUTPUT_DIR: 'outputs/ablation/multiscale_no_moe'
```

### 步骤 2: 运行训练

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python train_net.py \
    --config_file configs/RGBNT201/multiscale_no_moe.yml
```

---

## 📋 方法 3: 使用 --opts 参数（灵活配置）

```bash
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python train_net.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    MODEL.USE_CLIP_MULTI_SCALE True \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]" \
    MODEL.USE_MULTI_SCALE_MOE False \
    OUTPUT_DIR "outputs/ablation/multiscale_no_moe_$(date +%Y%m%d_%H%M%S)"
```

---

## 🔍 验证配置

训练开始后，检查日志文件，确认以下配置：

```
✅ USE_CLIP_MULTI_SCALE: True
✅ CLIP_MULTI_SCALE_SCALES: [4, 8, 16]
❌ USE_MULTI_SCALE_MOE: False
```

### 检查命令

```bash
# 查看最新日志文件
tail -f outputs/ablation/multiscale_no_moe_*/logs/train_*.log | grep -E "USE_CLIP_MULTI_SCALE|USE_MULTI_SCALE_MOE|CLIP_MULTI_SCALE_SCALES"
```

---

## 📊 实验对比

### 实验设置对比

| 配置项 | 多尺度 + MOE | 多尺度 + 无 MOE |
|--------|-------------|----------------|
| `USE_CLIP_MULTI_SCALE` | ✅ True | ✅ True |
| `CLIP_MULTI_SCALE_SCALES` | [4, 8, 16] | [4, 8, 16] |
| `USE_MULTI_SCALE_MOE` | ✅ True | ❌ False |
| 融合方式 | MOE 专家网络 + 门控网络 | 简单 MLP 融合 |
| 参数量 | 较多（专家网络） | 较少（仅 MLP） |

### 预期结果

- **多尺度 + MOE**: 使用专家网络和门控网络进行动态融合，性能较高
- **多尺度 + 无 MOE**: 使用简单 MLP 融合，性能可能略低，但计算量更小

---

## 🚀 快速启动脚本

创建脚本文件：`run_multiscale_no_moe.sh`

```bash
#!/bin/bash

cd /home/zhanghaoyang/Desktop/yzy/AboutReid

eval "$(conda shell.bash hook)"
conda activate MambaPro

python train_net.py \
    --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
    --use_multi_scale \
    --disable_moe \
    MODEL.CLIP_MULTI_SCALE_SCALES "[4,8,16]" \
    OUTPUT_DIR "outputs/ablation/multiscale_no_moe_$(date +%Y%m%d_%H%M%S)"

echo "✅ 训练完成！"
```

### 运行脚本

```bash
chmod +x run_multiscale_no_moe.sh
./run_multiscale_no_moe.sh
```

---

## 📝 注意事项

1. **多尺度滑动窗口必须启用**：`USE_CLIP_MULTI_SCALE = True`
2. **MOE 必须禁用**：`USE_MULTI_SCALE_MOE = False`
3. **尺度配置**：`CLIP_MULTI_SCALE_SCALES = [4, 8, 16]`
4. **融合方式**：当 MOE 禁用时，自动使用 `CLIPMultiScaleSlidingWindow` 模块的简单 MLP 融合
5. **输出目录**：建议使用带时间戳的输出目录，便于区分不同实验

---

## 🔗 相关文档

- **消融实验方案**: `消融实验_MOE替代方案.md`
- **多尺度配置说明**: `Readme合集/8-多尺度配置说明.md`
- **MoE 命令行开关**: `Readme合集/32-MoE命令行开关使用说明.md`

---

**最后更新**: 2025-12-21

