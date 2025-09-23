# MoE命令行开关使用说明

## 🎯 功能概述

为了方便进行消融实验对比，我们为MoE模块设计了灵活的命令行开关，支持通过`--use_moe`和`--disable_moe`参数来控制MoE模块的启用和禁用。

---

## 🔥 命令行参数

### 1. 训练模式

#### 1.1 启用MoE模块
```bash
python tools/train.py --config_file configs/RGBNT201/MambaPro.yml --use_moe
```

#### 1.2 禁用MoE模块
```bash
python tools/train.py --config_file configs/RGBNT201/MambaPro.yml --disable_moe
```

#### 1.3 使用配置文件默认设置
```bash
python tools/train.py --config_file configs/RGBNT201/MambaPro_multi_scale_moe.yml
```

### 2. 测试模式

#### 2.1 启用MoE模块测试
```bash
python tools/test.py --config_file configs/RGBNT201/MambaPro.yml --use_moe \
    TEST.WEIGHT /path/to/your/model.pth
```

#### 2.2 禁用MoE模块测试
```bash
python tools/test.py --config_file configs/RGBNT201/MambaPro.yml --disable_moe \
    TEST.WEIGHT /path/to/your/model.pth
```

---

## 🎯 参数优先级

命令行参数的优先级顺序如下：

1. **`--disable_moe`** (最高优先级)
2. **`--use_moe`** (中等优先级)
3. **配置文件设置** (最低优先级)

### 示例说明

```bash
# 即使配置文件中启用了MoE，--disable_moe也会强制禁用
python tools/train.py --config_file configs/RGBNT201/MambaPro_multi_scale_moe.yml --disable_moe

# 即使配置文件中禁用了MoE，--use_moe也会强制启用
python tools/train.py --config_file configs/RGBNT201/MambaPro.yml --use_moe
```

---

## 🚀 消融实验示例

### 1. 手动消融实验

#### 1.1 基线实验（无多尺度，无MoE）
```bash
python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --disable_moe \
    --max_epochs 30 \
    --output_dir outputs/baseline_experiment
```

#### 1.2 多尺度实验（仅多尺度滑动窗口）
```bash
python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --disable_moe \
    --max_epochs 30 \
    --output_dir outputs/multi_scale_experiment \
    MODEL.USE_CLIP_MULTI_SCALE True
```

#### 1.3 MoE实验（多尺度 + MoE）
```bash
python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --use_moe \
    --max_epochs 30 \
    --output_dir outputs/moe_experiment
```

### 2. 使用消融实验脚本

#### 2.1 快速消融实验（5个epoch）
```bash
bash run_moe_ablation.sh
# 选择选项 1
```

#### 2.2 标准消融实验（30个epoch）
```bash
bash run_moe_ablation.sh
# 选择选项 2
```

#### 2.3 完整消融实验（60个epoch）
```bash
bash run_moe_ablation.sh
# 选择选项 3
```

---

## 📊 实验对比分析

### 1. 实验组设计

| 实验组 | 多尺度滑动窗口 | MoE融合 | 描述 |
|--------|----------------|---------|------|
| **基线** | ❌ | ❌ | 原始CLIP方法 |
| **多尺度** | ✅ | ❌ | 仅多尺度滑动窗口 |
| **MoE** | ✅ | ✅ | 多尺度 + MoE融合 |

### 2. 性能对比指标

- **mAP**: 平均精度均值
- **Rank-1**: Top-1准确率
- **Rank-5**: Top-5准确率
- **训练时间**: 单epoch训练时间
- **参数量**: 模型参数总数

### 3. 预期结果分析

```
基线方法 < 多尺度方法 < MoE方法
```

**预期提升**：
- 多尺度 vs 基线：+1.0% ~ +1.5% mAP
- MoE vs 多尺度：+0.5% ~ +1.0% mAP
- MoE vs 基线：+1.5% ~ +2.5% mAP

---

## 🔧 高级用法

### 1. 与其他参数组合使用

```bash
# 启用MoE + 自定义学习率
python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --use_moe \
    SOLVER.BASE_LR 0.001 \
    SOLVER.MAX_EPOCHS 40

# 禁用MoE + 自定义批次大小
python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --disable_moe \
    SOLVER.IMS_PER_BATCH 16
```

### 2. 分布式训练

```bash
# 多GPU训练 + MoE
torchrun --nproc_per_node=2 tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --use_moe \
    MODEL.DIST_TRAIN True
```

### 3. 测试不同MoE配置

```bash
# 测试不同专家数量
python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --use_moe \
    MODEL.MOE_SCALES [4,8] \
    --output_dir outputs/moe_2_experts

python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --use_moe \
    MODEL.MOE_SCALES [4,8,16,32] \
    --output_dir outputs/moe_4_experts
```

---

## 📝 日志输出说明

### 1. 训练日志中的MoE状态

```
🔥 命令行参数 --use_moe: 启用MoE模块
🔥 MoE模块状态: 启用
🔥 多尺度滑动窗口状态: 启用
🔥 MoE滑动窗口尺度: [4, 8, 16]
```

### 2. 专家权重统计

```
Expert Usage Statistics:
  4x4窗口专家: 平均权重=0.32, 激活率=0.85
  8x8窗口专家: 平均权重=0.35, 激活率=0.90
  16x16窗口专家: 平均权重=0.33, 激活率=0.88
```

---

## 🐛 常见问题

### 1. 参数冲突

**问题**：同时使用`--use_moe`和`--disable_moe`
```bash
python tools/train.py --use_moe --disable_moe  # ❌ 错误用法
```

**解决**：`--disable_moe`优先级更高，会强制禁用MoE

### 2. 配置文件不存在

**问题**：配置文件路径错误
```bash
python tools/train.py --config_file wrong_path.yml --use_moe  # ❌ 错误路径
```

**解决**：检查配置文件路径是否正确

### 3. 依赖模块未安装

**问题**：MoE模块导入失败
```
ImportError: No module named 'modeling.fusion_part.multi_scale_moe'
```

**解决**：确保MoE模块文件存在且路径正确

---

## 💡 最佳实践

### 1. 实验记录

建议为每个实验创建独立的输出目录：

```bash
# 使用时间戳命名
OUTPUT_DIR="outputs/experiment_$(date +%Y%m%d_%H%M%S)"

python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --use_moe \
    --output_dir $OUTPUT_DIR
```

### 2. 参数验证

在开始长时间训练前，建议先进行快速测试：

```bash
# 快速测试（5个epoch）
python tools/train.py \
    --config_file configs/RGBNT201/MambaPro.yml \
    --use_moe \
    --max_epochs 5 \
    --log_period 1
```

### 3. 结果对比

使用消融实验脚本自动生成对比报告：

```bash
bash run_moe_ablation.sh
# 自动运行所有实验组并生成对比报告
```

---

## 🎯 总结

MoE命令行开关提供了灵活的实验控制能力：

1. **简单易用**：通过`--use_moe`和`--disable_moe`轻松控制
2. **优先级明确**：命令行参数 > 配置文件设置
3. **兼容性好**：与现有训练/测试流程完全兼容
4. **实验友好**：支持快速消融实验和详细对比分析

通过合理使用这些开关，您可以高效地进行MoE模块的消融实验，验证其在跨模态行人重识别任务中的有效性。
