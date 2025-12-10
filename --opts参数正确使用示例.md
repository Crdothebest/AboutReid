# --opts 参数正确使用示例

## ❌ 错误的参数路径

您之前使用的命令中，参数路径有误：

```bash
--opts \
  MODEL.MOE_BALANCE_LOSS_WEIGHT 0.10 \      # ❌ 错误：应该在 SOLVER 部分
  MODEL.MOE_DIVERSITY_LOSS_WEIGHT 0.10 \    # ❌ 错误：应该在 SOLVER 部分
  MODEL.MOE_SPARSITY_LOSS_WEIGHT 0.00005 \  # ❌ 错误：应该在 SOLVER 部分
  MODEL.MOE_NUM_EXPERTS 2                    # ❌ 错误：此参数不存在
```

## ✅ 正确的参数路径

### 1. 损失权重参数（在 SOLVER 部分）

```bash
./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --use_multi_scale --use_moe --disable_attention \
  --opts \
    MODEL.MOE_TEMPERATURE 1.6 \
    MODEL.MOE_EXPERT_HIDDEN_DIM 1024 \
    MODEL.MOE_EXPERT_LAYERS 2 \
    MODEL.MOE_SCALES "[8,16]" \
    SOLVER.MOE_BALANCE_LOSS_WEIGHT 0.10 \
    SOLVER.MOE_DIVERSITY_LOSS_WEIGHT 0.10 \
    SOLVER.MOE_SPARSITY_LOSS_WEIGHT 0.00005
```

### 2. 重要说明

- **专家数量**：由 `MODEL.MOE_SCALES` 决定，不需要单独设置 `MOE_NUM_EXPERTS`
  - `MOE_SCALES: [8, 16]` → 2个专家
  - `MOE_SCALES: [4, 8, 16]` → 3个专家

- **损失权重**：都在 `SOLVER` 部分，不在 `MODEL` 部分
  - `SOLVER.MOE_BALANCE_LOSS_WEIGHT`
  - `SOLVER.MOE_DIVERSITY_LOSS_WEIGHT`
  - `SOLVER.MOE_SPARSITY_LOSS_WEIGHT`

## 📋 完整的参数路径参考

### MODEL 部分参数

```bash
MODEL.MOE_TEMPERATURE              # 门控网络温度参数
MODEL.MOE_EXPERT_HIDDEN_DIM        # 专家网络隐藏层维度
MODEL.MOE_EXPERT_LAYERS            # 专家网络层数
MODEL.MOE_GATE_LAYERS              # 门控网络层数
MODEL.MOE_EXPERT_DROPOUT           # 专家网络Dropout
MODEL.MOE_GATE_DROPOUT             # 门控网络Dropout
MODEL.MOE_EXPERT_THRESHOLD         # 专家激活阈值
MODEL.MOE_RESIDUAL_WEIGHT          # 残差连接权重
MODEL.MOE_SCALES                   # MoE滑动窗口尺度（列表，如 "[8,16]"）
MODEL.USE_MULTI_SCALE_MOE          # 是否使用MoE（布尔值：True/False）
MODEL.USE_GATE_FUSION              # 是否使用门控融合（布尔值：True/False）
```

### SOLVER 部分参数

```bash
SOLVER.MOE_BALANCE_LOSS_WEIGHT     # 专家平衡损失权重
SOLVER.MOE_DIVERSITY_LOSS_WEIGHT   # 多样性损失权重
SOLVER.MOE_SPARSITY_LOSS_WEIGHT    # 稀疏性损失权重
```

## 🎯 完整示例

### 示例1：修改MoE核心参数

```bash
./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --use_multi_scale --use_moe \
  --opts \
    MODEL.MOE_TEMPERATURE 1.6 \
    MODEL.MOE_EXPERT_HIDDEN_DIM 1024 \
    MODEL.MOE_EXPERT_LAYERS 2 \
    MODEL.MOE_SCALES "[8,16]"
```

### 示例2：修改损失权重

```bash
./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --use_multi_scale --use_moe \
  --opts \
    SOLVER.MOE_BALANCE_LOSS_WEIGHT 0.10 \
    SOLVER.MOE_DIVERSITY_LOSS_WEIGHT 0.10 \
    SOLVER.MOE_SPARSITY_LOSS_WEIGHT 0.00005
```

### 示例3：同时修改所有参数

```bash
./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --use_multi_scale --use_moe --disable_attention \
  --opts \
    MODEL.MOE_TEMPERATURE 1.6 \
    MODEL.MOE_EXPERT_HIDDEN_DIM 1024 \
    MODEL.MOE_EXPERT_LAYERS 2 \
    MODEL.MOE_GATE_LAYERS 2 \
    MODEL.MOE_EXPERT_DROPOUT 0.1 \
    MODEL.MOE_GATE_DROPOUT 0.1 \
    MODEL.MOE_EXPERT_THRESHOLD 0.05 \
    MODEL.MOE_RESIDUAL_WEIGHT 1.0 \
    MODEL.MOE_SCALES "[8,16]" \
    SOLVER.MOE_BALANCE_LOSS_WEIGHT 0.10 \
    SOLVER.MOE_DIVERSITY_LOSS_WEIGHT 0.10 \
    SOLVER.MOE_SPARSITY_LOSS_WEIGHT 0.00005
```

## ⚠️ 注意事项

1. **列表值**：使用引号包裹，如 `"[8,16]"` 或 `"[4,8,16]"`
2. **布尔值**：使用 `True` 或 `False`（注意大小写）
3. **参数路径**：必须包含完整的前缀（`MODEL.` 或 `SOLVER.`）
4. **专家数量**：由 `MOE_SCALES` 的长度决定，不需要单独设置

## 🔍 验证参数是否生效

运行命令后，应该看到：

```
✅ 通过 --opts 修改配置: ['MODEL.MOE_TEMPERATURE', '1.6', 'MODEL.MOE_EXPERT_HIDDEN_DIM', '1024', ...]
```

如果看到错误信息，请检查参数路径是否正确。

