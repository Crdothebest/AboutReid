# 命令行格式问题分析

## 🔍 发现的问题

### 问题 1：注释行后的反斜杠

**问题位置**：
```bash
  MODEL.USE_ATTENTION_FUSION True \ 

  # 4. 数据增强：引入 Random Erasing (RE - 鲁棒性升级！)
```

**问题分析**：
- `True \` 后面的反斜杠用于续行
- 但下一行是注释 `# 4. ...`，bash 会忽略注释
- 这导致反斜杠后面没有实际内容，可能导致参数解析问题

### 问题 2：注释行在参数中间

**问题位置**：
```bash
  --opts \
  
  # 1. MoE 损失权重 (沿用成功的高精度模式: L_Bal=0.0010)
  SOLVER.MOE_BALANCE_LOSS_WEIGHT 0.0010 \
```

**问题分析**：
- `--opts` 后面的注释行会被 bash 忽略
- 但注释行可能会影响参数收集逻辑

### 问题 3：参数格式

**当前格式**：
```bash
MODEL.USE_ATTENTION_FUSION True
```

**可能的问题**：
- YACS 可能将 `True` 解析为字符串而不是布尔值
- 需要确认是否需要引号

---

## ✅ 修正后的命令行

```bash
#!/bin/bash

# ==========================================================
# 🚀 V23.0 策略：多头注意力预处理 + Random Erasing 
# 目标：突破 78.6% 历史峰值，冲击 80.0%。
# ==========================================================

./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --opts \
  SOLVER.MOE_BALANCE_LOSS_WEIGHT 0.0010 \
  SOLVER.MOE_DIVERSITY_LOSS_WEIGHT 0.0010 \
  SOLVER.STEPS "(35, 45)" \
  SOLVER.GAMMA "0.1" \
  MODEL.USE_ATTENTION_FUSION True \
  INPUT.RE_PROB 0.5 \
  MODEL.USE_GATE_FUSION False \
  MODEL.USE_MULTI_SCALE True \
  MODEL.USE_MULTI_SCALE_MOE True \
  OUTPUT_DIR "./outputs/V23.0_MHA_RE_Final"
```

**关键修改**：
1. **移除所有注释行**：注释行在 `--opts` 后面会导致参数解析问题
2. **移除不必要的反斜杠**：最后一个参数后不需要反斜杠
3. **保持参数格式**：`MODEL.USE_ATTENTION_FUSION True` 格式正确

---

## 🔧 或者使用带注释的版本（注释放在参数前）

```bash
#!/bin/bash

# ==========================================================
# 🚀 V23.0 策略：多头注意力预处理 + Random Erasing 
# 目标：突破 78.6% 历史峰值，冲击 80.0%。
# ==========================================================

# 1. MoE 损失权重 (沿用成功的高精度模式: L_Bal=0.0010)
# 2. LR 衰减时机 (沿用成功时机，确保充分训练)
# 3. 结构激活：启用多头注意力预处理 (MHA - 结构升级！)
# 4. 数据增强：引入 Random Erasing (RE - 鲁棒性升级！)
# 5. 结构关闭：确保关闭冲突的 MLP 门控
# 6. 模型结构配置 (保持不变)

./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --opts \
  SOLVER.MOE_BALANCE_LOSS_WEIGHT 0.0010 \
  SOLVER.MOE_DIVERSITY_LOSS_WEIGHT 0.0010 \
  SOLVER.STEPS "(35, 45)" \
  SOLVER.GAMMA "0.1" \
  MODEL.USE_ATTENTION_FUSION True \
  INPUT.RE_PROB 0.5 \
  MODEL.USE_GATE_FUSION False \
  MODEL.USE_MULTI_SCALE True \
  MODEL.USE_MULTI_SCALE_MOE True \
  OUTPUT_DIR "./outputs/V23.0_MHA_RE_Final"
```

---

## 📋 问题总结

| 问题 | 位置 | 影响 | 解决方案 |
|------|------|------|---------|
| 注释行在 `--opts` 后 | 整个命令行 | 参数可能被忽略 | 移除注释或移到 `--opts` 前 |
| 反斜杠后跟注释 | `True \` 后 | 续行失败 | 移除反斜杠或移除注释 |
| 参数格式 | `MODEL.USE_ATTENTION_FUSION True` | 可能被解析为字符串 | 代码已处理，格式正确 |

---

## 🎯 建议

**推荐使用第一个版本（无注释）**，因为：
1. 更简洁
2. 避免参数解析问题
3. 注释可以在脚本开头统一说明

