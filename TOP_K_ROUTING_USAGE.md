# 🔥 Top-k 路由使用指南

## 📋 功能概述

Top-k 路由机制允许您强制激活权重最大的 k 个专家，屏蔽其他专家，从而：
- ✅ 解决 E1 垄断问题：强制激活多个专家，避免单个专家主导
- ✅ 提高特征多样性：确保多个专家的知识都被利用
- ✅ 平衡稀疏性和性能：在效率和性能之间取得平衡

## 🎯 配置参数

### 1. 基本参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `MODEL.MOE_USE_TOP_K_ROUTING` | bool | `False` | 是否启用 Top-k 路由 |
| `MODEL.MOE_TOP_K` | int | `2` | Top-k 路由的 k 值（推荐 2，即 Top-2） |
| `MODEL.MOE_TOP_K_MODE` | str | `"soft"` | Top-k 路由模式：`"soft"` 或 `"hard"` |

### 2. 模式说明

#### 软 Top-k（推荐）
- **模式**：`"soft"`
- **特点**：重新归一化 Top-k 权重，保留被屏蔽专家的梯度
- **优势**：
  - ✅ 减少与 Load Balancing Loss 的冲突
  - ✅ 训练更稳定
  - ✅ 保留被屏蔽专家的梯度，便于后续优化
- **适用场景**：训练阶段，需要平衡专家使用

#### 硬 Top-k
- **模式**：`"hard"`
- **特点**：直接 mask 非 Top-k 专家，完全屏蔽其贡献
- **优势**：
  - ✅ 更彻底的稀疏激活
  - ✅ 推理效率更高
- **风险**：
  - ⚠️ 可能丢失关键信息
  - ⚠️ 与 Load Balancing Loss 冲突更严重
- **适用场景**：推理阶段，追求极致效率

## 📝 使用方法

### 方法 1：配置文件（YAML）

在配置文件中添加以下参数：

```yaml
MODEL:
  # Top-k 路由配置
  MOE_USE_TOP_K_ROUTING: True    # 启用 Top-k 路由
  MOE_TOP_K: 2                    # Top-2 路由（推荐）
  MOE_TOP_K_MODE: "soft"         # 软 Top-k（推荐）
```

### 方法 2：命令行参数（--opts）

```bash
./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --opts \
  MODEL.MOE_USE_TOP_K_ROUTING True \
  MODEL.MOE_TOP_K 2 \
  MODEL.MOE_TOP_K_MODE "soft" \
  OUTPUT_DIR "./outputs/V24.1_Top2_Soft"
```

### 方法 3：完整命令行示例

```bash
#!/bin/bash
# ==========================================================
# 🚀 V24.1 策略：Top-2 路由（软路由）
# 目标：解决 E1 垄断问题，提升 mAP
# ==========================================================
./run_experiment.sh \
  --config_file configs/RGBNT201/yzy_best_Mambapro_moe.yml \
  --opts \
  # 1. MoE 损失权重（沿用成功配置）
  SOLVER.MOE_BALANCE_LOSS_WEIGHT 0.0010 \
  SOLVER.MOE_DIVERSITY_LOSS_WEIGHT 0.0010 \
  # 2. LR 衰减时机（沿用成功时机）
  SOLVER.STEPS "(35, 45)" \
  SOLVER.GAMMA "0.1" \
  # 3. Top-k 路由配置（V24.1 策略核心）
  MODEL.MOE_USE_TOP_K_ROUTING True \
  MODEL.MOE_TOP_K 2 \
  MODEL.MOE_TOP_K_MODE "soft" \
  # 4. 模型结构配置（保持不变）
  MODEL.USE_MULTI_SCALE True \
  MODEL.USE_MULTI_SCALE_MOE True \
  MODEL.USE_GATE_FUSION False \
  MODEL.USE_ATTENTION_FUSION False \
  OUTPUT_DIR "./outputs/V24.1_Top2_Soft"
```

## 🔍 参数选择建议

### k 值选择

| k 值 | 激活比例 | 适用场景 |
|------|----------|----------|
| `1` | 33% (1/3) | 最稀疏，但可能丢失信息，不推荐 |
| `2` | 66% (2/3) | **推荐**，平衡稀疏性和信息保留 |
| `3` | 100% (3/3) | 等同于传统软路由，无需 Top-k |

**推荐**：对于 3 个专家的 MoE，使用 `k=2`（Top-2）。

### 模式选择

| 模式 | 训练阶段 | 推理阶段 | 推荐度 |
|------|----------|----------|--------|
| `"soft"` | ✅ 推荐 | ✅ 推荐 | ⭐⭐⭐⭐⭐ |
| `"hard"` | ⚠️ 不推荐 | ✅ 可选 | ⭐⭐⭐ |

**推荐**：使用 `"soft"` 模式，减少与损失函数的冲突。

## ⚠️ 注意事项

### 1. 与 Load Balancing Loss 的冲突

Top-k 路由与 Load Balancing Loss 可能存在目标冲突：
- **Load Balancing Loss**：希望所有专家平衡使用
- **Top-k 路由**：只激活 k 个专家

**建议**：
- 如果使用 Top-k 路由，可以降低 `SOLVER.MOE_BALANCE_LOSS_WEIGHT` 的权重
- 例如：从 `0.01` 降低到 `0.001` 或 `0.0001`

### 2. 训练稳定性

- **软 Top-k**：训练更稳定，推荐使用
- **硬 Top-k**：可能导致训练不稳定，不推荐在训练阶段使用

### 3. 特征维度

Top-k 路由**不会改变**输出特征维度，仍然是 `[B, 512]`，无需修改分类器。

## 📊 输出信息

启用 Top-k 路由后，训练时会输出以下信息：

```
🔥 Top-k 路由：已启用 (k=2, mode=soft)
🎯 Top-k 路由处理：强制激活 Top-2 专家
   - 输入权重形状: torch.Size([32, 3])
   - Top-k 值: 2
   - 路由模式: soft
   - 专家总数: 3
✅ 软 Top-k 路由完成
   - 输出权重形状: torch.Size([32, 3])
   - Top-2 专家权重已重新归一化
   - 非 Top-2 专家权重已设为 0（但保留梯度）
   - 说明：保留被屏蔽专家的梯度，减少与损失函数的冲突
```

## 🔬 实验建议

### 对比实验

1. **基线**：传统软路由（`MOE_USE_TOP_K_ROUTING False`）
2. **实验 1**：Top-2 软路由（`MOE_USE_TOP_K_ROUTING True, MOE_TOP_K 2, MOE_TOP_K_MODE "soft"`）
3. **实验 2**：Top-2 硬路由（`MOE_USE_TOP_K_ROUTING True, MOE_TOP_K 2, MOE_TOP_K_MODE "hard"`）

### 监控指标

- **专家权重分布**：观察 Top-k 路由后的权重分布
- **训练稳定性**：观察损失曲线是否平滑
- **性能指标**：对比 mAP、Rank-1、Rank-5 等指标

## 📚 相关文档

- `V24.1_TOP2_ROUTING_ANALYSIS_EVALUATION.md`：Top-2 路由的详细分析
- `V24.0_FEASIBILITY_ANALYSIS.md`：V24.0 策略的可行性分析
- `SOFT_VS_HARD_ROUTING_EXPLANATION.md`：软路由 vs 硬路由的详细解释

---

**最后更新**：2025-01-XX  
**版本**：V24.1

