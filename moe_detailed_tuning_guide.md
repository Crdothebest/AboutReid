# MoE详细调参指导手册

## 🎯 调参目标与原理

### 核心目标
通过系统性调参，找到MoE模块的最佳配置，实现相比baseline的显著性能提升（目标：mAP +2%以上，Rank-1 +2%以上）。

### 调参原理
MoE模块的性能取决于三个关键因素：
1. **专家网络容量**：决定特征变换能力
2. **门控网络决策**：决定专家选择质量
3. **训练稳定性**：决定收敛效果

## 🔧 详细调参策略

### 阶段1：核心参数调优 (最重要)

#### 1.1 专家网络隐藏层维度 (`MOE_EXPERT_HIDDEN_DIM`)

**为什么重要**：
- 专家网络的容量直接影响特征变换能力
- 太小：表达能力不足，无法充分学习多尺度特征
- 太大：计算开销大，容易过拟合

**调参策略**：
```bash
# 测试序列：512 → 1024 → 2048
python moe_quick_test.py expert_512 MOE_EXPERT_HIDDEN_DIM=512
python moe_quick_test.py expert_1024 MOE_EXPERT_HIDDEN_DIM=1024  
python moe_quick_test.py expert_2048 MOE_EXPERT_HIDDEN_DIM=2048
```

**预期结果分析**：
- 512维：可能欠拟合，性能较低
- 1024维：平衡点，通常是最佳选择
- 2048维：可能过拟合，性能下降

**判断标准**：
- 如果1024维性能最好，继续测试1024维
- 如果2048维性能最好，测试4096维
- 如果512维性能最好，测试256维

#### 1.2 门控网络温度参数 (`MOE_TEMPERATURE`)

**为什么重要**：
- 控制专家权重分布的尖锐程度
- 影响专家分工的明确性
- 决定MoE的动态选择能力

**调参策略**：
```bash
# 测试序列：0.3 → 0.5 → 0.7 → 1.0 → 1.5
python moe_quick_test.py temp_03 MOE_TEMPERATURE=0.3
python moe_quick_test.py temp_05 MOE_TEMPERATURE=0.5
python moe_quick_test.py temp_07 MOE_TEMPERATURE=0.7
python moe_quick_test.py temp_10 MOE_TEMPERATURE=1.0
python moe_quick_test.py temp_15 MOE_TEMPERATURE=1.5
```

**预期结果分析**：
- 0.3：权重分布很尖锐，专家分工明确，但可能过于极端
- 0.5：权重分布适中，专家分工清晰
- 0.7：权重分布合理，通常是最佳选择
- 1.0：权重分布均匀，专家分工模糊
- 1.5：权重分布很均匀，专家分工不明确

**判断标准**：
- 观察专家权重分布：应该有一个主要专家（权重>0.5）
- 观察训练稳定性：权重变化应该平滑
- 观察性能：选择性能最好的温度

### 阶段2：网络结构调优

#### 2.1 专家网络层数 (`MOE_EXPERT_LAYERS`)

**为什么重要**：
- 控制专家网络的深度
- 影响特征变换的复杂度
- 决定专家网络的学习能力

**调参策略**：
```bash
# 测试序列：1 → 2 → 3
python moe_quick_test.py expert_layers_1 MOE_EXPERT_LAYERS=1
python moe_quick_test.py expert_layers_2 MOE_EXPERT_LAYERS=2
python moe_quick_test.py expert_layers_3 MOE_EXPERT_LAYERS=3
```

**预期结果分析**：
- 1层：可能欠拟合，特征变换能力不足
- 2层：平衡点，通常是最佳选择
- 3层：可能过拟合，训练困难

**判断标准**：
- 观察训练损失：应该平滑下降
- 观察验证性能：应该稳定提升
- 观察专家权重：应该合理分布

#### 2.2 门控网络层数 (`MOE_GATE_LAYERS`)

**为什么重要**：
- 控制门控网络的决策复杂度
- 影响专家选择的准确性
- 决定门控网络的学习能力

**调参策略**：
```bash
# 测试序列：1 → 2
python moe_quick_test.py gate_layers_1 MOE_GATE_LAYERS=1
python moe_quick_test.py gate_layers_2 MOE_GATE_LAYERS=2
```

**预期结果分析**：
- 1层：决策简单，可能不够准确
- 2层：决策合理，通常是最佳选择

**判断标准**：
- 观察门控权重：应该合理分布
- 观察专家激活：应该平衡使用
- 观察训练稳定性：应该平滑收敛

### 阶段3：正则化调优

#### 3.1 专家网络Dropout (`MOE_EXPERT_DROPOUT`)

**为什么重要**：
- 防止专家网络过拟合
- 提升泛化能力
- 影响训练稳定性

**调参策略**：
```bash
# 测试序列：0.0 → 0.1 → 0.2
python moe_quick_test.py expert_dropout_0 MOE_EXPERT_DROPOUT=0.0
python moe_quick_test.py expert_dropout_01 MOE_EXPERT_DROPOUT=0.1
python moe_quick_test.py expert_dropout_02 MOE_EXPERT_DROPOUT=0.2
```

**预期结果分析**：
- 0.0：可能过拟合，训练不稳定
- 0.1：平衡点，通常是最佳选择
- 0.2：可能欠拟合，性能下降

**判断标准**：
- 观察训练/验证性能差距：差距应该合理
- 观察训练稳定性：损失应该平滑下降
- 观察专家权重：应该合理分布

#### 3.2 门控网络Dropout (`MOE_GATE_DROPOUT`)

**为什么重要**：
- 防止门控网络过拟合
- 提升门控决策的稳定性
- 影响专家选择的平衡性

**调参策略**：
```bash
# 测试序列：0.0 → 0.1 → 0.2
python moe_quick_test.py gate_dropout_0 MOE_GATE_DROPOUT=0.0
python moe_quick_test.py gate_dropout_01 MOE_GATE_DROPOUT=0.1
python moe_quick_test.py gate_dropout_02 MOE_GATE_DROPOUT=0.2
```

**预期结果分析**：
- 0.0：可能过拟合，门控决策不稳定
- 0.1：平衡点，通常是最佳选择
- 0.2：可能欠拟合，门控决策过于保守

**判断标准**：
- 观察门控权重：应该合理分布
- 观察专家激活：应该平衡使用
- 观察训练稳定性：应该平滑收敛

### 阶段4：损失权重调优

#### 4.1 专家平衡损失权重 (`MOE_BALANCE_LOSS_WEIGHT`)

**为什么重要**：
- 促进专家使用平衡
- 防止某些专家被忽略
- 影响MoE的整体性能

**调参策略**：
```bash
# 测试序列：0.0 → 0.01 → 0.1
python moe_quick_test.py balance_0 MOE_BALANCE_LOSS_WEIGHT=0.0
python moe_quick_test.py balance_001 MOE_BALANCE_LOSS_WEIGHT=0.01
python moe_quick_test.py balance_01 MOE_BALANCE_LOSS_WEIGHT=0.1
```

**预期结果分析**：
- 0.0：专家使用可能不平衡
- 0.01：平衡点，通常是最佳选择
- 0.1：可能过度平衡，影响性能

**判断标准**：
- 观察专家使用频率：应该相对平衡
- 观察专家权重分布：应该合理
- 观察整体性能：应该稳定提升

#### 4.2 稀疏性损失权重 (`MOE_SPARSITY_LOSS_WEIGHT`)

**为什么重要**：
- 促进专家选择稀疏性
- 防止所有专家同时激活
- 影响MoE的计算效率

**调参策略**：
```bash
# 测试序列：0.0 → 0.001 → 0.01
python moe_quick_test.py sparsity_0 MOE_SPARSITY_LOSS_WEIGHT=0.0
python moe_quick_test.py sparsity_001 MOE_SPARSITY_LOSS_WEIGHT=0.001
python moe_quick_test.py sparsity_01 MOE_SPARSITY_LOSS_WEIGHT=0.01
```

**预期结果分析**：
- 0.0：专家选择可能不稀疏
- 0.001：平衡点，通常是最佳选择
- 0.01：可能过度稀疏，影响性能

**判断标准**：
- 观察专家激活率：应该合理稀疏
- 观察专家权重：应该主要激活1-2个专家
- 观察整体性能：应该稳定提升

## 📊 调参结果分析

### 性能指标分析

#### 1. 主要性能指标
- **mAP**: 平均精度，主要指标
- **Rank-1**: 首位命中率，重要指标
- **Rank-5**: 前5位命中率，参考指标

#### 2. 训练稳定性指标
- **损失曲线**: 应该平滑下降
- **验证性能**: 应该稳定提升
- **专家权重**: 应该合理分布

#### 3. 专家分析指标
- **专家使用频率**: 应该相对平衡
- **专家激活率**: 应该合理稀疏
- **门控权重分布**: 应该合理

### 结果判断标准

#### 1. 性能提升标准
- **mAP提升**: > +2% 为显著提升
- **Rank-1提升**: > +2% 为显著提升
- **训练稳定性**: 损失曲线平滑下降

#### 2. 专家分析标准
- **专家平衡度**: 各专家使用频率方差 < 0.1
- **专家激活率**: 平均激活率 > 0.8
- **门控稳定性**: 权重分布方差 < 0.05

#### 3. 过拟合判断标准
- **训练/验证性能差距**: < 5% 为合理
- **专家权重分布**: 应该合理
- **训练稳定性**: 应该平滑收敛

## 🚀 调参执行计划

### 第1轮：核心参数测试
```bash
# 测试专家维度
python moe_quick_test.py expert_512 MOE_EXPERT_HIDDEN_DIM=512
python moe_quick_test.py expert_1024 MOE_EXPERT_HIDDEN_DIM=1024
python moe_quick_test.py expert_2048 MOE_EXPERT_HIDDEN_DIM=2048

# 测试温度参数
python moe_quick_test.py temp_03 MOE_TEMPERATURE=0.3
python moe_quick_test.py temp_05 MOE_TEMPERATURE=0.5
python moe_quick_test.py temp_07 MOE_TEMPERATURE=0.7
python moe_quick_test.py temp_10 MOE_TEMPERATURE=1.0
```

### 第2轮：网络结构测试
```bash
# 基于第1轮最佳结果，测试网络结构
python moe_quick_test.py expert_layers_1 MOE_EXPERT_LAYERS=1
python moe_quick_test.py expert_layers_2 MOE_EXPERT_LAYERS=2
python moe_quick_test.py expert_layers_3 MOE_EXPERT_LAYERS=3
```

### 第3轮：正则化测试
```bash
# 基于前两轮最佳结果，测试正则化
python moe_quick_test.py expert_dropout_0 MOE_EXPERT_DROPOUT=0.0
python moe_quick_test.py expert_dropout_01 MOE_EXPERT_DROPOUT=0.1
python moe_quick_test.py expert_dropout_02 MOE_EXPERT_DROPOUT=0.2
```

### 第4轮：损失权重测试
```bash
# 基于前三轮最佳结果，测试损失权重
python moe_quick_test.py balance_0 MOE_BALANCE_LOSS_WEIGHT=0.0
python moe_quick_test.py balance_001 MOE_BALANCE_LOSS_WEIGHT=0.01
python moe_quick_test.py balance_01 MOE_BALANCE_LOSS_WEIGHT=0.1
```

## 📝 实验记录模板

### 实验记录格式
```json
{
  "experiment_id": "moe_test_001",
  "experiment_name": "专家维度测试",
  "start_time": "2024-01-01T10:00:00",
  "parameters": {
    "MOE_EXPERT_HIDDEN_DIM": 1024,
    "MOE_TEMPERATURE": 0.7,
    "MOE_EXPERT_DROPOUT": 0.1
  },
  "results": {
    "mAP": 85.2,
    "Rank-1": 92.1,
    "Rank-5": 96.8
  },
  "expert_analysis": {
    "expert_usage": [0.35, 0.32, 0.33],
    "expert_activation_rate": 0.85,
    "gate_weight_variance": 0.03
  },
  "status": "completed"
}
```

### 结果对比表
| 实验ID | 专家维度 | 温度 | mAP | Rank-1 | 专家平衡度 | 状态 |
|--------|----------|------|-----|--------|------------|------|
| test1  | 512      | 0.7  | 83.1 | 90.2   | 0.08       | 完成 |
| test2  | 1024     | 0.7  | 85.2 | 92.1   | 0.05       | 完成 |
| test3  | 2048     | 0.7  | 84.8 | 91.8   | 0.06       | 完成 |

## 🎯 成功标准

### 性能标准
- mAP > baseline + 2%
- Rank-1 > baseline + 2%
- 训练稳定，无异常

### 专家标准
- 专家使用平衡
- 权重分布合理
- 激活模式清晰

### 效率标准
- 训练时间合理
- 内存使用可控
- 推理速度可接受

## 🔍 常见问题解决

### 1. 性能下降
- **原因**: 参数设置不当
- **解决**: 检查专家维度和温度参数
- **建议**: 回到baseline配置，逐步调整

### 2. 训练不稳定
- **原因**: Dropout设置不当
- **解决**: 调整Dropout比例
- **建议**: 从0.1开始，逐步调整

### 3. 专家不平衡
- **原因**: 平衡损失权重不当
- **解决**: 调整平衡损失权重
- **建议**: 从0.01开始，逐步调整

### 4. 过拟合
- **原因**: 网络容量过大
- **解决**: 减少专家维度或增加Dropout
- **建议**: 观察训练/验证性能差距

通过这个详细的调参指导，您可以系统性地找到MoE模块的最佳配置，实现显著的性能提升！
