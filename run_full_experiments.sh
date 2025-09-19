#!/bin/bash
"""
完整实验运行脚本
用于运行完整的60个epoch基线实验和创新点实验

作者修改：创建完整的实验运行脚本
功能：按顺序运行基线实验和创新点实验，并生成对比报告
撤销方法：删除此文件
"""

echo "🧪 MambaPro 多尺度MoE 完整实验对比"
echo "======================================"
echo "📅 开始时间: $(date)"
echo ""

# 激活conda环境
echo "🔧 激活MambaPro环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MambaPro

# 检查环境
echo "🔍 检查Python环境..."
python --version
echo "🔍 检查PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
echo ""

# 创建输出目录
OUTPUT_DIR="full_experiment_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
echo "📁 创建输出目录: $OUTPUT_DIR"
echo ""

# 运行基线实验
echo "🚀 开始运行基线实验（60个epoch）..."
echo "======================================"
echo "📋 配置文件: configs/RGBNT201/MambaPro_baseline.yml"
echo "⏱️  训练轮数: 60"
echo ""

BASELINE_START_TIME=$(date +%s)
python train_net.py --config_file configs/RGBNT201/MambaPro_baseline.yml SOLVER.MAX_EPOCHS 60 2>&1 | tee $OUTPUT_DIR/baseline_full_experiment.log
BASELINE_END_TIME=$(date +%s)
BASELINE_DURATION=$((BASELINE_END_TIME - BASELINE_START_TIME))

# 检查基线实验是否成功
if [ $? -eq 0 ]; then
    echo "✅ 基线实验完成"
    echo "⏱️  耗时: $((BASELINE_DURATION / 3600))小时$((BASELINE_DURATION % 3600 / 60))分钟"
else
    echo "❌ 基线实验失败"
    exit 1
fi

echo ""
echo "🚀 开始运行创新点实验（60个epoch）..."
echo "======================================"
echo "📋 配置文件: configs/RGBNT201/MambaPro.yml"
echo "⏱️  训练轮数: 60"
echo ""

MOE_START_TIME=$(date +%s)
python train_net.py --config_file configs/RGBNT201/MambaPro.yml SOLVER.MAX_EPOCHS 60 2>&1 | tee $OUTPUT_DIR/moe_full_experiment.log
MOE_END_TIME=$(date +%s)
MOE_DURATION=$((MOE_END_TIME - MOE_START_TIME))

# 检查MoE实验是否成功
if [ $? -eq 0 ]; then
    echo "✅ 创新点实验完成"
    echo "⏱️  耗时: $((MOE_DURATION / 3600))小时$((MOE_DURATION % 3600 / 60))分钟"
else
    echo "❌ 创新点实验失败"
    exit 1
fi

echo ""
echo "📊 生成实验对比报告..."
echo "======================================"

# 创建对比报告
cat > $OUTPUT_DIR/experiment_comparison_report.txt << EOF
MambaPro 多尺度MoE 完整实验对比报告
=====================================

实验时间: $(date)
输出目录: $OUTPUT_DIR

实验配置:
- 基线模型: configs/RGBNT201/MambaPro_baseline.yml
- 创新点模型: configs/RGBNT201/MambaPro.yml
- 训练轮数: 60 epochs

实验结果:
- 基线实验日志: baseline_full_experiment.log
- 创新点实验日志: moe_full_experiment.log

训练时间对比:
- 基线实验: $((BASELINE_DURATION / 3600))小时$((BASELINE_DURATION % 3600 / 60))分钟
- 创新点实验: $((MOE_DURATION / 3600))小时$((MOE_DURATION % 3600 / 60))分钟
- 时间差异: $(((MOE_DURATION - BASELINE_DURATION) / 3600))小时$(((MOE_DURATION - BASELINE_DURATION) % 3600 / 60))分钟

请查看日志文件获取详细的训练结果和性能指标。

EOF

echo "📋 实验报告已生成: $OUTPUT_DIR/experiment_comparison_report.txt"
echo ""
echo "🎉 所有实验完成！"
echo "📁 结果保存在: $OUTPUT_DIR"
echo "📅 结束时间: $(date)"

# 运行结果分析脚本
echo ""
echo "🔍 运行结果分析..."
python analyze_results.py --baseline_log $OUTPUT_DIR/baseline_full_experiment.log --moe_log $OUTPUT_DIR/moe_full_experiment.log --output_dir $OUTPUT_DIR
