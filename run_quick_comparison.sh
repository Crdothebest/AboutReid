#!/bin/bash
"""
快速对比实验脚本
用于运行较短时间的对比实验来验证创新点效果

作者修改：创建快速对比实验脚本
功能：运行较短时间的训练来快速验证创新点效果
撤销方法：删除此文件
"""

echo "🧪 MambaPro 多尺度MoE 快速对比实验"
echo "======================================"
echo "📅 开始时间: $(date)"
echo ""

# 激活conda环境
echo "🔧 激活MambaPro环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MambaPro

# 创建输出目录
OUTPUT_DIR="quick_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
echo "📁 创建输出目录: $OUTPUT_DIR"
echo ""

# 运行基线实验（10个epoch）
echo "🚀 开始运行基线实验（10个epoch）..."
echo "======================================"
echo "📋 配置文件: configs/RGBNT201/MambaPro_baseline.yml"
echo "⏱️  训练轮数: 10"
echo ""

BASELINE_START_TIME=$(date +%s)
python train_net.py --config_file configs/RGBNT201/MambaPro_baseline.yml SOLVER.MAX_EPOCHS 10 2>&1 | tee $OUTPUT_DIR/baseline_quick.log
BASELINE_END_TIME=$(date +%s)
BASELINE_DURATION=$((BASELINE_END_TIME - BASELINE_START_TIME))

# 检查基线实验是否成功
if [ $? -eq 0 ]; then
    echo "✅ 基线实验完成"
    echo "⏱️  耗时: $((BASELINE_DURATION / 60))分钟$((BASELINE_DURATION % 60))秒"
else
    echo "❌ 基线实验失败"
    exit 1
fi

echo ""
echo "🚀 开始运行创新点实验（10个epoch）..."
echo "======================================"
echo "📋 配置文件: configs/RGBNT201/MambaPro.yml"
echo "⏱️  训练轮数: 10"
echo ""

MOE_START_TIME=$(date +%s)
python train_net.py --config_file configs/RGBNT201/MambaPro.yml SOLVER.MAX_EPOCHS 10 2>&1 | tee $OUTPUT_DIR/moe_quick.log
MOE_END_TIME=$(date +%s)
MOE_DURATION=$((MOE_END_TIME - MOE_START_TIME))

# 检查MoE实验是否成功
if [ $? -eq 0 ]; then
    echo "✅ 创新点实验完成"
    echo "⏱️  耗时: $((MOE_DURATION / 60))分钟$((MOE_DURATION % 60))秒"
else
    echo "❌ 创新点实验失败"
    exit 1
fi

echo ""
echo "📊 生成快速对比报告..."
echo "======================================"

# 创建对比报告
cat > $OUTPUT_DIR/quick_comparison_report.txt << EOF
MambaPro 多尺度MoE 快速对比实验报告
=====================================

实验时间: $(date)
输出目录: $OUTPUT_DIR

实验配置:
- 基线模型: configs/RGBNT201/MambaPro_baseline.yml
- 创新点模型: configs/RGBNT201/MambaPro.yml
- 训练轮数: 10 epochs

实验结果:
- 基线实验日志: baseline_quick.log
- 创新点实验日志: moe_quick.log

训练时间对比:
- 基线实验: $((BASELINE_DURATION / 60))分钟$((BASELINE_DURATION % 60))秒
- 创新点实验: $((MOE_DURATION / 60))分钟$((MOE_DURATION % 60))秒
- 时间差异: $(((MOE_DURATION - BASELINE_DURATION) / 60))分钟$(((MOE_DURATION - BASELINE_DURATION) % 60))秒

请查看日志文件获取详细的训练结果和性能指标。

EOF

echo "📋 实验报告已生成: $OUTPUT_DIR/quick_comparison_report.txt"
echo ""
echo "🎉 快速对比实验完成！"
echo "📁 结果保存在: $OUTPUT_DIR"
echo "📅 结束时间: $(date)"

# 运行结果分析脚本
echo ""
echo "🔍 运行结果分析..."
python analyze_results.py --baseline_log $OUTPUT_DIR/baseline_quick.log --moe_log $OUTPUT_DIR/moe_quick.log --output_dir $OUTPUT_DIR
