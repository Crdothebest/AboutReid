#!/bin/bash
"""
实验运行脚本
用于运行基线模型和多尺度MoE模型的对比实验

作者修改：创建自动化实验运行脚本
功能：按顺序运行基线实验和MoE实验
撤销方法：删除此文件
"""

echo "🧪 MambaPro 多尺度MoE 实验对比"
echo "=================================="
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
OUTPUT_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
echo "📁 创建输出目录: $OUTPUT_DIR"
echo ""

# 运行基线实验
echo "🚀 开始运行基线实验..."
echo "=================================="
python train_net.py --config_file configs/RGBNT201/MambaPro_baseline.yml 2>&1 | tee $OUTPUT_DIR/baseline_experiment.log

# 检查基线实验是否成功
if [ $? -eq 0 ]; then
    echo "✅ 基线实验完成"
else
    echo "❌ 基线实验失败"
    exit 1
fi

echo ""
echo "🚀 开始运行MoE实验..."
echo "=================================="
python train_net.py --config_file configs/RGBNT201/MambaPro_moe.yml 2>&1 | tee $OUTPUT_DIR/moe_experiment.log

# 检查MoE实验是否成功
if [ $? -eq 0 ]; then
    echo "✅ MoE实验完成"
else
    echo "❌ MoE实验失败"
    exit 1
fi

echo ""
echo "📊 生成实验报告..."
echo "=================================="

# 创建简单的对比报告
cat > $OUTPUT_DIR/experiment_summary.txt << EOF
MambaPro 多尺度MoE 实验对比报告
=====================================

实验时间: $(date)
输出目录: $OUTPUT_DIR

实验配置:
- 基线模型: configs/RGBNT201/MambaPro_baseline.yml
- MoE模型:  configs/RGBNT201/MambaPro_moe.yml

实验结果:
- 基线实验日志: baseline_experiment.log
- MoE实验日志:   moe_experiment.log

请查看日志文件获取详细的训练结果和性能指标。

EOF

echo "📋 实验报告已生成: $OUTPUT_DIR/experiment_summary.txt"
echo ""
echo "🎉 所有实验完成！"
echo "📁 结果保存在: $OUTPUT_DIR"
echo "📅 结束时间: $(date)"
