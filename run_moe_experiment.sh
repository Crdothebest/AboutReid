#!/bin/bash

# 多尺度MoE实验启动脚本
# 功能：快速启动MoE实验，对比传统MLP融合和MoE融合的效果

echo "🔥 多尺度MoE实验启动脚本"
echo "================================"

# 设置实验参数
DATASET="RGBNT201"
CONFIG_BASELINE="configs/RGBNT201/MambaPro.yml"
CONFIG_MOE="configs/RGBNT201/MambaPro_multi_scale_moe.yml"
OUTPUT_DIR="outputs/moe_experiment_$(date +%Y%m%d_%H%M%S)"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "📊 实验配置："
echo "  数据集: $DATASET"
echo "  基线配置: $CONFIG_BASELINE"
echo "  MoE配置: $CONFIG_MOE"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_BASELINE" ]; then
    echo "❌ 基线配置文件不存在: $CONFIG_BASELINE"
    exit 1
fi

if [ ! -f "$CONFIG_MOE" ]; then
    echo "❌ MoE配置文件不存在: $CONFIG_MOE"
    exit 1
fi

echo "✅ 配置文件检查通过"
echo ""

# 选择实验模式
echo "🎯 请选择实验模式："
echo "1. 快速测试 (5个epoch)"
echo "2. 完整实验 (60个epoch)"
echo "3. 仅测试MoE模块功能"
echo "4. 对比实验 (基线 vs MoE)"
echo ""

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "🚀 启动快速测试模式..."
        EPOCHS=5
        QUICK_MODE=true
        ;;
    2)
        echo "🚀 启动完整实验模式..."
        EPOCHS=60
        QUICK_MODE=false
        ;;
    3)
        echo "🧪 启动MoE模块功能测试..."
        python test_multi_scale_moe.py
        exit 0
        ;;
    4)
        echo "📊 启动对比实验模式..."
        EPOCHS=30
        QUICK_MODE=false
        COMPARISON_MODE=true
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "🔧 实验参数："
echo "  训练轮数: $EPOCHS"
echo "  快速模式: $QUICK_MODE"
if [ "$COMPARISON_MODE" = true ]; then
    echo "  对比模式: 是"
fi
echo ""

# 修改配置文件中的训练轮数
if [ "$QUICK_MODE" = true ]; then
    echo "📝 修改配置文件为快速模式..."
    sed -i "s/MAX_EPOCHS: 60/MAX_EPOCHS: $EPOCHS/g" $CONFIG_MOE
    sed -i "s/MAX_EPOCHS: 60/MAX_EPOCHS: $EPOCHS/g" $CONFIG_BASELINE
fi

# 启动实验
if [ "$COMPARISON_MODE" = true ]; then
    echo "📊 启动对比实验..."
    
    # 1. 运行基线实验
    echo "🔵 运行基线实验 (传统MLP融合)..."
    python tools/train.py \
        --config_file $CONFIG_BASELINE \
        --output_dir $OUTPUT_DIR/baseline \
        --max_epochs $EPOCHS \
        --log_period 5 \
        --eval_period 5
    
    # 2. 运行MoE实验
    echo "🟢 运行MoE实验 (多尺度MoE融合)..."
    python tools/train.py \
        --config_file $CONFIG_MOE \
        --output_dir $OUTPUT_DIR/moe \
        --max_epochs $EPOCHS \
        --log_period 5 \
        --eval_period 5
    
    # 3. 生成对比报告
    echo "📈 生成对比报告..."
    python analyze_results.py \
        --baseline_dir $OUTPUT_DIR/baseline \
        --moe_dir $OUTPUT_DIR/moe \
        --output_dir $OUTPUT_DIR/comparison_report
    
else
    echo "🚀 启动MoE实验..."
    python tools/train.py \
        --config_file $CONFIG_MOE \
        --output_dir $OUTPUT_DIR \
        --max_epochs $EPOCHS \
        --log_period 10 \
        --eval_period 5
fi

echo ""
echo "🎉 实验完成！"
echo "📁 结果保存在: $OUTPUT_DIR"
echo ""

# 显示结果摘要
if [ -f "$OUTPUT_DIR/train_log.txt" ]; then
    echo "📊 训练结果摘要："
    echo "=================="
    tail -20 $OUTPUT_DIR/train_log.txt
fi

if [ "$COMPARISON_MODE" = true ] && [ -f "$OUTPUT_DIR/comparison_report/summary.txt" ]; then
    echo ""
    echo "📈 对比实验结果："
    echo "=================="
    cat $OUTPUT_DIR/comparison_report/summary.txt
fi

echo ""
echo "✅ 实验脚本执行完成！"
echo "💡 提示：可以使用以下命令查看详细结果："
echo "   cat $OUTPUT_DIR/train_log.txt"
echo "   ls -la $OUTPUT_DIR/"
