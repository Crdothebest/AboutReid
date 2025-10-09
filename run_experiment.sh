#!/bin/bash
# =============================================================================
# 智能实验记录脚本
# 功能：自动记录任何训练命令的结果，支持动态参数
# 作者：实验记录系统
# 日期：2024
# =============================================================================

# =============================================================================
# 第一部分：初始化设置
# =============================================================================

# 设置基础目录 - 所有实验结果的根目录
BASE_DIR="results/everyExperiments"
# 创建基础目录（如果不存在的话）
mkdir -p "$BASE_DIR"

# =============================================================================
# 第二部分：生成实验ID和目录结构
# =============================================================================

# 生成实验ID - 使用时间戳确保唯一性
# 格式：20241219_143022（年月日_时分秒）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ID="experiment_${TIMESTAMP}"

# 创建实验目录 - 每个实验都有独立的目录
EXPERIMENT_DIR="$BASE_DIR/$EXPERIMENT_ID"
# 创建实验目录下的三个子目录：
# - logs: 存放训练日志
# - models: 存放模型权重
# - configs: 存放配置文件
mkdir -p "$EXPERIMENT_DIR"/{logs,models,configs}

# 显示实验开始信息
echo "🚀 开始智能实验记录..."
echo "📁 实验目录: $EXPERIMENT_DIR"
echo "🆔 实验ID: $EXPERIMENT_ID"

# =============================================================================
# 第三部分：配置文件处理
# =============================================================================

# 读取原始配置文件路径
CONFIG_FILE="configs/RGBNT201/MambaPro_moe.yml"
# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1  # 如果配置文件不存在，退出脚本
fi

# 复制原始配置文件到实验目录
MODIFIED_CONFIG="$EXPERIMENT_DIR/configs/experiment_config.yml"
cp "$CONFIG_FILE" "$MODIFIED_CONFIG"

# 修改配置文件中的输出目录
# 将原来的输出目录改为实验目录下的logs文件夹
# 这样每个实验的训练日志都会保存在独立的位置
sed -i.bak "s|OUTPUT_DIR:.*|OUTPUT_DIR: '$EXPERIMENT_DIR/logs'|g" "$MODIFIED_CONFIG"

# =============================================================================
# 第四部分：动态参数处理
# =============================================================================

# 初始化注意力参数（默认值）
ATTENTION_ENABLED=false
ATTENTION_HEADS=8
ATTENTION_DROPOUT=0.1

# 检查是否有命令行参数
if [ $# -gt 0 ]; then
    echo "🔧 检测到命令行参数，将动态覆盖配置文件参数..."
    echo "📋 原始配置文件: $CONFIG_FILE"
    echo "📝 修改后配置文件: $MODIFIED_CONFIG"
    
    # 解析并应用参数覆盖
    # 格式：MODEL.MOE_EXPERT_HIDDEN_DIM 640
    while [ $# -gt 0 ]; do
        PARAM_NAME="$1"
        PARAM_VALUE="$2"
        
        # 🔥 新增：处理注意力相关参数
        if [[ "$PARAM_NAME" == "ATTENTION_ENABLED" ]]; then
            ATTENTION_ENABLED="$PARAM_VALUE"
            echo "  🎯 设置注意力机制: $ATTENTION_ENABLED"
        elif [[ "$PARAM_NAME" == "ATTENTION_HEADS" ]]; then
            ATTENTION_HEADS="$PARAM_VALUE"
            echo "  🎯 设置注意力头数: $ATTENTION_HEADS"
        elif [[ "$PARAM_NAME" == "ATTENTION_DROPOUT" ]]; then
            ATTENTION_DROPOUT="$PARAM_VALUE"
            echo "  🎯 设置注意力Dropout: $ATTENTION_DROPOUT"
        elif [ -n "$PARAM_NAME" ] && [ -n "$PARAM_VALUE" ]; then
            echo "  📝 覆盖参数: $PARAM_NAME = $PARAM_VALUE"
            
            # 使用sed动态修改配置文件中的参数
            # 处理不同的参数格式
            if [[ "$PARAM_NAME" == *"."* ]]; then
                # 处理嵌套参数，如 MODEL.MOE_EXPERT_HIDDEN_DIM
                SECTION=$(echo "$PARAM_NAME" | cut -d'.' -f1)
                KEY=$(echo "$PARAM_NAME" | cut -d'.' -f2-)
                
                # 查找并替换参数
                sed -i.bak "/^$SECTION:/,/^[A-Z_]*:/ {
                    s|^  $KEY:.*|  $KEY: $PARAM_VALUE|
                }" "$MODIFIED_CONFIG"
            else
                # 处理简单参数
                sed -i.bak "s|^$PARAM_NAME:.*|$PARAM_NAME: $PARAM_VALUE|" "$MODIFIED_CONFIG"
            fi
        fi
        
        # 移动到下一对参数
        shift 2
    done
    
    echo "✅ 参数覆盖完成"
    echo "📊 使用配置: 原配置 + 命令行参数覆盖"
else
    echo "ℹ️  未检测到命令行参数，使用配置文件默认值"
    echo "📊 使用配置: 原配置文件参数（无修改）"
fi

# =============================================================================
# 第五部分：构建训练命令
# =============================================================================

# 构建训练命令 - 使用修改后的配置文件
# 这里直接写明，命令行的运行是走 train_net.py 文件
# 由于参数已经在配置文件中动态修改，这里只需要使用配置文件
# 🔥 新增：支持注意力机制的命令行参数（可配置）
if [ "$ATTENTION_ENABLED" = "true" ]; then
    CMD="python train_net.py --config_file $MODIFIED_CONFIG --use_attention --attention_heads $ATTENTION_HEADS --attention_dropout $ATTENTION_DROPOUT"
    echo "🎯 启用多头注意力机制: $ATTENTION_HEADS个注意力头, Dropout=$ATTENTION_DROPOUT"
else
    CMD="python train_net.py --config_file $MODIFIED_CONFIG"
    echo "ℹ️  使用传统MoE融合机制（无注意力）"
fi

# 显示将要执行的完整命令
echo "🔧 执行命令: $CMD"

# =============================================================================
# 第六部分：记录实验信息
# =============================================================================

# 创建实验信息文件 - 记录实验的基本信息
cat > "$EXPERIMENT_DIR/experiment_info.txt" << EOF
实验ID: $EXPERIMENT_ID
开始时间: $(date)
命令: $CMD
参数: $@
状态: 运行中
目录: $EXPERIMENT_DIR
EOF

# =============================================================================
# 第七部分：运行训练
# =============================================================================

# 显示训练开始信息
echo "🏃 开始训练..."
# 执行训练命令
# eval 命令用于执行存储在变量中的命令
eval $CMD

# =============================================================================
# 第八部分：处理训练结果
# =============================================================================

# 检查训练结果
# $? 是上一个命令的退出码，0表示成功
if [ $? -eq 0 ]; then
    # 训练成功的情况
    echo "✅ 训练完成"
    
    # 复制模型权重文件
    # 检查训练生成的模型权重文件是否存在
    if [ -f "$EXPERIMENT_DIR/logs/MambaProbest.pth" ]; then
        # 将模型权重复制到models目录
        cp "$EXPERIMENT_DIR/logs/MambaProbest.pth" "$EXPERIMENT_DIR/models/MambaProbest.pth"
        echo "✅ 模型权重已保存"
    fi
    
    # 更新实验信息文件 - 记录成功状态
    cat > "$EXPERIMENT_DIR/experiment_info.txt" << EOF
实验ID: $EXPERIMENT_ID
开始时间: $(date)
命令: $CMD
参数: $@
状态: 完成
目录: $EXPERIMENT_DIR
训练日志: $EXPERIMENT_DIR/logs/train_log.txt
模型权重: $EXPERIMENT_DIR/models/MambaProbest.pth
配置文件: $EXPERIMENT_DIR/configs/experiment_config.yml
EOF
    
    # 显示成功信息
    echo "📝 实验信息已保存"
    echo "🎯 实验完成: $EXPERIMENT_ID"
    echo "📁 结果目录: $EXPERIMENT_DIR"
    
else
    # 训练失败的情况
    echo "❌ 训练失败"
    
    # 更新错误信息文件 - 记录失败状态
    cat > "$EXPERIMENT_DIR/experiment_info.txt" << EOF
实验ID: $EXPERIMENT_ID
开始时间: $(date)
命令: $CMD
参数: $@
状态: 失败
目录: $EXPERIMENT_DIR
错误: 训练过程中出现错误
EOF
    
    # 显示错误信息
    echo "📝 错误信息已保存"
fi

# =============================================================================
# 第九部分：结束
# =============================================================================

# 显示实验记录完成信息
echo "🏆 实验记录完成！"
