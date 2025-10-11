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
# 检查是否有--config_file参数
CONFIG_FILE="configs/RGBNT201/MambaPro_moe.yml"  # 默认配置

# 处理--config_file参数
if [[ "$1" == "--config_file" ]]; then
    CONFIG_FILE="$2"
    shift 2  # 移除--config_file和配置文件路径
    echo "📋 使用指定配置文件: $CONFIG_FILE"
else
    echo "📋 使用默认配置文件: $CONFIG_FILE"
fi

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

# 初始化门控融合参数（默认值）
# 默认禁用门控融合机制，使用传统MLP融合
ATTENTION_ENABLED=false
ATTENTION_HEADS=8
ATTENTION_DROPOUT=0.1

# 检查是否有命令行参数（除了--config_file）
if [ $# -gt 0 ]; then
    echo "🔧 检测到命令行参数，将动态覆盖配置文件参数..."
    echo "📋 原始配置文件: $CONFIG_FILE"
    echo "📝 修改后配置文件: $MODIFIED_CONFIG"
    echo "🔍 剩余参数数量: $#"
    echo "🔍 剩余参数: $@"
    
    # 解析并应用参数覆盖
    # 格式：MODEL.MOE_EXPERT_HIDDEN_DIM 640
    echo "🔍 调试：开始处理参数，剩余参数数量: $#"
    while [ $# -gt 0 ]; do
        PARAM_NAME="$1"
        PARAM_VALUE="$2"
        echo "🔍 调试：处理参数 $PARAM_NAME = $PARAM_VALUE"
        
        # 🔥 新增：处理门控融合相关参数
        if [[ "$PARAM_NAME" == "ATTENTION_ENABLED" ]]; then
            ATTENTION_ENABLED="$PARAM_VALUE"
            echo "  🎯 设置门控融合机制: $ATTENTION_ENABLED"
        elif [[ "$PARAM_NAME" == "ATTENTION_HEADS" ]]; then
            ATTENTION_HEADS="$PARAM_VALUE"
            echo "  🎯 设置门控网络头数: $ATTENTION_HEADS"
        elif [[ "$PARAM_NAME" == "ATTENTION_DROPOUT" ]]; then
            ATTENTION_DROPOUT="$PARAM_VALUE"
            echo "  🎯 设置门控网络Dropout: $ATTENTION_DROPOUT"
        elif [[ "$PARAM_NAME" == "MODEL.USE_GATE_FUSION" ]]; then
            echo "  🔍 调试：处理MODEL.USE_GATE_FUSION参数"
            if [[ "$PARAM_VALUE" == "True" || "$PARAM_VALUE" == "true" ]]; then
                ATTENTION_ENABLED="true"
                echo "  🎯 通过命令行启用门控融合机制"
                echo "  🔍 调试：ATTENTION_ENABLED设置为: $ATTENTION_ENABLED"
            elif [[ "$PARAM_VALUE" == "False" || "$PARAM_VALUE" == "false" ]]; then
                ATTENTION_ENABLED="false"
                echo "  🎯 通过命令行禁用门控融合机制"
                echo "  🔍 调试：ATTENTION_ENABLED设置为: $ATTENTION_ENABLED"
            fi
            echo "  🔍 调试：MODEL.USE_GATE_FUSION处理完成"
        elif [[ "$PARAM_NAME" == "MODEL.GATE_NUM_HEADS" ]]; then
            ATTENTION_HEADS="$PARAM_VALUE"
            echo "  🎯 通过MODEL.GATE_NUM_HEADS设置门控网络头数: $ATTENTION_HEADS"
        elif [[ "$PARAM_NAME" == "MODEL.GATE_DROPOUT" ]]; then
            ATTENTION_DROPOUT="$PARAM_VALUE"
            echo "  🎯 通过MODEL.GATE_DROPOUT设置门控网络Dropout: $ATTENTION_DROPOUT"
        elif [ -n "$PARAM_NAME" ] && [ -n "$PARAM_VALUE" ]; then
            echo "  📝 覆盖参数: $PARAM_NAME = $PARAM_VALUE"
            
            # 使用sed动态修改配置文件中的参数
            # 处理不同的参数格式
            if [[ "$PARAM_NAME" == *"."* ]]; then
                # 处理嵌套参数，如 MODEL.MOE_EXPERT_HIDDEN_DIM
                SECTION=$(echo "$PARAM_NAME" | cut -d'.' -f1)
                KEY=$(echo "$PARAM_NAME" | cut -d'.' -f2-)
                echo "  🔍 调试：处理嵌套参数 $SECTION.$KEY = $PARAM_VALUE"
                
                # 查找并替换参数
                echo "  🔍 调试：执行sed命令前"
                sed -i.bak "/^$SECTION:/,/^[A-Z_]*:/ s|^  $KEY:.*|  $KEY: $PARAM_VALUE|" "$MODIFIED_CONFIG"
                echo "  🔍 调试：执行sed命令后"
                
                # 🔥 特殊处理：如果参数在MODEL部分，确保正确替换
                if [[ "$SECTION" == "MODEL" ]]; then
                    # 先删除所有现有的该参数设置
                    sed -i.bak "/^  $KEY:/d" "$MODIFIED_CONFIG"
                    # 然后在MODEL部分末尾添加新设置
                    sed -i.bak "/^MODEL:/a\\  $KEY: $PARAM_VALUE" "$MODIFIED_CONFIG"
                fi
            else
                # 处理简单参数
                sed -i.bak "s|^$PARAM_NAME:.*|$PARAM_NAME: $PARAM_VALUE|" "$MODIFIED_CONFIG"
            fi
        fi
        
        # 移动到下一对参数
        shift 2
        echo "  🔍 调试：参数处理完成，剩余参数数量: $#"
    done
    echo "🔍 调试：所有参数处理完成"
    
    echo "✅ 参数覆盖完成"
    echo "📊 使用配置: 原配置 + 命令行参数覆盖"
    echo "🔍 调试：参数处理完成，继续执行..."
    
    # 🔥 调试：显示关键参数修改结果
    echo "🔍 关键参数检查："
    if grep -q "USE_GATE_FUSION:" "$MODIFIED_CONFIG"; then
        echo "  - USE_GATE_FUSION设置："
        grep "USE_GATE_FUSION:" "$MODIFIED_CONFIG" | head -5
    fi
    if grep -q "MOE_TEMPERATURE:" "$MODIFIED_CONFIG"; then
        echo "  - MOE_TEMPERATURE: $(grep "MOE_TEMPERATURE:" "$MODIFIED_CONFIG" | head -1)"
    fi
    echo "🔍 调试：关键参数检查完成，继续执行..."
else
    echo "ℹ️  未检测到命令行参数，使用配置文件默认值"
    echo "📊 使用配置: 原配置文件参数（无修改）"
fi

# =============================================================================
# 第五部分：构建训练命令
# =============================================================================

# 🔥 门控融合配置：根据命令行参数动态设置
echo "🔍 调试：进入门控融合配置阶段"
echo "🔍 调试：ATTENTION_ENABLED当前值: $ATTENTION_ENABLED"
echo "🔍 调试：ATTENTION_HEADS当前值: $ATTENTION_HEADS"
echo "🔍 调试：ATTENTION_DROPOUT当前值: $ATTENTION_DROPOUT"

if [ "$ATTENTION_ENABLED" = "true" ]; then
    echo "🎯 配置门控融合机制：启用门控融合"
    # 更新现有的门控融合配置
    sed -i.bak "s|^  USE_GATE_FUSION:.*|  USE_GATE_FUSION: True|" "$MODIFIED_CONFIG"
    sed -i.bak "s|^  GATE_NUM_HEADS:.*|  GATE_NUM_HEADS: $ATTENTION_HEADS|" "$MODIFIED_CONFIG"
    sed -i.bak "s|^  GATE_DROPOUT:.*|  GATE_DROPOUT: $ATTENTION_DROPOUT|" "$MODIFIED_CONFIG"
    echo "🎯 门控融合机制已启用: ${ATTENTION_HEADS}个门控头, Dropout=${ATTENTION_DROPOUT}"
elif [ "$ATTENTION_ENABLED" = "false" ]; then
    echo "🎯 配置门控融合机制：禁用门控融合机制，使用传统MLP融合"
    # 更新现有的门控融合配置
    sed -i.bak "s|^  USE_GATE_FUSION:.*|  USE_GATE_FUSION: False|" "$MODIFIED_CONFIG"
    sed -i.bak "s|^  GATE_NUM_HEADS:.*|  GATE_NUM_HEADS: 8|" "$MODIFIED_CONFIG"
    sed -i.bak "s|^  GATE_DROPOUT:.*|  GATE_DROPOUT: 0.1|" "$MODIFIED_CONFIG"
    echo "🎯 门控融合机制已禁用：使用传统MLP融合"
else
    echo "ℹ️  使用默认配置：传统MLP融合机制（无门控融合）"
fi

# 验证YAML格式
echo "🔍 验证配置文件YAML格式..."
if python -c "import yaml; yaml.safe_load(open('$MODIFIED_CONFIG'))" 2>/dev/null; then
    echo "✅ 配置文件YAML格式正确"
else
    echo "❌ 配置文件YAML格式错误"
    echo "🔍 配置文件内容检查："
    tail -10 "$MODIFIED_CONFIG"
    exit 1
fi

# 构建训练命令 - 使用修改后的配置文件
# 这里直接写明，命令行的运行是走 train_net.py 文件
# 由于参数已经在配置文件中动态修改，这里只需要使用配置文件
CMD="python train_net.py --config_file $MODIFIED_CONFIG"

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
echo "🔍 调试：即将执行命令: $CMD"
echo "🔍 调试：当前工作目录: $(pwd)"
echo "🔍 调试：Python路径: $(which python)"
echo "🔍 调试：配置文件内容检查:"
head -20 "$MODIFIED_CONFIG"
echo "🔍 调试：配置文件末尾内容:"
tail -10 "$MODIFIED_CONFIG"

# 执行训练命令
# eval 命令用于执行存储在变量中的命令
echo "🚀 开始执行训练命令..."
echo "⏰ 设置5分钟超时，如果卡住将显示错误信息..."

# 使用timeout命令设置超时（增加到5分钟）
timeout 300s bash -c "$CMD"
TIMEOUT_EXIT_CODE=$?

if [ $TIMEOUT_EXIT_CODE -eq 124 ]; then
    echo "❌ 训练命令在5分钟内没有响应，可能卡住了"
    echo "🔍 可能的原因："
    echo "  1. 数据加载问题"
    echo "  2. 模型初始化问题"
    echo "  3. CUDA设备问题"
    echo "  4. 配置文件格式问题"
    echo "💡 建议：直接运行 python train_net.py --config_file $MODIFIED_CONFIG 来调试"
    exit 1
elif [ $TIMEOUT_EXIT_CODE -ne 0 ]; then
    echo "❌ 训练命令执行失败，退出码: $TIMEOUT_EXIT_CODE"
    exit $TIMEOUT_EXIT_CODE
fi

# =============================================================================
# 第八部分：处理训练结果
# =============================================================================

# 检查训练结果
# 如果超时退出，TIMEOUT_EXIT_CODE已经处理了
# 如果正常执行，检查退出码
if [ $TIMEOUT_EXIT_CODE -eq 0 ]; then
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
