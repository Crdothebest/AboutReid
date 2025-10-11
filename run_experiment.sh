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
        
        # 🔥 安全检查：确保参数名和值都不为空
        if [ -z "$PARAM_NAME" ] || [ -z "$PARAM_VALUE" ]; then
            echo "  ⚠️ 跳过无效参数: 名称为空或值为空"
            shift 1
            continue
        fi
        
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
        
        # 🔥 安全移动：确保参数数量减少
        if [ $# -ge 2 ]; then
            shift 2
        else
            shift 1
        fi
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
echo "⏰ 无超时限制，让训练自然完成..."

# 🔥 取消超时限制，让RGBNT100数据集正常训练
bash -c "$CMD"
TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "❌ 训练命令执行失败，退出码: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# =============================================================================
# 第八部分：处理训练结果
# =============================================================================

# 检查训练结果
# 如果正常执行，检查退出码
# 🔥 安全检查：确保TRAIN_EXIT_CODE不为空
if [ -z "$TRAIN_EXIT_CODE" ]; then
    TRAIN_EXIT_CODE=0  # 如果为空，默认为成功
fi

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
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
# 第九部分：自动记录结果到Excel
# =============================================================================

# 创建Python脚本来解析训练日志并记录到Excel
cat > "$EXPERIMENT_DIR/record_results.py" << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动记录训练结果到Excel
"""

import pandas as pd
import re
import sys
import os
from datetime import datetime

def parse_training_log(log_file):
    """解析训练日志，提取最佳结果"""
    results = {
        'mAP': 0.0,
        'Rank-1': 0.0,
        'Rank-5': 0.0,
        'Rank-10': 0.0,
        'Best_mAP': 0.0,
        'Best_Rank-1': 0.0,
        'Best_Rank-5': 0.0,
        'Best_Rank-10': 0.0,
        '滑动窗口尺度': '',
        '拼接方式': '',
        '专家权重占比': '',
        '4x4专家权重': 0.0,
        '8x8专家权重': 0.0,
        '16x16专家权重': 0.0
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取最终结果
        mAP_match = re.search(r'mAP: ([\d.]+)%', content)
        if mAP_match:
            results['mAP'] = float(mAP_match.group(1))
            
        rank1_match = re.search(r'Rank-1\s*:([\d.]+)%', content)
        if rank1_match:
            results['Rank-1'] = float(rank1_match.group(1))
            
        rank5_match = re.search(r'Rank-5\s*:([\d.]+)%', content)
        if rank5_match:
            results['Rank-5'] = float(rank5_match.group(1))
            
        rank10_match = re.search(r'Rank-10\s*:([\d.]+)%', content)
        if rank10_match:
            results['Rank-10'] = float(rank10_match.group(1))
            
        # 提取最佳结果
        best_mAP_match = re.search(r'Best mAP: ([\d.]+)%', content)
        if best_mAP_match:
            results['Best_mAP'] = float(best_mAP_match.group(1))
            
        best_rank1_match = re.search(r'Best Rank-1: ([\d.]+)%', content)
        if best_rank1_match:
            results['Best_Rank-1'] = float(best_rank1_match.group(1))
            
        best_rank5_match = re.search(r'Best Rank-5: ([\d.]+)%', content)
        if best_rank5_match:
            results['Best_Rank-5'] = float(best_rank5_match.group(1))
            
        best_rank10_match = re.search(r'Best Rank-10: ([\d.]+)%', content)
        if best_rank10_match:
            results['Best_Rank-10'] = float(best_rank10_match.group(1))
            
        # 提取滑动窗口尺度信息
        window_scale_match = re.search(r'滑动窗口尺度: \[([\d, ]+)\]', content)
        if window_scale_match:
            results['滑动窗口尺度'] = window_scale_match.group(1).strip()
        else:
            # 从命令行参数中提取
            if 'CLIP_MULTI_SCALE_SCALES' in content:
                scale_match = re.search(r'CLIP_MULTI_SCALE_SCALES \[([\d, ]+)\]', content)
                if scale_match:
                    results['滑动窗口尺度'] = scale_match.group(1).strip()
            
        # 提取拼接方式信息
        if '门控融合机制：已启用' in content:
            results['拼接方式'] = '门控融合'
        elif '门控融合机制：已禁用' in content:
            results['拼接方式'] = '简单拼接'
        else:
            results['拼接方式'] = '简单拼接'  # 默认
            
        # 提取第一次和最后一次专家权重占比信息
        first_weight_match = re.search(r'第一次专家权重分布: \[([\d., ]+)\]', content)
        last_weight_match = re.search(r'最后一次专家权重分布: \[([\d., ]+)\]', content)
        
        # 调试：检查日志文件中的权重信息
        print(f"🔍 调试：检查日志文件中的权重信息")
        if '第一次专家权重分布' in content:
            print(f"✅ 找到第一次专家权重分布")
        else:
            print(f"❌ 未找到第一次专家权重分布")
            
        if '最后一次专家权重分布' in content:
            print(f"✅ 找到最后一次专家权重分布")
        else:
            print(f"❌ 未找到最后一次专家权重分布")
            
        if '专家权重分布:' in content:
            print(f"✅ 找到专家权重分布信息")
            # 显示找到的权重信息
            weight_matches = re.findall(r'专家权重分布: \[([\d., ]+)\]', content)
            print(f"🔍 找到的权重信息: {weight_matches}")
        else:
            print(f"❌ 未找到任何专家权重分布信息")
        
        if first_weight_match and last_weight_match:
            # 提取第一次权重
            first_weights_str = first_weight_match.group(1)
            first_weights = [float(x.strip()) for x in first_weights_str.split(',')]
            
            # 提取最后一次权重
            last_weights_str = last_weight_match.group(1)
            last_weights = [float(x.strip()) for x in last_weights_str.split(',')]
            
            if len(first_weights) >= 3 and len(last_weights) >= 3:
                # 记录最后一次权重（用于Excel）
                results['4x4专家权重'] = last_weights[0] * 100  # 转换为百分比
                results['8x8专家权重'] = last_weights[1] * 100
                results['16x16专家权重'] = last_weights[2] * 100
                results['专家权重占比'] = f"4x4:{last_weights[0]*100:.1f}%, 8x8:{last_weights[1]*100:.1f}%, 16x16:{last_weights[2]*100:.1f}%"
                
                # 记录首次和末次权重占比
                results['首次专家权重占比'] = f"4x4:{first_weights[0]*100:.1f}%, 8x8:{first_weights[1]*100:.1f}%, 16x16:{first_weights[2]*100:.1f}%"
                results['末次专家权重占比'] = f"4x4:{last_weights[0]*100:.1f}%, 8x8:{last_weights[1]*100:.1f}%, 16x16:{last_weights[2]*100:.1f}%"
                
                # 计算权重变化
                weight_change = [last_weights[i] - first_weights[i] for i in range(3)]
                results['权重变化'] = f"4x4:{weight_change[0]*100:+.1f}%, 8x8:{weight_change[1]*100:+.1f}%, 16x16:{weight_change[2]*100:+.1f}%"
                
                print(f"📊 第一次专家权重: [{first_weights[0]:.4f}, {first_weights[1]:.4f}, {first_weights[2]:.4f}]")
                print(f"📊 最后一次专家权重: [{last_weights[0]:.4f}, {last_weights[1]:.4f}, {last_weights[2]:.4f}]")
                print(f"📈 权重变化: [{weight_change[0]:+.4f}, {weight_change[1]:+.4f}, {weight_change[2]:+.4f}]")
            else:
                print("警告: 权重数量不足")
        else:
            # 备用方案：提取任何权重信息
            print(f"🔍 调试：使用备用方案提取权重信息")
            expert_weight_patterns = [
                r'专家权重分布: \[([\d., ]+)\]',
                r'expert weights: \[([\d., ]+)\]',
                r'MoE weights: \[([\d., ]+)\]',
                r'权重分布: \[([\d., ]+)\]',
                r'专家权重: \[([\d., ]+)\]',
                r'权重: \[([\d., ]+)\]'
            ]
            
            expert_weight_match = None
            for i, pattern in enumerate(expert_weight_patterns):
                expert_weight_match = re.search(pattern, content)
                if expert_weight_match:
                    print(f"✅ 使用模式 {i+1} 找到权重信息: {expert_weight_match.group(1)}")
                    break
                else:
                    print(f"❌ 模式 {i+1} 未匹配")
            
            if expert_weight_match:
                weights_str = expert_weight_match.group(1)
                weights = [float(x.strip()) for x in weights_str.split(',')]
                if len(weights) >= 3:
                    results['4x4专家权重'] = weights[0] * 100
                    results['8x8专家权重'] = weights[1] * 100
                    results['16x16专家权重'] = weights[2] * 100
                    results['专家权重占比'] = f"4x4:{weights[0]*100:.1f}%, 8x8:{weights[1]*100:.1f}%, 16x16:{weights[2]*100:.1f}%"
                    print(f"📊 备用方案 - 专家权重: [{weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}]")
                else:
                    print(f"警告: 权重数量不足，找到 {len(weights)} 个权重")
            else:
                print("警告: 未找到专家权重信息")
                # 显示日志文件的前几行和后几行，帮助调试
                print(f"🔍 日志文件前5行:")
                for i, line in enumerate(content.split('\n')[:5]):
                    print(f"  {i+1}: {line}")
                print(f"🔍 日志文件后5行:")
                for i, line in enumerate(content.split('\n')[-5:]):
                    print(f"  {len(content.split('\n'))-5+i+1}: {line}")
            
    except Exception as e:
        print(f"解析日志文件时出错: {e}")
        
    return results

def extract_dataset_info(command_line):
    """从命令行中提取数据集信息"""
    dataset = "Unknown"
    
    # 从命令行中提取数据集信息
    if "RGBNT100" in command_line:
        dataset = "RGBNT100"
    elif "RGBNT201" in command_line:
        dataset = "RGBNT201"
    elif "MSVR310" in command_line:
        dataset = "MSVR310"
    elif "Market1501" in command_line:
        dataset = "Market1501"
    elif "DukeMTMC" in command_line:
        dataset = "DukeMTMC"
    elif "MSMT17" in command_line:
        dataset = "MSMT17"
    
    return dataset

def update_excel_results(experiment_dir, command_line, results):
    """更新Excel结果文件"""
    excel_file = "experiment_results.xlsx"
    
    # 提取数据集信息
    dataset = extract_dataset_info(command_line)
    
    # 准备新记录
    new_record = {
        '实验时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '数据集': dataset,
        '实验目录': experiment_dir,
        '命令行': command_line,
        '滑动窗口尺度': results['滑动窗口尺度'],
        '拼接方式': results['拼接方式'],
        '专家权重占比': results['专家权重占比'],
        '4x4专家权重': results['4x4专家权重'],
        '8x8专家权重': results['8x8专家权重'],
        '16x16专家权重': results['16x16专家权重'],
        '首次专家权重占比': results.get('首次专家权重占比', ''),
        '末次专家权重占比': results.get('末次专家权重占比', ''),
        '权重变化': results.get('权重变化', ''),
        'mAP': results['mAP'],
        'Rank-1': results['Rank-1'],
        'Rank-5': results['Rank-5'],
        'Rank-10': results['Rank-10'],
        'Best_mAP': results['Best_mAP'],
        'Best_Rank-1': results['Best_Rank-1'],
        'Best_Rank-5': results['Best_Rank-5'],
        'Best_Rank-10': results['Best_Rank-10']
    }
    
    try:
        # 如果Excel文件存在，读取现有数据
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
        else:
            # 创建新的DataFrame
            df = pd.DataFrame(columns=[
                '实验时间', '数据集', '实验目录', '命令行', '滑动窗口尺度', '拼接方式', '专家权重占比',
                '4x4专家权重', '8x8专家权重', '16x16专家权重', '首次专家权重占比', '末次专家权重占比', '权重变化',
                'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'Best_mAP', 'Best_Rank-1', 'Best_Rank-5', 'Best_Rank-10'
            ])
        
        # 添加新记录
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
        # 保存到Excel
        df.to_excel(excel_file, index=False)
        print(f"✅ 结果已记录到 {excel_file}")
        
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python record_results.py <实验目录> <命令行>")
        sys.exit(1)
        
    experiment_dir = sys.argv[1]
    command_line = sys.argv[2]
    log_file = os.path.join(experiment_dir, "logs", "train_log.txt")
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        sys.exit(1)
        
    # 解析结果
    results = parse_training_log(log_file)
    
    # 更新Excel
    update_excel_results(experiment_dir, command_line, results)
    
    # 打印结果摘要
    print(f"📊 实验结果摘要:")
    print(f"   滑动窗口尺度: {results['滑动窗口尺度']}")
    print(f"   拼接方式: {results['拼接方式']}")
    print(f"   专家权重占比: {results['专家权重占比']}")
    print(f"   mAP: {results['mAP']:.1f}%")
    print(f"   Rank-1: {results['Rank-1']:.1f}%")
    print(f"   Rank-5: {results['Rank-5']:.1f}%")
    print(f"   Rank-10: {results['Rank-10']:.1f}%")
    print(f"   Best mAP: {results['Best_mAP']:.1f}%")
EOF

# 执行结果记录
echo "📊 自动记录结果到Excel..."
python3 "$EXPERIMENT_DIR/record_results.py" "$EXPERIMENT_DIR" "$CMD"

# =============================================================================
# 第十部分：结束
# =============================================================================

# 显示实验记录完成信息
echo "🏆 实验记录完成！"
