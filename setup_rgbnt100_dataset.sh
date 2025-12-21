#!/bin/bash
# RGBNT100 数据集设置脚本
# 功能：创建符号链接，使配置文件中的路径生效

echo "🔧 设置 RGBNT100 数据集路径"
echo "============================================================"

# 实际数据集位置
ACTUAL_DATASET="/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT100"

# 配置文件期望的位置
EXPECTED_ROOT="/home/zubuntu/workspace/MambaPro/MambaPro/data"
EXPECTED_DATASET="${EXPECTED_ROOT}/RGBNT100"

# 检查实际数据集是否存在
if [ ! -d "$ACTUAL_DATASET" ]; then
    echo "❌ 实际数据集不存在: $ACTUAL_DATASET"
    exit 1
fi

echo "✅ 实际数据集位置: $ACTUAL_DATASET"

# 检查期望的根目录是否存在
if [ ! -d "$EXPECTED_ROOT" ]; then
    echo "⚠️  期望的根目录不存在: $EXPECTED_ROOT"
    echo "   创建目录..."
    mkdir -p "$EXPECTED_ROOT"
fi

# 检查符号链接是否已存在
if [ -L "$EXPECTED_DATASET" ]; then
    echo "✅ 符号链接已存在: $EXPECTED_DATASET"
    echo "   指向: $(readlink -f $EXPECTED_DATASET)"
elif [ -d "$EXPECTED_DATASET" ]; then
    echo "⚠️  目录已存在（非符号链接）: $EXPECTED_DATASET"
    echo "   建议删除后重新创建符号链接"
else
    echo "📝 创建符号链接..."
    ln -s "$ACTUAL_DATASET" "$EXPECTED_DATASET"
    if [ $? -eq 0 ]; then
        echo "✅ 符号链接创建成功: $EXPECTED_DATASET -> $ACTUAL_DATASET"
    else
        echo "❌ 符号链接创建失败"
        exit 1
    fi
fi

# 验证数据集结构
echo ""
echo "📊 验证数据集结构:"
DATASET_RGBIR="${EXPECTED_DATASET}/rgbir"
if [ -d "$DATASET_RGBIR" ]; then
    echo "✅ rgbir 目录存在"
    
    for split in "bounding_box_train" "query" "bounding_box_test"; do
        split_dir="${DATASET_RGBIR}/${split}"
        if [ -d "$split_dir" ]; then
            count=$(find "$split_dir" -name "*.jpg" | wc -l)
            echo "  ✅ $split: $count 张图像"
        else
            echo "  ❌ $split: 目录不存在"
        fi
    done
else
    echo "❌ rgbir 目录不存在"
fi

echo ""
echo "============================================================"
echo "✅ 数据集设置完成！"
echo "============================================================"
