# Category: dev_utils (开发调试)
# Description: 开发辅助工具，包括进程清理、层输出调试、环境诊断及后端 API

#!/bin/bash
# 诊断 --use_moe 参数问题

cd /home/zhanghaoyang/Desktop/yzy/AboutReid

echo "=========================================="
echo "🔍 诊断 --use_moe 参数问题"
echo "=========================================="
echo ""

# 1. 检查 Python 环境
echo "1. 检查 Python 环境："
echo "   Python 路径: $(which python)"
echo "   Python 版本: $(python --version)"
echo ""

# 2. 检查 train_net.py 文件
echo "2. 检查 train_net.py 文件："
if [ -f "train_net.py" ]; then
    echo "   ✅ train_net.py 存在"
    echo "   文件路径: $(pwd)/train_net.py"
    echo "   修改时间: $(stat -c %y train_net.py)"
else
    echo "   ❌ train_net.py 不存在"
fi
echo ""

# 3. 检查 --use_moe 参数是否定义
echo "3. 检查 --use_moe 参数定义："
if grep -q "add_argument.*--use_moe" train_net.py; then
    echo "   ✅ --use_moe 参数已定义"
    grep -n "add_argument.*--use_moe" train_net.py | head -1
else
    echo "   ❌ --use_moe 参数未定义"
fi
echo ""

# 4. 检查 opts 参数位置
echo "4. 检查 opts 参数位置："
opts_line=$(grep -n "add_argument.*opts.*REMAINDER" train_net.py | head -1 | cut -d: -f1)
use_moe_line=$(grep -n "add_argument.*--use_moe" train_net.py | head -1 | cut -d: -f1)
if [ -n "$opts_line" ] && [ -n "$use_moe_line" ]; then
    if [ "$opts_line" -gt "$use_moe_line" ]; then
        echo "   ✅ opts 在 --use_moe 之后（正确）"
        echo "   --use_moe 行号: $use_moe_line"
        echo "   opts 行号: $opts_line"
    else
        echo "   ❌ opts 在 --use_moe 之前（错误）"
        echo "   --use_moe 行号: $use_moe_line"
        echo "   opts 行号: $opts_line"
    fi
else
    echo "   ⚠️  无法确定参数位置"
fi
echo ""

# 5. 清除 Python 缓存
echo "5. 清除 Python 缓存："
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   ✅ 缓存已清除"
echo ""

# 6. 测试 --use_moe 参数
echo "6. 测试 --use_moe 参数："
if python train_net.py --help 2>&1 | grep -q "use_moe"; then
    echo "   ✅ --use_moe 参数在 help 中可见"
    python train_net.py --help 2>&1 | grep "use_moe" | head -1
else
    echo "   ❌ --use_moe 参数在 help 中不可见"
fi
echo ""

# 7. 实际测试命令
echo "7. 实际测试命令："
echo "   运行: python train_net.py --config_file configs/RGBNT100/jzb_baseline_optimize.yml --use_moe"
python train_net.py --config_file configs/RGBNT100/jzb_baseline_optimize.yml --use_moe 2>&1 | head -5
echo ""

echo "=========================================="
echo "✅ 诊断完成"
echo "=========================================="
