#!/bin/bash
# ================================================================================
#                    AboutReid V4 命令行指南
# ================================================================================
# 版本: V4.0 - IDEA风格文本处理 + AboutReid灵活融合 + CDA跨模态增强
# 更新时间: 2025.12.24
# 使用前请确保在MambaPro环境下运行
# ================================================================================

echo "🎯 AboutReid V4 命令行指南"
echo "=========================="

# ================================================================================
# 0. 环境准备和验证
# ================================================================================

echo "
🔧 步骤0: 环境检查和准备
"

# 0.1 检查当前环境
echo "📍 当前工作目录:"
pwd

echo "
🐍 Python环境检查:"
python --version
echo "CUDA版本:"
nvidia-smi | grep "CUDA Version" | head -1

echo "
📦 依赖包检查:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

# 0.2 数据集验证
echo "
📊 数据集验证:"
echo "检查数据集文件..."
ls -la data/datasets/RGBNT201/train_171/ | wc -l
echo "检查文本数据..."
ls -la data/datasets/QwenVL_Anno/RGBNT201/text/

# 0.3 模型组件验证
echo "
🤖 模型组件验证:"
python -c "
import sys
sys.path.append('.')
try:
    from data.datasets.RGBNT201_IDEA_Text import RGBNT201_IDEA_Text
    from modeling.clip.idea_text_encoder import IDEATextEncoder
    from modeling.idea_meta_arch import build_transformer
    from modeling.fusion_part.CDA_Module import CDA_Module
    print('✅ 所有V4核心组件导入成功')
except Exception as e:
    print(f'❌ 组件导入失败: {e}')
"

# ================================================================================
# 1. 训练命令
# ================================================================================

echo "
🚀 步骤1: 模型训练
"

# 1.1 标准训练命令
echo "1.1 标准V4训练 (单GPU):"
echo "CUDA_VISIBLE_DEVICES=0 python train_net.py \\
    --config-file configs/RGBNT201/IDEA_style.yml \\
    MODEL.TRANSFORMER_TYPE ViT-B-16 \\
    DATASETS.NAMES RGBNT201_IDEA"

# 1.2 多GPU训练
echo "
1.2 多GPU训练:"
echo "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \\
    --nproc_per_node=2 \\
    --master_port=12345 \\
    train_net.py \\
    --config-file configs/RGBNT201/IDEA_style.yml \\
    MODEL.TRANSFORMER_TYPE ViT-B-16 \\
    DATASETS.NAMES RGBNT201_IDEA \\
    SOLVER.IMS_PER_BATCH 128"

# 1.3 自定义配置训练
echo "
1.3 自定义配置训练:"
echo "CUDA_VISIBLE_DEVICES=0 python train_net.py \\
    --config-file configs/RGBNT201/IDEA_style.yml \\
    MODEL.TRANSFORMER_TYPE ViT-B-16 \\
    MODEL.TEXT_PROMPT 4 \\
    MODEL.PREFIX True \\
    SOLVER.BASE_LR 0.0005 \\
    SOLVER.MAX_EPOCHS 60 \\
    DATASETS.NAMES RGBNT201_IDEA"

# 1.4 调试模式训练
echo "
1.4 调试模式训练 (小批量，快速验证):"
echo "CUDA_VISIBLE_DEVICES=0 python train_net.py \\
    --config-file configs/RGBNT201/IDEA_style.yml \\
    MODEL.TRANSFORMER_TYPE ViT-B-16 \\
    SOLVER.IMS_PER_BATCH 16 \\
    SOLVER.MAX_EPOCHS 2 \\
    DATASETS.NAMES RGBNT201_IDEA"

# ================================================================================
# 2. 验证和测试命令
# ================================================================================

echo "
📊 步骤2: 模型验证和测试
"

# 2.1 训练后验证
echo "2.1 验证训练结果:"
echo "CUDA_VISIBLE_DEVICES=0 python train_net.py \\
    --config-file configs/RGBNT201/IDEA_style.yml \\
    --eval-only \\
    MODEL.WEIGHT outputs/RGBNT201_IDEA_style/model_best.pth"

# 2.2 指定checkpoint验证
echo "
2.2 指定checkpoint验证:"
echo "CUDA_VISIBLE_DEVICES=0 python train_net.py \\
    --config-file configs/RGBNT201/IDEA_style.yml \\
    --eval-only \\
    MODEL.WEIGHT path/to/your/checkpoint.pth"

# 2.3 测试模式运行
echo "
2.3 完整测试流程:"
echo "CUDA_VISIBLE_DEVICES=0 python test.py \\
    --config-file configs/RGBNT201/IDEA_style.yml \\
    --checkpoint outputs/RGBNT201_IDEA_style/model_best.pth \\
    --output-dir results/V4_test_results"

# ================================================================================
# 3. 调试和分析命令
# ================================================================================

echo "
🔍 步骤3: 调试和性能分析
"

# 3.1 数据加载器测试
echo "3.1 测试数据加载器:"
echo "python -c \"
import sys
sys.path.append('.')
from data.datasets.make_dataloader import make_dataloader
import yaml

# 加载配置
with open('configs/RGBNT201/IDEA_style.yml', 'r') as f:
    cfg = yaml.safe_load(f)

# 创建数据加载器
train_loader, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
print(f'✅ 数据加载器创建成功')
print(f'训练批次: {len(train_loader)}')
print(f'验证批次: {len(val_loader)}')
print(f'查询样本数: {num_query}')
print(f'类别数: {num_classes}')
\""

# 3.2 模型结构检查
echo "
3.2 检查模型结构:"
echo "python -c \"
import sys
sys.path.append('.')
from modeling.idea_meta_arch import build_transformer
import yaml

# 加载配置
with open('configs/RGBNT201/IDEA_style.yml', 'r') as f:
    cfg = yaml.safe_load(f)

# 创建模型
model = build_transformer(
    num_classes=751,
    cfg=type('Config', (), cfg)(),
    camera_num=4,
    view_num=0,
    factory=None,
    feat_dim=512
)

print(f'✅ 模型创建成功')
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
\""

# 3.3 内存使用分析
echo "
3.3 GPU内存使用分析:"
echo "CUDA_VISIBLE_DEVICES=0 python -c \"
import torch
import gc
print(f'GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
print(f'GPU内存缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB')
\""

# 3.4 训练日志监控
echo "
3.4 实时监控训练日志:"
echo "tail -f outputs/RGBNT201_IDEA_style/log.txt"

# ================================================================================
# 4. 批量实验命令
# ================================================================================

echo "
🧪 步骤4: 批量实验管理
"

# 4.1 参数消融实验
echo "4.1 TEXT_PROMPT参数消融:"
echo "# 不同可学习提示数量的对比
for prompt_num in 0 2 4 8; do
    echo \"训练 TEXT_PROMPT=\$prompt_num\"
    CUDA_VISIBLE_DEVICES=0 python train_net.py \\
        --config-file configs/RGBNT201/IDEA_style.yml \\
        MODEL.TEXT_PROMPT \$prompt_num \\
        OUTPUT_DIR outputs/V4_prompt_\${prompt_num} \\
        DATASETS.NAMES RGBNT201_IDEA
done"

# 4.2 学习率实验
echo "
4.2 学习率对比实验:"
echo "# 不同学习率的对比
for lr in 0.0001 0.00035 0.0005 0.001; do
    echo \"训练 LR=\$lr\"
    CUDA_VISIBLE_DEVICES=0 python train_net.py \\
        --config-file configs/RGBNT201/IDEA_style.yml \\
        SOLVER.BASE_LR \$lr \\
        OUTPUT_DIR outputs/V4_lr_\${lr} \\
        DATASETS.NAMES RGBNT201_IDEA
done"

# ================================================================================
# 5. 实用工具命令
# ================================================================================

echo "
🛠️ 步骤5: 实用工具
"

# 5.1 清理缓存
echo "5.1 清理PyTorch缓存:"
echo "rm -rf ~/.cache/torch/hub/checkpoints/*"
echo "python -c \"import torch; torch.cuda.empty_cache()\""

# 5.2 检查磁盘空间
echo "
5.2 检查磁盘使用情况:"
echo "df -h | grep -E \"(Filesystem|${PWD:1:1})\")"

# 5.3 进程监控
echo "
5.3 监控训练进程:"
echo "ps aux | grep python | grep train_net"
echo "nvidia-smi"

# 5.4 日志分析
echo "
5.4 分析训练日志:"
echo "# 查看最新训练指标
tail -20 outputs/RGBNT201_IDEA_style/log.txt | grep -E \"(Epoch|Rank-1|mAP)\"

# 绘制损失曲线
python -c \"
import matplotlib.pyplot as plt
# 读取日志文件并绘制曲线
\""

# ================================================================================
# 6. 快速开始模板
# ================================================================================

echo "
⚡ 步骤6: 快速开始模板
"

cat << 'EOF'
# 快速开始脚本 (复制到终端执行)

# 1. 激活环境
conda activate MambaPro

# 2. 进入项目目录
cd /home/zhanghaoyang/Desktop/yzy/AboutReid

# 3. 开始训练
CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --config-file configs/RGBNT201/IDEA_style.yml \
    MODEL.TRANSFORMER_TYPE ViT-B-16 \
    DATASETS.NAMES RGBNT201_IDEA \
    OUTPUT_DIR outputs/V4_quick_start

# 4. 监控训练 (新终端)
tail -f outputs/V4_quick_start/log.txt

# 5. 验证结果 (训练完成后)
CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --config-file configs/RGBNT201/IDEA_style.yml \
    --eval-only \
    MODEL.WEIGHT outputs/V4_quick_start/model_best.pth
EOF

echo "
🎉 V4命令行指南准备完成！

💡 使用建议:
1. 首次运行建议使用调试模式验证环境
2. 正式训练前检查GPU内存是否充足
3. 定期备份重要的checkpoint文件
4. 监控训练日志，及时发现问题

🚀 祝训练顺利！如有问题请查看日志文件。
"