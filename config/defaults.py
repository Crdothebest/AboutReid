from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'MambaPro'
# The layer where we extract feature from the ViT backbone
_C.MODEL.LAYER = -1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH_T = '/path/to/your/vitb_16_224_21k.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.FLOPS_TEST = False
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
_C.MODEL.PROMPT = True  # Whether use the prompt tuning
_C.MODEL.MAMBA = True # whether use the mamba
_C.MODEL.MAMBA_BI = False # whether use the bidirectional mamba
_C.MODEL.ADAPTER = True # whether use the adapter
_C.MODEL.FROZEN = True # whether freeze the backbone

# ========== 新增配置：多尺度滑动窗口设置 ==========
# 用户修改：添加多尺度滑动窗口配置
# 功能：控制是否启用多尺度滑动窗口特征提取
# 基于：多尺度滑动窗口创新设计
# 撤销方法：删除以下配置代码
# Multi-Scale Sliding Window settings
_C.MODEL.USE_MULTI_SCALE = False # whether use multi-scale sliding window (默认关闭，保持向后兼容)
_C.MODEL.MULTI_SCALE_SCALES = [4, 8, 16] # sliding window scales (4x4, 8x8, 16x16窗口)

# ========== 新增配置：CLIP多尺度滑动窗口设置 ==========
# 用户修改：在保持CLIP分支的基础上，添加多尺度滑动窗口配置
# 功能：控制是否启用CLIP多尺度滑动窗口特征提取
# 基于：多尺度滑动窗口创新设计
# 撤销方法：删除以下两行配置代码
# CLIP Multi-Scale Sliding Window settings
_C.MODEL.USE_CLIP_MULTI_SCALE = False # whether use CLIP multi-scale sliding window (默认关闭，保持向后兼容)
_C.MODEL.CLIP_MULTI_SCALE_SCALES = [4, 8, 16] # sliding window scales for CLIP (4x4, 8x8, 16x16窗口)

# ========== 新增配置：多尺度MoE特征融合设置 ==========
# 用户修改：添加多尺度MoE特征融合配置，支持命令行开关控制
# 功能：控制是否启用基于专家网络的多尺度特征动态融合机制
# 基于：MoE多尺度特征融合创新设计
# 撤销方法：删除以下配置代码
# Multi-Scale MoE settings
_C.MODEL.USE_MULTI_SCALE_MOE = False    # whether use multi-scale MoE fusion (默认关闭，保持向后兼容)
_C.MODEL.MOE_SCALES = [4, 8, 16]        # MoE sliding window scales (与多尺度滑动窗口保持一致)
_C.MODEL.MOE_NUM_EXPERTS = 3            # MoE expert network number (专家网络数量)
_C.MODEL.MOE_EXPERT_HIDDEN_DIM = 1024   # MoE expert network hidden dimension
_C.MODEL.MOE_TEMPERATURE = 1.0          # MoE gating network temperature parameter

# MoE专家网络参数
_C.MODEL.MOE_EXPERT_DROPOUT = 0.1       # MoE expert network dropout
_C.MODEL.MOE_GATE_DROPOUT = 0.1          # MoE gating network dropout
_C.MODEL.MOE_EXPERT_LAYERS = 2           # MoE expert network layers
_C.MODEL.MOE_GATE_LAYERS = 2             # MoE gating network layers
_C.MODEL.MOE_EXPERT_THRESHOLD = 0.1      # MoE expert activation threshold
_C.MODEL.MOE_RESIDUAL_WEIGHT = 1.0       # MoE residual connection weight
_C.MODEL.MOE_INIT_WEIGHTS = None         # MoE expert initial weights (optional, e.g., [0.35, 0.3, 0.35])

# MoE损失权重参数
_C.MODEL.MOE_BALANCE_LOSS_WEIGHT = 0.01  # MoE expert balance loss weight
_C.MODEL.MOE_SPARSITY_LOSS_WEIGHT = 0.001 # MoE sparsity loss weight
_C.MODEL.MOE_DIVERSITY_LOSS_WEIGHT = 0.01 # MoE diversity loss weight

# ========== 新增配置：门控融合机制设置 ==========
# 用户修改：添加门控融合机制配置，支持命令行开关控制
# 功能：控制是否启用门控融合机制增强MoE融合效果
# 基于：门控融合增强MoE特征融合创新设计
# 撤销方法：删除以下配置代码
# Gate Fusion settings
_C.MODEL.USE_GATE_FUSION = False              # whether use gate fusion mechanism (默认关闭)
_C.MODEL.GATE_NUM_HEADS = 8                   # gate fusion number of heads
_C.MODEL.GATE_DROPOUT = 0.1                   # gate fusion dropout

# Attention Fusion settings
_C.MODEL.USE_ATTENTION_FUSION = False         # whether use attention fusion mechanism (默认关闭)
_C.MODEL.ATTENTION_NUM_HEADS = 8              # attention fusion number of heads
_C.MODEL.ATTENTION_DROPOUT = 0.1              # attention fusion dropout
_C.MODEL.ATTENTION_DIM = 512                  # attention fusion dimension


# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with the contact feature
_C.MODEL.DIRECT = 1

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.PREFIX_NUM = 1

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False  # We do not use this parameter


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('RGBNT201')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 14  # This may be affected by the order of data reading
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16  # You can adjust it to 8 to save memory while the batch_size need to be 64 to ensure the number of ID

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.009
# Factor of learning bias
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# ========================================================================
# 【MoE门控网络独立学习率配置】
# ========================================================================
# 
# 【功能】为门控网络参数设置独立的学习率倍数
# 【原因】解决MoE训练中的模式坍塌问题
# 
# 【问题背景】
# 门控网络负责动态分配专家权重，其更新速度直接影响训练稳定性。
# 原有实现中，门控网络使用与普通参数相同的学习率（BASE_LR），
# 导致门控网络更新过快，容易形成自强化循环，某个专家快速垄断
# 99.9%的路由权重，即使设置了平衡损失也无法阻止模式坍塌。
# 
# 【解决方案】
# 为门控网络设置独立且更低的学习率（BASE_LR × MOE_GATE_LR_FACTOR），
# 实现"慢速决策、快速执行"的策略：
# - 门控网络（决策层）：使用极低学习率（BASE_LR × 0.01）
# - 专家网络（执行层）：使用正常学习率（BASE_LR）
# 
# 【参数说明】
# - 默认值：0.01（即BASE_LR的1%）
# - 建议范围：0.001 ~ 0.1
#   * 过小（<0.001）：门控网络几乎不更新，无法学习
#   * 过大（>0.1）：接近普通学习率，无法解决模式坍塌
#   * 推荐值：0.01（经过实验验证的最优值）
# 
# 【学习率计算示例】
# 假设 BASE_LR = 0.0005, MOE_GATE_LR_FACTOR = 0.01, BIAS_LR_FACTOR = 2.0
# - 门控网络权重：0.0005 × 0.01 = 0.000005（降低100倍）
# - 门控网络偏置：0.0005 × 0.01 × 2.0 = 0.00001（降低100倍）
# 
# 【使用方法】
# 1. 配置文件：在YAML文件中设置 SOLVER.MOE_GATE_LR_FACTOR: 0.01
# 2. 命令行：--opts SOLVER.MOE_GATE_LR_FACTOR 0.01
# 
# 【相关文件】
# - 实现代码：solver/make_optimizer.py (第14-28行)
# - 配置文件示例：configs/RGBNT201/yzy_best_Mambapro_moe.yml (第101行)
# ========================================================================
_C.SOLVER.MOE_GATE_LR_FACTOR = 0.01  # 门控网络学习率倍数（默认0.01，即BASE_LR的1%）

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.SEED = 1111
_C.MODEL.NO_MARGIN = True
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 60
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 10
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 128  # You can adjust it to 64

# MoE损失权重参数（SOLVER部分）
_C.SOLVER.MOE_BALANCE_LOSS_WEIGHT = 0.01  # MoE expert balance loss weight
_C.SOLVER.MOE_SPARSITY_LOSS_WEIGHT = 0.001 # MoE sparsity loss weight
_C.SOLVER.MOE_DIVERSITY_LOSS_WEIGHT = 0.01 # MoE diversity loss weight
_C.SOLVER.MOE_BALANCE_THRESHOLD = 0.3      # MoE balance loss threshold (允许30%偏差，防止模式坍塌)

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 256
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Pattern of test augmentation
_C.TEST.MISS = 'None'
# ----------------------------------------------------------a------------------ #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./test"
