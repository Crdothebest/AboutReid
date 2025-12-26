"""
统一配置输出模块

功能：在训练开始前，统一输出最终配置信息（考虑命令行覆盖）
要求：精简、清晰、只输出重要信息
"""

def print_final_config(cfg):
    """
    打印最终配置信息（考虑命令行覆盖后的配置）
    
    Args:
        cfg: YACS配置对象
    """
    print("=" * 40)
    print("📋 最终训练配置（已考虑命令行覆盖）")
    print("=" * 40)
    
    # ========== 1. 基础配置 ==========
    print("\n【基础配置 BaseConfig】")
    print(f"  - 输出目录 OutputDir = {cfg.OUTPUT_DIR}")
    print(f"  - 随机种子 Seed = {cfg.SOLVER.SEED}")
    print(f"  - 最大Epoch MaxEpoch = {cfg.SOLVER.MAX_EPOCHS}")
    print(f"  - 基础学习率 BaseLR = {cfg.SOLVER.BASE_LR}")
    print(f"  - 优化器 Optimizer = {cfg.SOLVER.OPTIMIZER_NAME}")
    print(f"  - 学习率调度 LR_Schedule = Steps{cfg.SOLVER.STEPS}, Gamma={cfg.SOLVER.GAMMA}")
    print(f"  - Warmup WarmupIters = {cfg.SOLVER.WARMUP_ITERS}, WarmupFactor = {cfg.SOLVER.WARMUP_FACTOR}")

    # 预训练模型路径
    pretrain_path = getattr(cfg.MODEL, 'PRETRAIN_PATH_T', 'None')
    print(f"  - 预训练模型路径 PretrainPath (MODEL.PRETRAIN_PATH_T) = {pretrain_path}")

    # 设备配置
    device_id = getattr(cfg.MODEL, 'DEVICE_ID', 'auto')
    print(f"  - GPU设备 DeviceID (MODEL.DEVICE_ID) = {device_id}")
    
    # ========== 2. 模型结构 ==========
    print("\n【模型结构 ModelConfig】")
    print(f"  - 骨干 Backbone (MODEL.TRANSFORMER_TYPE) = {cfg.MODEL.TRANSFORMER_TYPE}")
    multi_scale_status = "✅ ENABLED" if cfg.MODEL.USE_CLIP_MULTI_SCALE else "❌ DISABLED"
    print(f"  - 多尺度滑动窗口 MultiScaleWindow (MODEL.USE_CLIP_MULTI_SCALE) = {multi_scale_status}")
    if cfg.MODEL.USE_CLIP_MULTI_SCALE:
        scales = cfg.MODEL.CLIP_MULTI_SCALE_SCALES
        # 将尺度列表转换为友好的显示格式，例如 [4] -> "4x4窗口", [4,8] -> "4x4+8x8窗口"
        if isinstance(scales, (list, tuple)):
            scale_labels = [f"{s}x{s}" for s in scales]
            scale_display = "+".join(scale_labels) + "窗口"
        else:
            scale_display = str(scales)
        print(f"    * 滑动窗口尺度 WindowScales (MODEL.CLIP_MULTI_SCALE_SCALES) = {scales} ({scale_display})")
    
    moe_status = "✅ ENABLED" if cfg.MODEL.USE_MULTI_SCALE_MOE else "❌ DISABLED"
    print(f"  - MoE特征融合 MultiScaleMoE (MODEL.USE_MULTI_SCALE_MOE) = {moe_status}")
    if cfg.MODEL.USE_MULTI_SCALE_MOE:
        moe_scales = cfg.MODEL.MOE_SCALES
        # 将尺度列表转换为友好的显示格式
        if isinstance(moe_scales, (list, tuple)):
            moe_scale_labels = [f"{s}x{s}" for s in moe_scales]
            moe_scale_display = "+".join(moe_scale_labels) + "窗口"
        else:
            moe_scale_display = str(moe_scales)
        print(f"    * MoE尺度 MoEScales (MODEL.MOE_SCALES) = {moe_scales} ({moe_scale_display})")
        print(f"    * 专家数量 NumExperts (MODEL.MOE_NUM_EXPERTS) = {cfg.MODEL.MOE_NUM_EXPERTS} (自动匹配窗口数量)")
        print(f"    * 专家隐藏层维度 ExpertHiddenDim (MODEL.MOE_EXPERT_HIDDEN_DIM) = {cfg.MODEL.MOE_EXPERT_HIDDEN_DIM}")
        print(f"    * 门控网络温度 GatingTemperature (MODEL.MOE_TEMPERATURE) = {cfg.MODEL.MOE_TEMPERATURE}")

        # 固定权重模式
        use_fixed = getattr(cfg.MODEL, 'MOE_USE_FIXED_WEIGHTS', False)
        if use_fixed:
            fixed_weights = getattr(cfg.MODEL, 'MOE_FIXED_WEIGHTS', [0.33, 0.33, 0.34])
            print(f"    * 固定权重模式 FixedWeights (MODEL.MOE_USE_FIXED_WEIGHTS) = ✅ ENABLED, Weights = {fixed_weights}")
        else:
            print(f"    * 固定权重模式 FixedWeights (MODEL.MOE_USE_FIXED_WEIGHTS) = ❌ DISABLED (使用动态门控)")

        # Top-k路由
        use_top_k = getattr(cfg.MODEL, 'MOE_USE_TOP_K_ROUTING', False)
        if use_top_k:
            top_k = getattr(cfg.MODEL, 'MOE_TOP_K', 2)
            top_k_mode = getattr(cfg.MODEL, 'MOE_TOP_K_MODE', 'soft')
            print(f"    * Top-k路由 TopKRouting (MODEL.MOE_USE_TOP_K_ROUTING) = ✅ ENABLED, k={top_k}, mode={top_k_mode}")
        else:
            print(f"    * Top-k路由 TopKRouting (MODEL.MOE_USE_TOP_K_ROUTING) = ❌ DISABLED (Soft Routing)")

        # 预处理机制
        use_gate = getattr(cfg.MODEL, 'USE_GATE_FUSION', False)
        use_attention = getattr(cfg.MODEL, 'USE_ATTENTION_FUSION', False)
        if use_gate:
            gate_dropout = getattr(cfg.MODEL, 'GATE_DROPOUT', 0.1)
            print(f"    * 门控融合预处理 GateFusion (MODEL.USE_GATE_FUSION) = ✅ ENABLED, Dropout={gate_dropout}")
        else:
            print(f"    * 门控融合预处理 GateFusion (MODEL.USE_GATE_FUSION) = ❌ DISABLED")

        if use_attention:
            attn_heads = getattr(cfg.MODEL, 'ATTENTION_NUM_HEADS', 8)
            attn_dropout = getattr(cfg.MODEL, 'ATTENTION_DROPOUT', 0.1)
            print(f"    * 注意力融合预处理 AttentionFusion (MODEL.USE_ATTENTION_FUSION) = ✅ ENABLED, Heads={attn_heads}, Dropout={attn_dropout}")
        else:
            print(f"    * 注意力融合预处理 AttentionFusion (MODEL.USE_ATTENTION_FUSION) = ❌ DISABLED")

    # ========== 新增：文本融合和模态内引导配置 ==========
    text_fusion_status = "✅ ENABLED" if getattr(cfg.MODEL, 'USE_TEXT_FUSION', False) else "❌ DISABLED"
    print(f"  - 文本融合 TextFusion (MODEL.USE_TEXT_FUSION) = {text_fusion_status}")
    if getattr(cfg.MODEL, 'USE_TEXT_FUSION', False):
        fusion_method = getattr(cfg.MODEL, 'TEXT_FUSION_METHOD', 'attention')
        fusion_weight = getattr(cfg.MODEL, 'TEXT_FUSION_WEIGHT', 0.3)
        text_feature_dim = getattr(cfg.MODEL, 'TEXT_FEATURE_DIM', 512)

        # 将方法转换为友好的中文名称
        method_names = {
            'attention': '注意力融合 (Cross-Modal Attention)',
            'concat': '特征拼接融合 (Feature Concatenation)',
            'residual': '残差增强融合 (Residual Enhancement)'
        }
        method_display = method_names.get(fusion_method, f'{fusion_method}融合')
        print(f"    * 融合方法 FusionMethod (MODEL.TEXT_FUSION_METHOD) = {fusion_method} ({method_display})")

        if fusion_method == 'residual':
            print(f"    * 融合权重 FusionWeight (MODEL.TEXT_FUSION_WEIGHT) = {fusion_weight}")
        print(f"    * 文本特征维度 TextFeatureDim (MODEL.TEXT_FEATURE_DIM) = {text_feature_dim}")

    modal_guidance_status = "✅ ENABLED" if getattr(cfg.MODEL, 'USE_MODAL_GUIDANCE', False) else "❌ DISABLED"
    print(f"  - 模态内引导 ModalGuidance (MODEL.USE_MODAL_GUIDANCE) = {modal_guidance_status}")
    if getattr(cfg.MODEL, 'USE_MODAL_GUIDANCE', False):
        guidance_residual = getattr(cfg.MODEL, 'GUIDANCE_RESIDUAL', True)
        guidance_scale = getattr(cfg.MODEL, 'GUIDANCE_SCALE', 0.1)
        residual_status = "✅ ENABLED" if guidance_residual else "❌ DISABLED"
        print(f"    * 残差结构 ResidualStructure (MODEL.GUIDANCE_RESIDUAL) = {residual_status}")
        print(f"    * 引导幅度 GuidanceScale (MODEL.GUIDANCE_SCALE) = {guidance_scale}")

    # ========== 新增：数据集配置 ==========
    print(f"  - 数据集类型 Dataset (DATASETS.NAMES) = {cfg.DATASETS.NAMES}")
    use_text_features = getattr(cfg.DATASETS, 'USE_TEXT_FEATURES', False)
    text_features_status = "✅ ENABLED" if use_text_features else "❌ DISABLED"
    print(f"  - 文本特征加载 UseTextFeatures (DATASETS.USE_TEXT_FEATURES) = {text_features_status}")
    if use_text_features:
        qwen_anno_dir = getattr(cfg.DATASETS, 'QWEN_VL_ANNO_DIR', 'data/datasets/QwenVL_Anno')
        print(f"    * QwenVL标注目录 QwenVLAnnoDir (DATASETS.QWEN_VL_ANNO_DIR) = {qwen_anno_dir}")

    # ========== 测试配置 ==========
    print("\n【测试配置 TestConfig】")
    test_batch_size = getattr(cfg.TEST, 'IMS_PER_BATCH', 64)
    print(f"  - 测试批次大小 TestBatchSize (TEST.IMS_PER_BATCH) = {test_batch_size}")

    re_ranking = getattr(cfg.TEST, 'RE_RANKING', 'no')
    re_ranking_status = "✅ ENABLED" if re_ranking == 'yes' else "❌ DISABLED"
    print(f"  - 重新排序 ReRanking (TEST.RE_RANKING) = {re_ranking_status}")

    test_weight = getattr(cfg.TEST, 'WEIGHT', '')
    if test_weight:
        print(f"  - 测试权重路径 TestWeight (TEST.WEIGHT) = {test_weight}")
    else:
        print(f"  - 测试权重路径 TestWeight (TEST.WEIGHT) = 使用训练后的最新权重")

    neck_feat = getattr(cfg.TEST, 'NECK_FEAT', 'after')
    print(f"  - 特征类型 NeckFeat (TEST.NECK_FEAT) = {neck_feat}")

    feat_norm = getattr(cfg.TEST, 'FEAT_NORM', 'yes')
    feat_norm_status = "✅ ENABLED" if feat_norm == 'yes' else "❌ DISABLED"
    print(f"  - 特征归一化 FeatureNorm (TEST.FEAT_NORM) = {feat_norm_status}")

    # ========== 3. 损失函数 ==========
    print("\n【损失函数 LossConfig】")
    print(f"  - ID损失权重 IDLossWeight (MODEL.ID_LOSS_WEIGHT) = {cfg.MODEL.ID_LOSS_WEIGHT}")
    print(f"  - Triplet损失权重 TripletWeight (MODEL.TRIPLET_LOSS_WEIGHT) = {cfg.MODEL.TRIPLET_LOSS_WEIGHT}")
    print(f"  - Triplet Margin (SOLVER.MARGIN) = {cfg.SOLVER.MARGIN}")
    label_smooth_status = "✅ ENABLED" if cfg.MODEL.IF_LABELSMOOTH == 'on' else "❌ DISABLED"
    print(f"  - 标签平滑 LabelSmooth (MODEL.IF_LABELSMOOTH) = {label_smooth_status}")
    
    use_center = getattr(cfg.MODEL, 'IF_WITH_CENTER', 'no')
    if use_center == 'yes':
        center_weight = getattr(cfg.SOLVER, 'CENTER_LOSS_WEIGHT', 0.0005)
        print(f"  - Center Loss (MODEL.IF_WITH_CENTER) = ✅ ENABLED, Weight={center_weight}")
    else:
        print(f"  - Center Loss (MODEL.IF_WITH_CENTER) = ❌ DISABLED")
    
    if cfg.MODEL.USE_MULTI_SCALE_MOE:
        balance_weight = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT', 0.01)
        diversity_weight = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT', 0.01)
        sparsity_weight = getattr(cfg.SOLVER, 'MOE_SPARSITY_LOSS_WEIGHT', 0.001)
        print(f"  - MoE平衡损失权重 MoEBalanceWeight (SOLVER.MOE_BALANCE_LOSS_WEIGHT) = {balance_weight}")
        print(f"  - MoE多样性损失权重 MoEDiversityWeight (SOLVER.MOE_DIVERSITY_LOSS_WEIGHT) = {diversity_weight}")
        print(f"  - MoE稀疏性损失权重 MoESparsityWeight (SOLVER.MOE_SPARSITY_LOSS_WEIGHT) = {sparsity_weight}")
        
        # 动态权重调度
        use_dynamic = getattr(cfg.SOLVER, 'MOE_USE_DYNAMIC_LOSS_WEIGHT', False)
        if use_dynamic:
            print(f"  - MoE动态权重调度 MoEDynamicSchedule (SOLVER.MOE_USE_DYNAMIC_LOSS_WEIGHT) = ✅ ENABLED")
            print(f"    * 平衡损失 BalanceWeightSchedule = {getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_START', 0.001)} -> {getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_END', 0.1)}")
            print(f"    * 多样性损失 DiversityWeightSchedule = {getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_START', 0.001)} -> {getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_END', 0.1)}")
        else:
            print(f"  - MoE动态权重调度 MoEDynamicSchedule (SOLVER.MOE_USE_DYNAMIC_LOSS_WEIGHT) = ❌ DISABLED")
    
    # ========== 4. 数据增强 ==========
    print("\n【数据增强 DataAugConfig】")
    re_prob = getattr(cfg.INPUT, 'RE_PROB', 0.5)
    if re_prob > 0:
        print(f"  - 随机擦除 RandomErasing (INPUT.RE_PROB) = ✅ ENABLED, Prob={re_prob}")
    else:
        print(f"  - 随机擦除 RandomErasing (INPUT.RE_PROB) = ❌ DISABLED")
    
    flip_prob = getattr(cfg.INPUT, 'PROB', 0.5)
    print(f"  - 水平翻转概率 HorizontalFlipProb (INPUT.PROB) = {flip_prob}")
    
    # ========== 5. 数据加载配置 ==========
    print("\n【数据加载配置 DataLoaderConfig】")
    sampler = getattr(cfg.DATALOADER, 'SAMPLER', 'softmax_triplet')

    # 将采样器转换为友好的中文名称
    sampler_names = {
        'softmax': 'Softmax采样器 (Classification Only)',
        'triplet': 'Triplet采样器 (Triplet Only)',
        'softmax_triplet': 'Softmax+Triplet联合采样器 (Joint Training)'
    }
    sampler_display = sampler_names.get(sampler, f'{sampler}采样器')
    print(f"  - 数据采样器 DataSampler (DATALOADER.SAMPLER) = {sampler} ({sampler_display})")

    print(f"  - 批次大小 BatchSize (SOLVER.IMS_PER_BATCH) = {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  - 每个批次实例数 NumInstances (DATALOADER.NUM_INSTANCE) = {cfg.DATALOADER.NUM_INSTANCE}")
    print(f"  - 数据加载线程数 NumWorkers (DATALOADER.NUM_WORKERS) = {cfg.DATALOADER.NUM_WORKERS}")

    # ========== 6. 优化器配置 ==========
    print("\n【优化器配置 OptimizerConfig】")
    gate_lr_factor = getattr(cfg.SOLVER, 'MOE_GATE_LR_FACTOR', 0.01)
    if cfg.MODEL.USE_MULTI_SCALE_MOE:
        gate_lr = cfg.SOLVER.BASE_LR * gate_lr_factor
        print(f"  - 门控网络学习率倍数 GateLRFactor = {gate_lr_factor}")
        print(f"  - 门控网络实际学习率 GateLRValue = {gate_lr:.8f} (BASE_LR × {gate_lr_factor})")

    print("=" * 80)

