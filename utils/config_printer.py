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
    print(f"  - Batch Size BatchSize = {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  - 基础学习率 BaseLR = {cfg.SOLVER.BASE_LR}")
    print(f"  - 优化器 Optimizer = {cfg.SOLVER.OPTIMIZER_NAME}")
    print(f"  - 学习率调度 LR_Schedule = Steps{cfg.SOLVER.STEPS}, Gamma={cfg.SOLVER.GAMMA}")
    print(f"  - Warmup WarmupIters = {cfg.SOLVER.WARMUP_ITERS}, WarmupFactor = {cfg.SOLVER.WARMUP_FACTOR}")
    
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
    
    # ========== 5. 优化器配置 ==========
    print("\n【优化器配置 OptimizerConfig】")
    gate_lr_factor = getattr(cfg.SOLVER, 'MOE_GATE_LR_FACTOR', 0.01)
    if cfg.MODEL.USE_MULTI_SCALE_MOE:
        gate_lr = cfg.SOLVER.BASE_LR * gate_lr_factor
        print(f"  - 门控网络学习率倍数 GateLRFactor = {gate_lr_factor}")
        print(f"  - 门控网络实际学习率 GateLRValue = {gate_lr:.8f} (BASE_LR × {gate_lr_factor})")
    
    print("=" * 80)

