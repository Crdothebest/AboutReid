import torch


def make_optimizer(cfg, model, center_criterion):
    """
    创建优化器，为不同参数组设置不同的学习率
    
    ========================================================================
    【功能模块：门控网络独立学习率设置】
    ========================================================================
    
    【问题背景】
    -----------
    在MoE（Mixture of Experts）架构中，门控网络（Gating Network）负责动态分配
    专家权重，其更新速度直接影响训练稳定性和模式坍塌风险。
    
    原有问题：
    1. 门控网络使用与普通参数相同的学习率（BASE_LR = 0.0005）
    2. 门控网络更新过快，容易形成自强化循环，导致模式坍塌
    3. 某个专家（如4×4细节专家）快速垄断99.9%的路由权重
    4. 即使设置了平衡损失，也无法阻止模式坍塌
    
    【解决方案】
    -----------
    为门控网络参数设置独立且更低的学习率，实现"慢速决策、快速执行"的策略：
    - 门控网络（决策层）：使用极低学习率（BASE_LR × 0.01 = 0.000005）
    - 专家网络（执行层）：使用正常学习率（BASE_LR = 0.0005）
    
    【实现原理】
    -----------
    1. 参数识别：通过参数名中的"gating"关键字识别门控网络参数
    2. 学习率计算：采用双重乘法逻辑
       - 第一步：应用MOE_GATE_LR_FACTOR（默认0.01）压低学习率
       - 第二步：如果是偏置参数，再应用BIAS_LR_FACTOR（2.0）略微提升
    3. 优先级设置：门控网络逻辑放在最前面，避免被其他逻辑覆盖
    
    【学习率对比】
    -----------
    修改前：
    - 门控网络权重：BASE_LR = 0.0005
    - 门控网络偏置：BASE_LR × 2.0 = 0.001
    
    修改后（BASE_LR = 0.0005, MOE_GATE_LR_FACTOR = 0.01）：
    - 门控网络权重：BASE_LR × 0.01 = 0.000005（降低100倍）
    - 门控网络偏置：BASE_LR × 0.01 × 2.0 = 0.00001（降低100倍）
    
    【预期效果】
    -----------
    1. 门控网络更新更慢，减少模式坍塌风险
    2. 给专家网络更多时间学习，形成更好的专业化分工
    3. 训练更稳定，收敛更平滑
    4. 提高最终mAP性能
    
    【配置参数】
    -----------
    - SOLVER.MOE_GATE_LR_FACTOR: 门控网络学习率倍数（默认0.01）
      * 可通过命令行 --opts SOLVER.MOE_GATE_LR_FACTOR 0.01 动态调整
      * 建议范围：0.001 ~ 0.1（过小会导致门控网络几乎不更新）
    
    【代码位置】
    -----------
    - 配置文件：config/defaults.py (第202行)
    - 实现代码：solver/make_optimizer.py (第14-28行)
    - 使用示例：configs/RGBNT201/yzy_best_Mambapro_moe.yml (第101行)
    
    ========================================================================
    """
    params = []
    moe_param_count = 0  # 🔧 统计MoE相关参数数量
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        
        # ====================================================================
        # 【模块一：门控网络独立学习率设置】
        # ====================================================================
        # 
        # 【功能】为门控网络参数设置独立且更低的学习率
        # 【原因】门控网络负责资源分配决策，需要谨慎、缓慢地调整策略
        #         快速更新会导致模式坍塌（某个专家垄断路由权重）
        # 
        # 【做了什么】
        # 1. 识别门控网络参数（参数名包含"gating"）
        # 2. 应用MOE_GATE_LR_FACTOR（默认0.01）大幅降低学习率
        # 3. 对偏置参数额外应用BIAS_LR_FACTOR（2.0），但仍保持低学习率
        # 4. 设置最高优先级，避免被其他学习率逻辑覆盖
        # 
        # 【学习率计算】
        # - 门控网络权重：BASE_LR × MOE_GATE_LR_FACTOR
        # - 门控网络偏置：BASE_LR × MOE_GATE_LR_FACTOR × BIAS_LR_FACTOR
        # 
        # 【示例】BASE_LR=0.0005, MOE_GATE_LR_FACTOR=0.01, BIAS_LR_FACTOR=2.0
        # - 门控网络权重：0.0005 × 0.01 = 0.000005（降低100倍）
        # - 门控网络偏置：0.0005 × 0.01 × 2.0 = 0.00001（降低100倍）
        # ====================================================================
        if "gating" in key.lower():
            # 获取门控网络学习率倍数（默认0.01，可通过配置文件或命令行设置）
            gate_lr_factor = getattr(cfg.SOLVER, 'MOE_GATE_LR_FACTOR', 0.01)
            
            # 第一步：应用Gating LR Factor（压低学习率）
            # 目的：让门控网络更新更慢，减少模式坍塌风险
            lr = cfg.SOLVER.BASE_LR * gate_lr_factor
            
            # 第二步：如果是偏置参数，再应用BIAS LR Factor（略微提升）
            # 目的：保持偏置学习率略高于权重，符合科研惯例
            # 注意：即使乘以2.0，门控网络偏置的学习率仍然很低（0.00001）
            if "bias" in key:
                lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        else:
            # ====================================================================
            # 【模块二：非门控网络参数学习率设置（原有逻辑）】
            # ====================================================================
            # 
            # 【功能】为其他参数（专家网络、普通参数等）设置学习率
            # 【原因】保持向后兼容，不影响现有训练流程
            # 
            # 【做了什么】
            # 1. 偏置参数：使用BASE_LR × BIAS_LR_FACTOR（通常是2倍）
            # 2. Backbone参数：如果未冻结，使用固定低学习率0.000005（保护预训练权重）
            # 3. 分类器参数：如果启用LARGE_FC_LR，使用2倍学习率
            # 
            # 【注意】门控网络参数不会进入此分支，因为已在上面单独处理
            # ====================================================================
            
            # 偏置参数：使用2倍基础学习率（科研惯例）
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            
            # Backbone参数：如果未冻结，使用极低学习率保护预训练权重
            # 原因：CLIP等预训练模型权重需要谨慎微调，避免破坏已学特征
            if not cfg.MODEL.FROZEN:
                if "base" in key:
                    if "adapter" not in key:
                        lr = 0.000005  # 固定值：BASE_LR的1%（与门控网络相同）
            
            # 分类器参数：如果启用，使用2倍学习率
            if cfg.SOLVER.LARGE_FC_LR:
                if "classifier" in key or "arcface" in key:
                    lr = cfg.SOLVER.BASE_LR * 2

        if "moe" in key.lower() or "gating" in key.lower() or "expert" in key.lower():
            moe_param_count += 1

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
