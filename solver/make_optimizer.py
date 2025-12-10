import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    moe_param_count = 0  # 🔧 统计MoE相关参数数量
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if not cfg.MODEL.FROZEN:
            if "base" in key:
                if "adapter" not in key:
                    lr = 0.000005
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        # 🔧 统计MoE相关参数
        if "moe" in key.lower() or "gating" in key.lower() or "expert" in key.lower():
            moe_param_count += 1
            if moe_param_count <= 5:  # 只打印前5个参数作为示例
                print(f"✅ MoE参数已添加到优化器: {key}, shape={value.shape}, requires_grad={value.requires_grad}")

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    # 🔧 打印MoE参数统计信息
    if moe_param_count > 0:
        print(f"✅ 总共找到 {moe_param_count} 个MoE相关参数，已全部添加到优化器")
    else:
        print(f"⚠️  警告: 未找到MoE相关参数！请检查模型是否正确初始化MoE模块")
    
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
