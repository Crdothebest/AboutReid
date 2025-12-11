"""
MambaPro 训练与推理处理器（中文说明）

职责：
- do_train: 完整训练流程（前向、损失、反传、优化、日志、验证、保存最佳/周期权重）
- do_inference: 评估/推理流程（特征抽取、评估指标计算与日志）

说明：
- 支持混合精度训练（torch.cuda.amp）与分布式训练（DDP）
- 评估器根据数据集类型在 R1_mAP 与 R1_mAP_eval 间切换（MSVR310 特殊）
"""
import logging
import os
import time
import math
import torch
import torch.nn as nn
from utils.meter import AverageMeter            # 记录/平滑指标（loss/acc等）
from utils.metrics import R1_mAP_eval, R1_mAP   # 评估指标计算器（mAP & CMC）
from torch.cuda import amp                      # 混合精度工具：autocast + GradScaler
import torch.distributed as dist                # 分布式训练
from layers.supcontrast import SupConLoss       # 监督对比损失（本文件未直接使用）

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD                  # 日志打印间隔（iter）
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD    # 保存权重间隔（epoch）
    eval_period = cfg.SOLVER.EVAL_PERIOD                # 验证间隔（epoch）

    device = "cuda"                                     # 统一使用 GPU
    epochs = cfg.SOLVER.MAX_EPOCHS                      # 最大训练轮数
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("MambaPro.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None                         # 预留：本文件未使用

    if device:
        model.to(local_rank)                            # 将模型放到当前进程对应的GPU
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            # 若多卡 + 开启分布式训练，用 DDP 包裹（每个进程绑定一个 device_id）
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    loss_meter = AverageMeter()                         # 记录平均 loss
    acc_meter = AverageMeter()                          # 记录平均 acc
    # 根据不同数据集选择不同评估器（MSVR310 需要额外视角/场景信息）
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()                           # 混合精度：缩放器
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0, 'best_epoch': 0, 'best_expert_weights': None}  # 记录最好指标和对应epoch及专家权重
    # 🔥 新增：记录每次验证的current和best值（用于趋势分析）
    validation_history = {
        'epochs': [],
        'current_mAP': [],
        'best_mAP': [],
        'current_Rank1': [],
        'best_Rank1': [],
        'current_Rank5': [],
        'best_Rank5': [],
        'current_Rank10': [],
        'best_Rank10': [],
        'expert_weights': []  # 🔥 新增：记录每次验证的专家权重分布
    }

    # =========================
    # 主训练循环
    # =========================
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)                           # epoch 级 LR 调度（注意：用法依赖你的 scheduler 实现）
        model.train()

        # -------- 单个 epoch 内的迭代 --------
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            # 将三模态图像搬到 GPU
            img = {'RGB': img['RGB'].to(device),
                   'NI':  img['NI'].to(device),
                   'TI':  img['TI'].to(device)}
            target = vid.to(device)                     # 行人ID标签
            target_cam = target_cam.to(device)          # 摄像头ID
            target_view = target_view.to(device)        # 视角/场景ID（数据集定义）

            # 前向：混合精度
            with amp.autocast(enabled=True):
                # 模型前向；部分模型会根据 label/cam/view 执行不同分支（如 BNNeck/part head）
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)

                # output 通常是 [logits_0, feat_0, logits_1, feat_1, ...]
                loss = 0
                index = len(output)
                for i in range(0, index, 2):
                    # 自定义的多头/多尺度损失：按对（score/feat）计算并累加
                    loss_tmp = loss_fn(score=output[i], feat=output[i + 1],
                                       target=target, target_cam=target_cam)
                    loss = loss + loss_tmp
                
                # 🔥 新增：MoE损失计算
                # 功能：为MoE模块添加专门的损失函数
                # 包含：平衡损失、稀疏性损失、多样性损失
                if hasattr(model, 'BACKBONE') and hasattr(model.BACKBONE, 'current_expert_weights'):
                    expert_weights = model.BACKBONE.current_expert_weights
                    if expert_weights is not None:
                        # 从损失函数中获取MoE损失函数
                        if hasattr(loss_fn, 'moe_loss_fn') and loss_fn.moe_loss_fn is not None:
                            # 🔧 关键修复：如果权重被detach，说明梯度连接已断开
                            # 这种情况下，我们需要直接从门控网络重新获取权重
                            if not expert_weights.requires_grad:
                                # ⚠️ 警告：权重没有梯度，可能无法更新门控网络
                                if n_iter % 100 == 0:
                                    print(f"⚠️  警告: 专家权重没有梯度！门控网络可能无法更新。")
                                    print(f"   请检查 modeling/make_model.py 中权重保存时是否被detach")
                            
                            # 🔥 新增：动态损失权重调度
                            # 功能：根据训练阶段动态调整MoE损失权重
                            # 原理：早期epoch降低专家约束权重（让模型先学习主任务），后期epoch提高权重（防止专家固化）
                            # 🔥 修复：确保命令行设置的0.0权重有最高优先级
                            # 配置优先级：命令行参数 > 动态调度 > 默认值
                            
                            # 🔥 修复：同时支持MODEL和SOLVER命名空间，优先使用SOLVER
                            # 首先读取命令行设置的静态权重（最高优先级）
                            static_balance_weight = None
                            static_diversity_weight = None
                            # 优先读取SOLVER命名空间，如果不存在则读取MODEL命名空间
                            if hasattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT'):
                                static_balance_weight = cfg.SOLVER.MOE_BALANCE_LOSS_WEIGHT
                            elif hasattr(cfg.MODEL, 'MOE_BALANCE_LOSS_WEIGHT'):
                                static_balance_weight = cfg.MODEL.MOE_BALANCE_LOSS_WEIGHT
                            
                            if hasattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT'):
                                static_diversity_weight = cfg.SOLVER.MOE_DIVERSITY_LOSS_WEIGHT
                            elif hasattr(cfg.MODEL, 'MOE_DIVERSITY_LOSS_WEIGHT'):
                                static_diversity_weight = cfg.MODEL.MOE_DIVERSITY_LOSS_WEIGHT
                            
                            # 🔥 修复：检查是否启用动态调度（需要显式检查，因为可能是字符串）
                            use_dynamic_loss_weight = getattr(cfg.SOLVER, 'MOE_USE_DYNAMIC_LOSS_WEIGHT', False)
                            # 处理YACS可能将"False"解析为字符串的情况
                            if isinstance(use_dynamic_loss_weight, str):
                                use_dynamic_loss_weight = use_dynamic_loss_weight.lower() not in ('false', '0', 'no')
                            else:
                                use_dynamic_loss_weight = bool(use_dynamic_loss_weight)
                            
                            # 🔥 修复：如果命令行设置了0.0，直接使用0.0，跳过动态调度（最高优先级）
                            if static_balance_weight == 0.0 and static_diversity_weight == 0.0:
                                dynamic_balance_weight = 0.0
                                dynamic_diversity_weight = 0.0
                                if n_iter % 500 == 0:  # 每500次迭代打印一次
                                    print(f"🔒 命令行强制禁用所有MoE Loss（权重=0.0），跳过动态调度")
                            elif use_dynamic_loss_weight:
                                # 计算动态权重
                                max_epochs = cfg.SOLVER.MAX_EPOCHS
                                warmup_epochs = getattr(cfg.SOLVER, 'MOE_LOSS_WEIGHT_WARMUP_EPOCHS', 5)
                                schedule_type = getattr(cfg.SOLVER, 'MOE_LOSS_WEIGHT_SCHEDULE_TYPE', 'cosine')
                                
                                # 计算当前epoch的进度（0.0到1.0）
                                if epoch <= warmup_epochs:
                                    # 预热期：使用起始权重
                                    progress = 0.0
                                else:
                                    # 调度期：计算进度
                                    progress = min(1.0, (epoch - warmup_epochs) / (max_epochs - warmup_epochs))
                                
                                # 根据调度类型计算权重插值
                                if schedule_type == 'linear':
                                    # 线性调度：直接线性插值
                                    weight_factor = progress
                                elif schedule_type == 'cosine':
                                    # 余弦调度：使用余弦函数平滑过渡
                                    weight_factor = 0.5 * (1 - math.cos(math.pi * progress))
                                else:
                                    # 默认使用线性调度
                                    weight_factor = progress
                                
                                # 计算动态权重
                                balance_start = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_START', 0.001)
                                balance_end = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_END', 0.1)
                                dynamic_balance_weight = balance_start + (balance_end - balance_start) * weight_factor
                                
                                diversity_start = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_START', 0.001)
                                diversity_end = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_END', 0.1)
                                dynamic_diversity_weight = diversity_start + (diversity_end - diversity_start) * weight_factor
                                
                                # 🔥 修复：如果命令行设置了静态权重，使用静态权重覆盖动态权重
                                if static_balance_weight is not None:
                                    dynamic_balance_weight = static_balance_weight
                                if static_diversity_weight is not None:
                                    dynamic_diversity_weight = static_diversity_weight
                            else:
                                # 未启用动态调度，使用命令行设置的静态权重
                                dynamic_balance_weight = static_balance_weight
                                dynamic_diversity_weight = static_diversity_weight
                            
                            # 🔥 最终验证：确保命令行设置的0.0权重不被覆盖（最高优先级）
                            # 配置优先级：命令行参数 > 动态调度 > 默认值
                            # 同时检查SOLVER和MODEL命名空间
                            if static_diversity_weight == 0.0:
                                dynamic_diversity_weight = 0.0
                            if static_balance_weight == 0.0:
                                dynamic_balance_weight = 0.0
                            
                            # 调用MoE损失函数，传入动态权重
                            moe_loss, moe_loss_dict = loss_fn.moe_loss_fn(
                                expert_weights,
                                balance_weight=dynamic_balance_weight,
                                diversity_weight=dynamic_diversity_weight
                            )
                            loss = loss + moe_loss
                            
                            # 记录MoE损失信息（可选）
                            if n_iter % 100 == 0:  # 每100个iteration打印一次
                                # 🔧 添加调试信息：显示权重分布和梯度状态
                                with torch.no_grad():
                                    avg_weights = expert_weights.mean(dim=0).cpu().numpy()
                                    weight_std = expert_weights.std(dim=0).mean().item()
                                    has_grad = expert_weights.requires_grad
                                
                                # 显示动态权重信息（如果启用）
                                weight_info = ""
                                if getattr(cfg.SOLVER, 'MOE_USE_DYNAMIC_LOSS_WEIGHT', False):
                                    weight_info = f" (动态权重: 平衡={moe_loss_dict.get('moe_balance_weight', 'N/A'):.4f}, 多样性={moe_loss_dict.get('moe_diversity_weight', 'N/A'):.4f})"
                                
                                # 🔥 新增：检查多样性损失是否激活
                                diversity_loss_value = moe_loss_dict['moe_diversity_loss']
                                diversity_status = "✅ 已激活" if diversity_loss_value > 1e-6 else "⚠️ 未激活(0.0)"
                                
                                print(f"🔥 MoE损失: 平衡={moe_loss_dict['moe_balance_loss']:.4f}, "
                                      f"稀疏性={moe_loss_dict['moe_sparsity_loss']:.4f}, "
                                      f"多样性={moe_loss_dict['moe_diversity_loss']:.4f} {diversity_status}{weight_info}")
                                print(f"   📊 专家权重分布: [{avg_weights[0]:.4f}, {avg_weights[1]:.4f}, {avg_weights[2]:.4f}], "
                                      f"权重变化标准差: {weight_std:.6f}, 有梯度: {has_grad}")
                                
                                # 🔥 新增：详细MoE Loss日志，检查多样性损失是否成功激活
                                if n_iter % 500 == 0:  # 每500个iteration打印一次详细日志
                                    print(f"📋 MoE Loss 详细日志 (Epoch {epoch}, Iter {n_iter}):")
                                    print(f"   - 平衡损失 (L_Bal): {moe_loss_dict['moe_balance_loss']:.6f}")
                                    print(f"   - 稀疏性损失 (L_Spar): {moe_loss_dict['moe_sparsity_loss']:.6f}")
                                    print(f"   - 多样性损失 (L_Div): {moe_loss_dict['moe_diversity_loss']:.6f} {'✅ 已激活' if diversity_loss_value > 1e-6 else '⚠️ 未激活(0.0)'}")
                                    print(f"   - 总损失 (L_Total): {moe_loss_dict['moe_total_loss']:.6f}")
                                    if 'moe_balance_weight' in moe_loss_dict:
                                        print(f"   - 损失权重: 平衡={moe_loss_dict['moe_balance_weight']:.4f}, "
                                              f"稀疏性={moe_loss_dict.get('moe_sparsity_weight', 'N/A')}, "
                                              f"多样性={moe_loss_dict['moe_diversity_weight']:.4f}")

            # 反传 + 参数更新（混合精度）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 若包含 center loss，需要对其梯度按权重缩放，并单独更新其中心参数
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # 训练准确率（从分类 logits 中取 argmax）
            if isinstance(output, list):
                # output[0] 可能是 (logits, ...) 的结构，这里取 output[0][0] 做分类
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            # 更新度量器（按样本数记权）
            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()                    # 避免异步导致计时不准
            if (n_iter + 1) % log_period == 0:
                # 注意：scheduler._get_lr(epoch) 非标准API，取决于自定义调度器
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        # -------- epoch 训练结束：统计耗时/速度 --------
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass  # 分布式下通常由各 rank 分别统计或仅 rank0 打印
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # -------- 保存 checkpoint --------
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:                # 仅主进程保存
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # -------- 周期性验证 --------
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = {'RGB': img['RGB'].to(device),
                                   'NI':  img['NI'].to(device),
                                   'TI':  img['TI'].to(device)}
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            # 特征抽取（评估时不传 label）
                            feat = model(img, cam_label=camids, view_label=target_view)
                            if cfg.DATASETS.NAMES == "MSVR310":
                                evaluator.update((feat, vid, camid, target_view, _))  # 注意这里把 view 参与评估
                            else:
                                evaluator.update((feat, vid, camid))
                                ## 论文里的评价指标 cmc map
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()  # 在这里计算
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("Current mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:   # 还可以加 234 等，可以看到别的值
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))  # 打印日至
                    
                    # 🔥 新增：获取当前验证的专家权重分布和MoE损失
                    current_expert_weights = None
                    current_moe_loss_dict = None
                    if hasattr(model, 'BACKBONE') and hasattr(model.BACKBONE, 'current_expert_weights'):
                        expert_weights = model.BACKBONE.current_expert_weights
                        if expert_weights is not None:
                            # 计算batch平均权重分布 [num_experts]
                            with torch.no_grad():
                                avg_weights = expert_weights.mean(dim=0).cpu().numpy()
                                current_expert_weights = avg_weights.tolist()
                                
                                # 🔥 新增：计算当前验证时的MoE损失
                                if hasattr(loss_fn, 'moe_loss_fn') and loss_fn.moe_loss_fn is not None:
                                    # 使用与训练时相同的权重逻辑
                                    static_balance_weight = None
                                    static_diversity_weight = None
                                    if hasattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT'):
                                        static_balance_weight = cfg.SOLVER.MOE_BALANCE_LOSS_WEIGHT
                                    elif hasattr(cfg.MODEL, 'MOE_BALANCE_LOSS_WEIGHT'):
                                        static_balance_weight = cfg.MODEL.MOE_BALANCE_LOSS_WEIGHT
                                    
                                    if hasattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT'):
                                        static_diversity_weight = cfg.SOLVER.MOE_DIVERSITY_LOSS_WEIGHT
                                    elif hasattr(cfg.MODEL, 'MOE_DIVERSITY_LOSS_WEIGHT'):
                                        static_diversity_weight = cfg.MODEL.MOE_DIVERSITY_LOSS_WEIGHT
                                    
                                    # 检查是否启用动态调度
                                    use_dynamic_loss_weight = getattr(cfg.SOLVER, 'MOE_USE_DYNAMIC_LOSS_WEIGHT', False)
                                    if isinstance(use_dynamic_loss_weight, str):
                                        use_dynamic_loss_weight = use_dynamic_loss_weight.lower() not in ('false', '0', 'no')
                                    else:
                                        use_dynamic_loss_weight = bool(use_dynamic_loss_weight)
                                    
                                    if static_balance_weight == 0.0 and static_diversity_weight == 0.0:
                                        dynamic_balance_weight = 0.0
                                        dynamic_diversity_weight = 0.0
                                    elif use_dynamic_loss_weight:
                                        max_epochs = cfg.SOLVER.MAX_EPOCHS
                                        warmup_epochs = getattr(cfg.SOLVER, 'MOE_LOSS_WEIGHT_WARMUP_EPOCHS', 5)
                                        schedule_type = getattr(cfg.SOLVER, 'MOE_LOSS_WEIGHT_SCHEDULE_TYPE', 'cosine')
                                        
                                        if epoch <= warmup_epochs:
                                            progress = 0.0
                                        else:
                                            progress = min(1.0, (epoch - warmup_epochs) / (max_epochs - warmup_epochs))
                                        
                                        if schedule_type == 'linear':
                                            weight_factor = progress
                                        elif schedule_type == 'cosine':
                                            weight_factor = 0.5 * (1 - math.cos(math.pi * progress))
                                        else:
                                            weight_factor = progress
                                        
                                        balance_start = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_START', 0.001)
                                        balance_end = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_END', 0.1)
                                        dynamic_balance_weight = balance_start + (balance_end - balance_start) * weight_factor
                                        
                                        diversity_start = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_START', 0.001)
                                        diversity_end = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_END', 0.1)
                                        dynamic_diversity_weight = diversity_start + (diversity_end - diversity_start) * weight_factor
                                        
                                        if static_balance_weight is not None:
                                            dynamic_balance_weight = static_balance_weight
                                        if static_diversity_weight is not None:
                                            dynamic_diversity_weight = static_diversity_weight
                                    else:
                                        dynamic_balance_weight = static_balance_weight
                                        dynamic_diversity_weight = static_diversity_weight
                                    
                                    if static_diversity_weight == 0.0:
                                        dynamic_diversity_weight = 0.0
                                    if static_balance_weight == 0.0:
                                        dynamic_balance_weight = 0.0
                                    
                                    # 计算MoE损失
                                    _, current_moe_loss_dict = loss_fn.moe_loss_fn(
                                        expert_weights,
                                        balance_weight=dynamic_balance_weight,
                                        diversity_weight=dynamic_diversity_weight
                                    )
                    
                    # 🔥 新增：更新并输出最佳指标（分布式训练）
                    if mAP >= best_index['mAP']:
                        best_index['mAP']     = mAP
                        best_index['Rank-1']  = cmc[0]
                        best_index['Rank-5']  = cmc[4]
                        best_index['Rank-10'] = cmc[9]
                        best_index['best_epoch'] = epoch
                        best_index['best_expert_weights'] = current_expert_weights.copy() if current_expert_weights else None
                        logger.info("🎯 New Best! Saving model...")
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                    
                    # 🔥 新增：记录当前验证的current和best值
                    validation_history['epochs'].append(epoch)
                    validation_history['current_mAP'].append(mAP * 100)  # 转换为百分比
                    validation_history['best_mAP'].append(best_index['mAP'] * 100)
                    validation_history['current_Rank1'].append(cmc[0] * 100)
                    validation_history['best_Rank1'].append(best_index['Rank-1'] * 100)
                    validation_history['current_Rank5'].append(cmc[4] * 100)
                    validation_history['best_Rank5'].append(best_index['Rank-5'] * 100)
                    validation_history['current_Rank10'].append(cmc[9] * 100)
                    validation_history['best_Rank10'].append(best_index['Rank-10'] * 100)
                    validation_history['expert_weights'].append(current_expert_weights if current_expert_weights else [0.0, 0.0, 0.0])  # 默认值
                    
                    current_summary = (
                        f"[Epoch {epoch}] CURRENT -> "
                        f"mAP:{mAP * 100:.1f} | "
                        f"Rank-1:{cmc[0] * 100:.1f} | "
                        f"Rank-5:{cmc[4] * 100:.1f} | "
                        f"Rank-10:{cmc[9] * 100:.1f}"
                    )
                    if current_expert_weights is not None:
                        current_summary += " | Experts:[{:.4f}, {:.4f}, {:.4f}]".format(
                            current_expert_weights[0],
                            current_expert_weights[1],
                            current_expert_weights[2],
                        )
                    logger.info(current_summary)
                    
                    best_summary = (
                        f"[Epoch {best_index['best_epoch']}] BEST -> "
                        f"mAP:{best_index['mAP'] * 100:.1f} | "
                        f"Rank-1:{best_index['Rank-1'] * 100:.1f} | "
                        f"Rank-5:{best_index['Rank-5'] * 100:.1f} | "
                        f"Rank-10:{best_index['Rank-10'] * 100:.1f}"
                    )
                    if best_index['best_expert_weights'] is not None:
                        weights = best_index['best_expert_weights']
                        best_summary += " | Experts:[{:.4f}, {:.4f}, {:.4f}]".format(
                            weights[0], weights[1], weights[2]
                        )
                    logger.info(best_summary)
                    
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = {'RGB': img['RGB'].to(device),
                               'NI':  img['NI'].to(device),
                               'TI':  img['TI'].to(device)}
                        camids = camids.to(device)
                        scenceids = target_view                    # 保留原始 scene id（变量名有拼写：scenceids）
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator.update((feat, vid, camid, scenceids, _))
                        else:
                            evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("Current mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                # 🔥 新增：获取当前验证的专家权重分布和MoE损失
                current_expert_weights = None
                current_moe_loss_dict = None
                if hasattr(model, 'BACKBONE') and hasattr(model.BACKBONE, 'current_expert_weights'):
                    expert_weights = model.BACKBONE.current_expert_weights
                    if expert_weights is not None:
                        # 计算batch平均权重分布 [num_experts]
                        with torch.no_grad():
                            avg_weights = expert_weights.mean(dim=0).cpu().numpy()
                            current_expert_weights = avg_weights.tolist()
                            
                            # 🔥 新增：计算当前验证时的MoE损失
                            if hasattr(loss_fn, 'moe_loss_fn') and loss_fn.moe_loss_fn is not None:
                                # 使用与训练时相同的权重逻辑
                                static_balance_weight = None
                                static_diversity_weight = None
                                if hasattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT'):
                                    static_balance_weight = cfg.SOLVER.MOE_BALANCE_LOSS_WEIGHT
                                elif hasattr(cfg.MODEL, 'MOE_BALANCE_LOSS_WEIGHT'):
                                    static_balance_weight = cfg.MODEL.MOE_BALANCE_LOSS_WEIGHT
                                
                                if hasattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT'):
                                    static_diversity_weight = cfg.SOLVER.MOE_DIVERSITY_LOSS_WEIGHT
                                elif hasattr(cfg.MODEL, 'MOE_DIVERSITY_LOSS_WEIGHT'):
                                    static_diversity_weight = cfg.MODEL.MOE_DIVERSITY_LOSS_WEIGHT
                                
                                # 检查是否启用动态调度
                                use_dynamic_loss_weight = getattr(cfg.SOLVER, 'MOE_USE_DYNAMIC_LOSS_WEIGHT', False)
                                if isinstance(use_dynamic_loss_weight, str):
                                    use_dynamic_loss_weight = use_dynamic_loss_weight.lower() not in ('false', '0', 'no')
                                else:
                                    use_dynamic_loss_weight = bool(use_dynamic_loss_weight)
                                
                                if static_balance_weight == 0.0 and static_diversity_weight == 0.0:
                                    dynamic_balance_weight = 0.0
                                    dynamic_diversity_weight = 0.0
                                elif use_dynamic_loss_weight:
                                    max_epochs = cfg.SOLVER.MAX_EPOCHS
                                    warmup_epochs = getattr(cfg.SOLVER, 'MOE_LOSS_WEIGHT_WARMUP_EPOCHS', 5)
                                    schedule_type = getattr(cfg.SOLVER, 'MOE_LOSS_WEIGHT_SCHEDULE_TYPE', 'cosine')
                                    
                                    if epoch <= warmup_epochs:
                                        progress = 0.0
                                    else:
                                        progress = min(1.0, (epoch - warmup_epochs) / (max_epochs - warmup_epochs))
                                    
                                    if schedule_type == 'linear':
                                        weight_factor = progress
                                    elif schedule_type == 'cosine':
                                        weight_factor = 0.5 * (1 - math.cos(math.pi * progress))
                                    else:
                                        weight_factor = progress
                                    
                                    balance_start = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_START', 0.001)
                                    balance_end = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT_END', 0.1)
                                    dynamic_balance_weight = balance_start + (balance_end - balance_start) * weight_factor
                                    
                                    diversity_start = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_START', 0.001)
                                    diversity_end = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT_END', 0.1)
                                    dynamic_diversity_weight = diversity_start + (diversity_end - diversity_start) * weight_factor
                                    
                                    if static_balance_weight is not None:
                                        dynamic_balance_weight = static_balance_weight
                                    if static_diversity_weight is not None:
                                        dynamic_diversity_weight = static_diversity_weight
                                else:
                                    dynamic_balance_weight = static_balance_weight
                                    dynamic_diversity_weight = static_diversity_weight
                                
                                if static_diversity_weight == 0.0:
                                    dynamic_diversity_weight = 0.0
                                if static_balance_weight == 0.0:
                                    dynamic_balance_weight = 0.0
                                
                                # 计算MoE损失
                                _, current_moe_loss_dict = loss_fn.moe_loss_fn(
                                    expert_weights,
                                    balance_weight=dynamic_balance_weight,
                                    diversity_weight=dynamic_diversity_weight
                                )
                
                # 🔥 维护最佳指标并保存 best.pth（仅非分布式分支）
                if mAP >= best_index['mAP']:
                    best_index['mAP']     = mAP
                    best_index['Rank-1']  = cmc[0]
                    best_index['Rank-5']  = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    best_index['best_epoch'] = epoch
                    best_index['best_expert_weights'] = current_expert_weights.copy() if current_expert_weights else None
                    logger.info("🎯 New Best! Saving model...")
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                
                # 🔥 新增：记录当前验证的current和best值
                validation_history['epochs'].append(epoch)
                validation_history['current_mAP'].append(mAP * 100)  # 转换为百分比
                validation_history['best_mAP'].append(best_index['mAP'] * 100)
                validation_history['current_Rank1'].append(cmc[0] * 100)
                validation_history['best_Rank1'].append(best_index['Rank-1'] * 100)
                validation_history['current_Rank5'].append(cmc[4] * 100)
                validation_history['best_Rank5'].append(best_index['Rank-5'] * 100)
                validation_history['current_Rank10'].append(cmc[9] * 100)
                validation_history['best_Rank10'].append(best_index['Rank-10'] * 100)
                validation_history['expert_weights'].append(current_expert_weights if current_expert_weights else [0.0, 0.0, 0.0])  # 默认值
                
                # 输出当前/最佳结果摘要
                current_summary = (
                    f"[Epoch {epoch}] CURRENT -> "
                    f"mAP:{mAP * 100:.1f} | "
                    f"Rank-1:{cmc[0] * 100:.1f} | "
                    f"Rank-5:{cmc[4] * 100:.1f} | "
                    f"Rank-10:{cmc[9] * 100:.1f}"
                )
                if current_expert_weights is not None:
                    current_summary += " | Experts:[{:.4f}, {:.4f}, {:.4f}]".format(
                        current_expert_weights[0],
                        current_expert_weights[1],
                        current_expert_weights[2],
                    )
                logger.info(current_summary)

                best_summary = (
                    f"[Epoch {best_index['best_epoch']}] BEST -> "
                    f"mAP:{best_index['mAP'] * 100:.1f} | "
                    f"Rank-1:{best_index['Rank-1'] * 100:.1f} | "
                    f"Rank-5:{best_index['Rank-5'] * 100:.1f} | "
                    f"Rank-10:{best_index['Rank-10'] * 100:.1f}"
                )
                if best_index['best_expert_weights'] is not None:
                    weights = best_index['best_expert_weights']
                    best_summary += " | Experts:[{:.4f}, {:.4f}, {:.4f}]".format(
                        weights[0], weights[1], weights[2]
                    )
                logger.info(best_summary)

                torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("✅ Training Finished")
    logger.info("   Total Epochs: {}".format(epochs))
    logger.info("   Best Epoch: {}".format(best_index['best_epoch']))
    logger.info("   Best mAP: {:.1%}".format(best_index['mAP']))
    logger.info("   Best Rank-1: {:.1%}".format(best_index['Rank-1']))
    logger.info("   Best Rank-5: {:.1%}".format(best_index['Rank-5']))
    logger.info("   Best Rank-10: {:.1%}".format(best_index['Rank-10']))
    if best_index['best_expert_weights'] is not None:
        weights = best_index['best_expert_weights']
        logger.info("   🎯 Best Expert Weights: [{:.4f}, {:.4f}, {:.4f}]".format(
            weights[0], weights[1], weights[2]))
    logger.info("=" * 60)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("MambaPro.test")
    logger.info("Enter inferencing")

    # 与训练相同：根据数据集选择评估器
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            # 推理阶段使用 DataParallel（无需DDP初始化/进程组）
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []                                  # 可选：收集图像路径，供外部使用
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI':  img['NI'].to(device),
                   'TI':  img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view                     # 保留原始 scene id
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]                               # 返回 Rank-1 与 Rank-5
