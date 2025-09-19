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
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}  # 记录最好指标

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
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:   # 还可以加 234 等，可以看到别的值
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))  # 打印日至
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
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                # 维护最佳指标并保存 best.pth（仅非分布式分支）
                if mAP >= best_index['mAP']:
                    best_index['mAP']     = mAP
                    best_index['Rank-1']  = cmc[0]
                    best_index['Rank-5']  = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                torch.cuda.empty_cache()


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
