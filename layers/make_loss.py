# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):  # modified by gu
    # 获取数据加载器采样策略
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048  # 特征维度（根据模型输出的特征大小设定）

    # 创建 CenterLoss 对象，计算样本类中心的损失
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False)  # center loss
    
    # MoE损失函数
    moe_loss_fn = None
    if getattr(cfg.MODEL, 'USE_MULTI_SCALE_MOE', False):
        from .moe_loss import make_moe_loss
        moe_loss_fn = make_moe_loss(cfg)

    # 三元组损失函数
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)

    # 标签平滑
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    # 根据采样策略，选择对应的损失计算方式
    if sampler == 'softmax':  # 采用 softmax 损失
        def loss_func(score, feat, target, target_cam):
            return F.cross_entropy(score, target)  # 计算交叉熵损失

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':  # 采用 softmax 和 triplet 损失
        def loss_func(score, feat, target, target_cam):
            # 如果使用三元组损失
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # 如果启用了标签平滑
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # 对于 score 为列表的情况，分别计算每个元素的损失并加权
                    if isinstance(score, list):
                        # 计算每个 score 的交叉熵损失并求平均
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        # 对第一个 score 特别处理，平均后再加权
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    # 对于 feat 为列表的情况，分别计算每个特征的三元组损失并加权
                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    # 返回加权后的损失，包含 ID 损失和三元组损失
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    # 如果没有启用标签平滑，直接计算交叉熵损失和三元组损失
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    # 返回加权后的损失，包含 ID 损失和三元组损失
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

    # 🔥 修改：将MoE损失函数附加到损失函数上
    # 功能：让损失函数能够访问MoE损失函数
    if moe_loss_fn is not None:
        loss_func.moe_loss_fn = moe_loss_fn
    
    # 返回损失函数和中心损失（CenterLoss）
    return loss_func, center_criterion
