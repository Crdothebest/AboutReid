"""
Center Loss 实现（中文说明）

职责：
- 维护每个类别的可学习中心向量，并最小化样本到其类别中心的距离
- 提升类内紧致性，与分类/度量损失结合使用效果更佳

要点：
- forward 接收特征 `x` 与标签 `labels`，计算到对应中心的平方欧氏距离均值
- `self.centers` 为可训练参数，尺寸 [num_classes, feat_dim]
"""
from __future__ import absolute_import

import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    # 该类继承 nn.Module，表示这是一个可训练的损失模块。
    #
    # Center Loss 的核心思想是：让同类特征更接近它们的类别中心，从而提升特征判别力。

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):  # 定义构造函数，设置类别数、特征维度、是否使用GPU
        super(CenterLoss, self).__init__()  # 调用父类nn.Module的构造函数，保证参数和子模块能被正确注册
        self.num_classes = num_classes  # 保存类别总数，每个类别对应一个中心向量
        self.feat_dim = feat_dim  # 保存特征维度，决定中心向量的维度
        self.use_gpu = use_gpu  # 保存是否使用GPU的标志

        if self.use_gpu:  # 如果启用GPU
            self.centers = nn.Parameter(  # 定义一个可学习的参数，表示所有类别的中心向量
                torch.randn(self.num_classes, self.feat_dim).cuda()  # 按标准正态分布随机初始化中心向量，并拷贝到GPU上
            )
        else:  # 如果只在CPU上训练
            self.centers = nn.Parameter(  # 定义同样的可学习参数
                torch.randn(self.num_classes, self.feat_dim)  # 在CPU上按标准正态分布随机初始化中心向量
            )

    def forward(self, x, labels):  # 定义前向传播，输入特征向量 x 和对应的类别标签 labels
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).              # x 是一个(batch_size, 特征维度) 的特征矩阵
            labels: ground truth labels with shape (batch_size).              # labels 是(batch_size,) 的真实类别标签
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        # 检查输入的特征数量和标签数量是否一致，防止数据对不上

        batch_size = x.size(0)  # 获取当前批次的样本数量

        # 计算特征与中心的平方欧氏距离矩阵
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # 公式：(x - c)^2 = x^2 + c^2 - 2*x*c，前两项分别计算 x^2 和 c^2，并扩展到同样的维度

        distmat.addmm_(1, -2, x, self.centers.t())  # 加上 -2*x*c 部分，完成欧氏距离平方的计算

        classes = torch.arange(self.num_classes).long()  # 生成类别索引 [0, 1, 2, ..., num_classes-1]
        if self.use_gpu: classes = classes.cuda()  # 如果在GPU上训练，则将类别索引转到GPU

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)  # 将 labels 扩展成(batch_size, num_classes) 方便做掩码
        mask = labels.eq(classes.expand(batch_size, self.num_classes))  # mask 里，每个样本在其真实类别位置是 True，其它位置是 False

        dist = distmat * mask.float()  # 只保留每个样本与真实类别中心的距离，其它类别的距离置0
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size  # 对距离做截断避免数值不稳定，求和后取平均得到最终损失

        return loss  # 返回 Center Loss


if __name__ == '__main__':
    use_gpu = False
    center_loss = CenterLoss(use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)
    print(loss)
