import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable

# 多模态边际损失（multi-modal margin loss），其目的是通过对不同模态（例如图像、文本或其他特征）之间的相似度进行度量来优化模型，确保相似模态的特征向量接近，不同模态之间的特征向量保持一定的距离

class multiModalMarginLossNew(nn.Module):
    def __init__(self, margin=3, dist_type='l2'):
        super(multiModalMarginLossNew, self).__init__()
        # 初始化距离类型和边际
        self.dist_type = dist_type
        self.margin = margin

        # 根据 dist_type 选择距离计算方法
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')  # L2 距离（欧氏距离）
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)  # 余弦相似度
        if dist_type == 'l1':
            self.dist = nn.L1Loss()  # L1 距离（曼哈顿距离）

    def forward(self, feat1, feat2, feat3, label1):
        """
        前向传播：计算多模态特征之间的损失。

        参数：
        - feat1, feat2, feat3: 三个模态的特征向量。
        - label1: 样本的标签，用于计算每个类别的中心。

        返回：
        - dist: 最终的损失值。
        """
        # 获取标签中独特的类别数量
        label_num = len(label1.unique())

        # 将每个模态的特征按标签划分成子集
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        feat3 = feat3.chunk(label_num, 0)

        dist = 0  # 初始化损失

        for i in range(label_num):
            # 计算每个类别的均值中心（centroid）
            center1 = torch.mean(feat1[i], dim=0)  # 第一个模态的均值中心
            center2 = torch.mean(feat2[i], dim=0)  # 第二个模态的均值中心
            center3 = torch.mean(feat3[i], dim=0)  # 第三个模态的均值中心

            # 计算每对模态之间的距离，并与边际值（margin）进行比较
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    # 计算距离并与边际值进行比较，确保最小距离不小于边际值
                    dist = max(abs(self.margin - self.dist(center1, center2)),
                               abs(self.margin - self.dist(center2, center3)),
                               abs(self.margin - self.dist(center1, center3)))
                else:
                    dist += max(abs(self.margin - self.dist(center1, center2)),
                                abs(self.margin - self.dist(center2, center3)),
                                abs(self.margin - self.dist(center1, center3)))

        # 返回总的损失
        return dist


