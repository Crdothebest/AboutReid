import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable

class hetero_loss(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(hetero_loss, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()


def forward(self, feat1, feat2, label1):
    feat_size = feat1.size()[1]  # 获取特征维度
    feat_num = feat1.size()[0]  # 获取特征数目（样本数量）
    label_num = len(label1.unique())  # 获取标签的种类数（即类别数）

    feat1 = feat1.chunk(label_num, 0)  # 按照标签数将特征分成多个小块
    feat2 = feat2.chunk(label_num, 0)  # 同样地分割第二组特征

    # 遍历每个类别，计算每个类别内特征中心之间的距离
    for i in range(label_num):
        center1 = torch.mean(feat1[i], dim=0)  # 计算第一组特征的类别中心
        center2 = torch.mean(feat2[i], dim=0)  # 计算第二组特征的类别中心

        # 如果使用的是L2距离或L1距离
        if self.dist_type == 'l2' or self.dist_type == 'l1':
            if i == 0:
                dist = max(0, abs(self.dist(center1, center2)))  # 计算距离并进行裁剪
            else:
                dist += max(0, abs(self.dist(center1, center2)))  # 累加各类别的距离

        # 如果使用的是余弦相似度
        elif self.dist_type == 'cos':
            if i == 0:
                dist = max(0, 1 - self.dist(center1, center2))  # 余弦相似度，1减去相似度
            else:
                dist += max(0, 1 - self.dist(center1, center2))  # 累加

    return dist  # 返回最终的损失值

