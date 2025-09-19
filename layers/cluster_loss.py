from __future__ import absolute_import        # 确保 Python2/3 兼容的绝对导入行为
import torch                                  # 导入 PyTorch 核心库
from torch import nn                          # 导入神经网络模块
import torch.nn.functional as F               # 导入 PyTorch 提供的常用函数接口



class ClusterLoss(nn.Module):                           # 定义全局聚类损失类
    # 功能：
    # 1. 负责存储训练参数
    # 2. 定义 margin 作为正负类距离的分隔阈值
    def __init__(self, margin=10, use_gpu=True, ordered=True, ids_per_batch=16, imgs_per_id=4):
        super(ClusterLoss, self).__init__()              # 调用父类初始化
        self.use_gpu = use_gpu                           # 是否使用 GPU
        self.margin = margin                             # 分类间最小间隔
        self.ordered = ordered                           # 数据是否按 [id1, id1, id2, id2...] 排序
        self.ids_per_batch = ids_per_batch               # 每个 batch 的 ID 数
        self.imgs_per_id = imgs_per_id                   # 每个 ID 的图像数

    def _euclidean_dist(self, x, y):  # 计算 x 与 y 的欧氏距离
        m, n = x.size(0), y.size(0)  # m 行 n 列
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)  # x 的平方和，扩展维度
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()  # y 的平方和，转置后扩展维度
        dist = xx + yy  # 先加 x^2 + y^2
        dist.addmm_(1, -2, x, y.t())  # 减去 2xy 得到 (x-y)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 开根号并截断避免数值不稳定
        return dist  # 返回 [m, n] 的距离矩阵

    def _cluster_loss(self, features, targets, ordered=True, ids_per_batch=16, imgs_per_id=4):
        # features: (batch_size, feature_dim)
        # targets: (batch_size)

        if self.use_gpu:  # GPU 情况
            if ordered and targets.size(0) == ids_per_batch * imgs_per_id:
                unique_labels = targets[0:targets.size(0):imgs_per_id]  # 直接按间隔取每个 ID
            else:
                unique_labels = targets.cpu().unique().cuda()  # 去重得到 ID 列表
        else:  # CPU 情况
            if ordered and targets.size(0) == ids_per_batch * imgs_per_id:
                unique_labels = targets[0:targets.size(0):imgs_per_id]
            else:
                unique_labels = targets.unique()

        # 初始化存储变量
        inter_min_distance = torch.zeros(unique_labels.size(0))  # 类间最小距离
        intra_max_distance = torch.zeros(unique_labels.size(0))  # 类内最大距离
        center_features = torch.zeros(unique_labels.size(0), features.size(1))  # 各类中心特征

        if self.use_gpu:  # 如果使用 GPU
            inter_min_distance = inter_min_distance.cuda()
            intra_max_distance = intra_max_distance.cuda()
            center_features = center_features.cuda()

        index = torch.range(0, unique_labels.size(0) - 1)  # 类别索引

        # 计算类内最大距离
        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_features = features[targets == label]  # 当前类的所有样本
            center_features[i] = same_class_features.mean(dim=0)  # 当前类中心
            intra_class_distance = self._euclidean_dist(center_features[index == i], same_class_features)
            intra_max_distance[i] = intra_class_distance.max()  # 类内最大距离

        # 计算类间最小距离
        for i in range(unique_labels.size(0)):
            inter_class_distance = self._euclidean_dist(center_features[index == i], center_features[index != i])
            inter_min_distance[i] = inter_class_distance.min()  # 类间最小距离

        # 计算最终聚类损失
        cluster_loss = torch.mean(torch.relu(intra_max_distance - inter_min_distance + self.margin))
        return cluster_loss, intra_max_distance, inter_min_distance

    def forward(self, features, targets):
        # 调用 _cluster_loss 计算损失
        # 输出 损失值、类内最大距离、类间最小距离
        assert features.size(0) == targets.size(0), "features.size(0) is not equal to targets.size(0)"
        cluster_loss, cluster_dist_ap, cluster_dist_an = self._cluster_loss(
            features, targets, self.ordered, self.ids_per_batch, self.imgs_per_id)
        return cluster_loss, cluster_dist_ap, cluster_dist_an


class ClusterLoss_local(nn.Module):
    def __init__(self, margin=10, use_gpu=True, ordered=True, ids_per_batch=32, imgs_per_id=4):
        super(ClusterLoss_local, self).__init__()
        self.use_gpu = use_gpu
        self.margin = margin #用于聚类损失的边界值。这个值控制了不同类之间的距离与同类之间的距离的最小差距。
        self.ordered =   ordered  # 如果数据是有序的，即每个 ID 有多个图像（例如，每个 ID 包含多个图片），则设置为 True，否则为 False
        self.ids_per_batch = ids_per_batch   # 每个批次中包含的不同 ID 数量
        self.imgs_per_id =  imgs_per_id     # 每个 ID 包含的图像数量

    def _euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _shortest_dist(self, dist_mat):
        """Parallel version.
        Args:
          dist_mat: pytorch Variable, available shape:
            1) [m, n]
            2) [m, n, N], N is batch size
            3) [m, n, *], * can be arbitrary additional dimensions
        Returns:
          dist: three cases corresponding to `dist_mat`:
            1) scalar
            2) pytorch Variable, with shape [N]
            3) pytorch Variable, with shape [*]
        """
        m, n = dist_mat.size()[:2]
        # Just offering some reference for accessing intermediate distance.
        dist = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if (i == 0) and (j == 0):
                    dist[i][j] = dist_mat[i, j]
                elif (i == 0) and (j > 0):
                    dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
                elif (i > 0) and (j == 0):
                    dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
                else:
                    dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
        dist = dist[-1][-1]
        return dist

    def _local_dist(self, x, y):
        # 计算两个输入张量 x 和 y 之间的局部距离。首先将 x 和 y 进行展平，然后计算它们之间的欧氏距离，接着通过一个非线性函数对距离进行变换，最后得到最终的局部距离矩阵。
        """
        Args:
          x: pytorch Variable, with shape [M, m, d]
          y: pytorch Variable, with shape [N, n, d]
        Returns:
          dist: pytorch Variable, with shape [M, N]
        """
        M, m, d = x.size()
        N, n, d = y.size()
        x = x.contiguous().view(M * m, d)
        y = y.contiguous().view(N * n, d)
        # shape [M * m, N * n]
        dist_mat = self._euclidean_dist(x, y)
        dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
        # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
        dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
        # shape [M, N]
        dist_mat = self._shortest_dist(dist_mat)
        return dist_mat

    def _cluster_loss(self, features, targets, ordered=True, ids_per_batch=32, imgs_per_id=4):
        # 计算聚类损失，核心思想是通过最小化同类样本之间的最大距离，同时最大化不同类样本之间的最小距离。

        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, H, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             cluster_loss
        """
        if self.use_gpu:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.cpu().unique().cuda()
            else:
                unique_labels = targets.cpu().unique().cuda()
        else:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.unique()
            else:
                unique_labels = targets.unique()

        inter_min_distance = torch.zeros(unique_labels.size(0))
        intra_max_distance = torch.zeros(unique_labels.size(0))
        center_features = torch.zeros(unique_labels.size(0), features.size(1), features.size(2))

        if self.use_gpu:
            inter_min_distance = inter_min_distance.cuda()
            intra_max_distance = intra_max_distance.cuda()
            center_features = center_features.cuda()

        index = torch.range(0, unique_labels.size(0) - 1)
        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_features = features[targets == label]
            center_features[i] = same_class_features.mean(dim=0)
            intra_class_distance = self._local_dist(center_features[index == i], same_class_features)
            # print('intra_class_distance', intra_class_distance)
            intra_max_distance[i] = intra_class_distance.max()
        # print('intra_max_distance:', intra_max_distance)

        for i in range(unique_labels.size(0)):
            inter_class_distance = self._local_dist(center_features[index == i], center_features[index != i])
            # print('inter_class_distance', inter_class_distance)
            inter_min_distance[i] = inter_class_distance.min()
        # print('inter_min_distance:', inter_min_distance)

        cluster_loss = torch.mean(torch.relu(intra_max_distance - inter_min_distance + self.margin))
        return cluster_loss, intra_max_distance, inter_min_distance

    def forward(self, features, targets):
        # 该损失函数的主要计算部分。它确保输入的特征和标签的大小匹配
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, H, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             cluster_loss
        """
        assert features.size(0) == targets.size(0), "features.size(0) is not equal to targets.size(0)"
        cluster_loss, cluster_dist_ap, cluster_dist_an = self._cluster_loss(features, targets, self.ordered,
                                                                            self.ids_per_batch, self.imgs_per_id)
        return cluster_loss, cluster_dist_ap, cluster_dist_an


if __name__ == '__main__':
    use_gpu = True
    cluster_loss = ClusterLoss(use_gpu=use_gpu, ids_per_batch=4, imgs_per_id=4)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).cuda()
    loss = cluster_loss(features, targets)
    print(loss)

    cluster_loss_local = ClusterLoss_local(use_gpu=use_gpu, ids_per_batch=4, imgs_per_id=4)
    features = torch.rand(16, 8, 2048)
    targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    if use_gpu:
        features = torch.rand(16, 8, 2048).cuda()
        targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).cuda()
    loss = cluster_loss_local(features, targets)
    print(loss)