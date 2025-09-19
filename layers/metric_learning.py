import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math

# 以下都是用于度量学习的损失函数

# 对比损失
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # 设置对比损失中的边际（margin）

    def forward(self, inputs, targets):
        n = inputs.size(0)  # 获取样本数（批次大小）
        # 计算相似度矩阵
        sim_mat = torch.matmul(inputs, inputs.t())  # 计算输入特征的相似度矩阵
        targets = targets  # 获取目标标签
        loss = list()  # 存储每一对样本的损失
        c = 0  # 用于损失的计数

        for i in range(n):
            # 获取与当前样本标签相同的正样本对
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])
            # 过滤掉与自身的相似度
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            # 获取与当前样本标签不同的负样本对
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            # 对正样本和负样本进行排序
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            # 过滤掉负样本对中大于 margin 的部分
            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            neg_loss = 0  # 初始化负样本损失

            # 计算正样本损失：与正样本的相似度越小，损失越大
            pos_loss = torch.sum(-pos_pair_ + 1)
            # 如果存在负样本，则计算负样本损失：与负样本的相似度越大，损失越小
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            # 将正负样本损失相加
            loss.append(pos_loss + neg_loss)

        # 计算所有样本的平均损失
        loss = sum(loss) / n
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # 设置对比损失中的边际（margin）

    def forward(self, inputs, targets):
        n = inputs.size(0)  # 获取样本数（批次大小）
        # 计算相似度矩阵
        sim_mat = torch.matmul(inputs, inputs.t())  # 计算输入特征的相似度矩阵
        targets = targets  # 获取目标标签
        loss = list()  # 存储每一对样本的损失
        c = 0  # 用于损失的计数

        for i in range(n):
            # 获取与当前样本标签相同的正样本对
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])
            # 过滤掉与自身的相似度
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            # 获取与当前样本标签不同的负样本对
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            # 对正样本和负样本进行排序
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            # 过滤掉负样本对中大于 margin 的部分
            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            neg_loss = 0  # 初始化负样本损失

            # 计算正样本损失：与正样本的相似度越小，损失越大
            pos_loss = torch.sum(-pos_pair_ + 1)
            # 如果存在负样本，则计算负样本损失：与负样本的相似度越大，损失越小
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            # 将正负样本损失相加
            loss.append(pos_loss + neg_loss)

        # 计算所有样本的平均损失
        loss = sum(loss) / n
        return loss


class CircleLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=256, m=0.25):
        super(CircleLoss, self).__init__()
        # 初始化分类权重
        self.weight = Parameter(torch.Tensor(num_classes, in_features))
        self.s = s  # 缩放因子
        self.m = m  # 边际（margin）
        self._num_classes = num_classes  # 类别数量
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 He 初始化方法初始化权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __call__(self, bn_feat, targets):
        # 计算归一化后的输入特征与权重之间的相似度矩阵
        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        # 计算正负样本的 alpha 值
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        # 计算正负样本的缩放值
        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        # 将标签转换为 one-hot 编码
        targets = F.one_hot(targets, num_classes=self._num_classes)

        # 计算最终的预测类别 logits
        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits

# 大角度边际损失
class Arcface(nn.Module):
    r"""Implement of large margin arc distance: cos(theta + m)"""
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0):
        super(Arcface, self).__init__()
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.s = s  # 特征缩放因子
        self.m = m  # 边际（margin）
        self.ls_eps = ls_eps  # 标签平滑参数
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # 类别权重
        nn.init.xavier_uniform_(self.weight)  # 使用 Xavier 初始化权重

        self.easy_margin = easy_margin  # 是否使用简单边际
        self.cos_m = math.cos(m)  # cos(m)
        self.sin_m = math.sin(m)  # sin(m)
        self.th = math.cos(math.pi - m)  # cos(pi - m)
        self.mm = math.sin(math.pi - m) * m  # sin(pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # 计算余弦相似度
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # 计算 sin(theta)
        phi = cosine * self.cos_m - sine * self.sin_m  # 计算加边际后的余弦值
        phi = phi.type_as(cosine)  # 保持数据类型一致
        if self.easy_margin:
            # 如果使用简单边际，则将小于零的值保留为余弦值
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 否则，如果余弦值小于 cos(pi - m)，则使用 phi - mm
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')  # 初始化 one-hot 标签
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 将标签转为 one-hot
        if self.ls_eps > 0:
            # 如果启用了标签平滑，则进行平滑处理
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # 计算最终输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # 缩放输出

        return output


class Cosface(nn.Module):
    r"""Implement of large margin cosine distance: cos(theta) - m"""
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(Cosface, self).__init__()
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.s = s  # 特征缩放因子
        self.m = m  # 边际（margin）
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # 类别权重
        nn.init.xavier_uniform_(self.weight)  # 使用 Xavier 初始化权重

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # 计算余弦相似度
        phi = cosine - self.m  # 计算加边际后的余弦值
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')  # 初始化 one-hot 标签
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 将标签转为 one-hot
        # 计算最终输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # 缩放输出
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(AMSoftmax, self).__init__()
        self.m = m  # 边际（margin）
        self.s = s  # 特征缩放因子
        self.in_feats = in_features  # 输入特征维度
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)  # 类别权重
        self.ce = nn.CrossEntropyLoss()  # 交叉熵损失
        nn.init.x



class Arcface(nn.Module):
    r"""Implement of large margin arc distance: cos(theta + m)"""
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0):
        super(Arcface, self).__init__()
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.s = s  # 特征缩放因子
        self.m = m  # 边际（margin）
        self.ls_eps = ls_eps  # 标签平滑参数
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # 类别权重
        nn.init.xavier_uniform_(self.weight)  # 使用 Xavier 初始化权重

        self.easy_margin = easy_margin  # 是否使用简单边际
        self.cos_m = math.cos(m)  # cos(m)
        self.sin_m = math.sin(m)  # sin(m)
        self.th = math.cos(math.pi - m)  # cos(pi - m)
        self.mm = math.sin(math.pi - m) * m  # sin(pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # 计算余弦相似度
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # 计算 sin(theta)
        phi = cosine * self.cos_m - sine * self.sin_m  # 计算加边际后的余弦值
        phi = phi.type_as(cosine)  # 保持数据类型一致
        if self.easy_margin:
            # 如果使用简单边际，则将小于零的值保留为余弦值
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 否则，如果余弦值小于 cos(pi - m)，则使用 phi - mm
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')  # 初始化 one-hot 标签
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 将标签转为 one-hot
        if self.ls_eps > 0:
            # 如果启用了标签平滑，则进行平滑处理
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # 计算最终输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # 缩放输出

        return output


class Cosface(nn.Module):
    r"""Implement of large margin cosine distance: cos(theta) - m"""
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(Cosface, self).__init__()
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.s = s  # 特征缩放因子
        self.m = m  # 边际（margin）
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # 类别权重
        nn.init.xavier_uniform_(self.weight)  # 使用 Xavier 初始化权重

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # 计算余弦相似度
        phi = cosine - self.m  # 计算加边际后的余弦值
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')  # 初始化 one-hot 标签
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 将标签转为 one-hot
        # 计算最终输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # 缩放输出
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(AMSoftmax, self).__init__()
        self.m = m  # 边际（margin）
        self.s = s  # 特征缩放因子
        self.in_feats = in_features  # 输入特征维度
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)  # 类别权重
        self.ce = nn.CrossEntropyLoss()  # 交叉熵损失
        nn.init.x



class CircleLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=256, m=0.25):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(num_classes, in_features))
        self.s = s
        self.m = m
        self._num_classes = num_classes
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __call__(self, bn_feat, targets):

        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        targets = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits


class Arcface(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0):
        super(Arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type_as(cosine)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class Cosface(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(Cosface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_features
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        delt_costh = torch.zeros(costh.size(), device='cuda').scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        return costh_m_s