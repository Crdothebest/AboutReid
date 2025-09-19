import torch
import os
from utils.reranking import re_ranking
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns  # 使用Seaborn库绘制KDE图


def euclidean_distance(qf, gf):
    m = qf.shape[0]  # 获取查询特征的样本数量 m
    n = gf.shape[0]  # 获取数据库特征的样本数量 n

    # 计算查询特征和数据库特征的欧式距离矩阵
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # 第一部分是每个查询样本和数据库样本的平方和
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    # 第二部分通过矩阵相乘得到每对查询样本和数据库样本的内积，然后加到原矩阵中

    return dist_mat.cpu().numpy()  # 返回计算得到的欧式距离矩阵，并转为numpy格式



def cosine_similarity(qf, gf):
    epsilon = 0.00001  # 防止除零错误的小常数
    dist_mat = qf.mm(gf.t())  # 计算查询特征和数据库特征的内积

    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # 计算查询特征的 L2 范数，得到 (m, 1)
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # 计算数据库特征的 L2 范数，得到 (n, 1)

    qg_normdot = qf_norm.mm(gf_norm.t())  # 计算查询和数据库特征的范数的外积，得到 (m, n)

    dist_mat = dist_mat.mul(1 / qg_normdot)  # 对内积矩阵进行范数归一化
    dist_mat = dist_mat.cpu().numpy()  # 转为 numpy 数组

    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)  # 防止数值溢出，将值限制在[-1, 1]之间
    dist_mat = np.arccos(dist_mat)  # 计算余弦相似度的角度（弧度制）

    return dist_mat  # 返回余弦相似度矩阵


def eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape  # 获取查询和数据库样本的数量

    if num_g < max_rank:  # 如果数据库样本数量小于最大排名数量
        max_rank = num_g  # 调整最大排名为数据库样本数
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)  # 对距离矩阵按行进行排序，得到每个查询与所有数据库样本的排序索引

    query_arg = np.argsort(q_pids, axis=0)  # 按照查询的pid（身份标签）排序
    result = g_pids[indices]  # 根据排序的索引获取数据库的身份标签
    gall_re = result[query_arg]  # 按照查询身份标签排序数据库的结果
    gall_re = gall_re.astype(np.str)  # 转换为字符串类型

    result = gall_re[:, :100]  # 取出前100个最匹配的数据库样本（最大排名100）

    with open('re.txt', 'w') as f:  # 创建一个文件保存rank list
        f.write('rank list file\n')

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # 计算查询样本与数据库样本是否匹配

    # compute cmc curve for each query
    all_cmc = []  # 用于保存所有查询的 CMC 曲线
    all_AP = []  # 用于保存所有查询的平均精度（AP）
    num_valid_q = 0.  # 有效查询的数量
    for q_idx in range(num_q):  # 遍历每个查询
        q_pid = q_pids[q_idx]  # 查询样本的pid
        q_camid = q_camids[q_idx]  # 查询样本的摄像头id
        q_sceneid = q_sceneids[q_idx]  # 查询样本的场景id

        order = indices[q_idx]  # 获取该查询在数据库中的排序

        # 根据新的协议，移除具有相同pid和场景id的样本
        remove = (g_pids[order] == q_pid) & (g_sceneids[order] == q_sceneid)
        keep = np.invert(remove)  # 保留满足条件的样本

        # 写入到rank list文件
        with open('re.txt', 'a') as f:
            f.write('{}_s{}_v{}:\n'.format(q_pid, q_sceneid, q_camid))  # 保存查询样本的pid, sceneid, camid
            v_ids = g_pids[order][keep][:max_rank]  # 获取前max_rank个符合条件的数据库样本pid
            v_cams = g_camids[order][keep][:max_rank]  # 获取对应的摄像头id
            v_scenes = g_sceneids[order][keep][:max_rank]  # 获取对应的场景id
            for vid, vcam, vscene in zip(v_ids, v_cams, v_scenes):  # 遍历数据库中的结果
                f.write('{}_s{}_v{}  '.format(vid, vscene, vcam))  # 格式化输出
            f.write('\n')

        orig_cmc = matches[q_idx][keep]  # 获取当前查询样本的 CMC

        if not np.any(orig_cmc):  # 如果当前查询没有匹配项，跳过
            continue

        cmc = orig_cmc.cumsum()  # 计算 CMC 曲线
        cmc[cmc > 1] = 1  # 保证 CMC 曲线中只有1和0

        all_cmc.append(cmc[:max_rank])  # 将计算出的 CMC 曲线加入列表
        num_valid_q += 1.  # 有效查询数加1

        # 计算平均精度（AP）
        num_rel = orig_cmc.sum()  # 获取相关匹配的数量
        tmp_cmc = orig_cmc.cumsum()  # 累加 CMC
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]  # 计算精度
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc  # 计算加权精度
        AP = tmp_cmc.sum() / num_rel  # 计算平均精度
        all_AP.append(AP)  # 保存到 AP 列表

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"  # 确保有有效的查询样本

    all_cmc = np.asarray(all_cmc).astype(np.float32)  # 转为 numpy 数组并转换为 float32 类型
    all_cmc = all_cmc.sum(0) / num_valid_q  # 计算平均 CMC 曲线
    mAP = np.mean(all_AP)  # 计算 mAP

    return all_cmc, mAP  # 返回计算结果

# q_pids 查询样本的pid
# g_pids 数据库样本的pid
# q_camids 查询样本的摄像头id
# g_camids 数据库样本的摄像头id
# max_rank 最大排名
# distmat 距离矩阵  
def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape  # 获取查询和数据库样本的数量
    # distmat 示例：
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:  # 如果数据库样本数量小于最大排名数量
        max_rank = num_g  # 调整最大排名为数据库样本数
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)  # 按行对距离矩阵排序，返回排序后的索引
    # 排序后的索引示例：
    #  0 2 1 3
    #  1 2 3 0

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # 计算每个查询样本与数据库样本是否匹配

    # compute cmc curve for each query
    all_cmc = []  # 用于保存所有查询的 CMC 曲线
    all_AP = []  # 用于保存所有查询的平均精度（AP）
    num_valid_q = 0.  # 有效查询的数量

    for q_idx in range(num_q):  # 遍历每个查询
        q_pid = q_pids[q_idx]  # 查询样本的pid
        q_camid = q_camids[q_idx]  # 查询样本的摄像头id

        order = indices[q_idx]  # 获取该查询在数据库中的排序

        # 移除与查询样本具有相同pid和摄像头ID的数据库样本
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)  # 保留符合条件的样本

        # 计算 CMC 曲线
        orig_cmc = matches[q_idx][keep]  # 获取当前查询样本的 CMC
        if not np.any(orig_cmc):  # 如果没有匹配项，跳过该查询
            continue

        cmc = orig_cmc.cumsum()  # 计算 CMC 曲线（累加和）
        cmc[cmc > 1] = 1  # 保证 CMC 曲线中的值仅为 0 或 1

        all_cmc.append(cmc[:max_rank])  # 将计算出的 CMC 曲线加入列表
        num_valid_q += 1.  # 有效查询数加1

        # 计算平均精度（AP）
        num_rel = orig_cmc.sum()  # 获取相关匹配的数量
        tmp_cmc = orig_cmc.cumsum()  # 累加 CMC
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]  # 计算精度（这行代码已被注释掉）
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0  # 生成排名序列
        tmp_cmc = tmp_cmc / y  # 计算排名下的精度
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc  # 计算加权精度
        AP = tmp_cmc.sum() / num_rel  # 计算平均精度
        all_AP.append(AP)  # 保存到 AP 列表

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"  # 确保至少有一个有效查询

    all_cmc = np.asarray(all_cmc).astype(np.float32)  # 转换为 numpy 数组并转换为 float32 类型
    all_cmc = all_cmc.sum(0) / num_valid_q  # 计算平均 CMC 曲线
    mAP = np.mean(all_AP)  # 计算 mAP

    return all_cmc, mAP  # 返回计算结果


class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        q_sceneids = np.asarray(self.sceneids[:self.num_query])  # zxp
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        g_sceneids = np.asarray(self.sceneids[self.num_query:])  # zxp

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def showPointMultiModal(self, features, real_label, draw_label, save_path='..'):
        id_show = 25
        num_ids = len(np.unique(real_label))
        save_path = os.path.join(save_path, str(draw_label) + ".pdf")
        print("Draw points of features to {}".format(save_path))
        indices = find_label_indices(real_label, draw_label, max_indices_per_label=id_show)
        feat = features[indices]
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=1, learning_rate=100, perplexity=60)
        features_tsne = tsne.fit_transform(feat)
        colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99',
                  '#fdbf6f', '#cab2d6', '#ffff99']
        MARKS = ['*']
        plt.figure(figsize=(10, 10))
        for i in range(features_tsne.shape[0]):
            plt.scatter(features_tsne[i, 0], features_tsne[i, 1], s=300, color=colors[i // id_show], marker=MARKS[0],
                        alpha=0.4)
        plt.title("t-SNE Visualization of Different IDs")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        # plt.legend()
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        # self.showPointMultiModal(feats, real_label=self.pids,
        #                          draw_label=[258, 260, 269, 271, 273, 280, 282, 284, 285, 286, 287, 289])
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


def find_label_indices(label_list, target_labels, max_indices_per_label=1):
    indices = []
    counts = {label: 0 for label in target_labels}
    for index, label in enumerate(label_list):
        if label in target_labels and counts[label] < max_indices_per_label:
            indices.append(index)
            counts[label] += 1
    sorted_indices = sorted(indices, key=lambda index: (label_list[index], index))
    return sorted_indices


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def _calculate_similarity(pre_fusion_tokens, post_fusion_tokens):
    """
    计算融合前后patch token的相似度

    Args:
        pre_fusion_tokens: 融合前patch token
        post_fusion_tokens: 融合后patch token

    Returns:
        similarities: 融合前后patch token的相似度
    """

    # 计算余弦相似度
    similarities = torch.nn.functional.cosine_similarity(pre_fusion_tokens, post_fusion_tokens,
                                                         dim=-1).cpu().detach().numpy()

    # # 将相似度平均到每个patch
    # similarities = torch.mean(similarities, dim=1).squeeze().cpu().detach().numpy()

    return similarities


def visualize_similarity(pre_fusion_src_tokens, pre_fusion_tgt_tokens, post_fusion_src_tokens, post_fusion_tgt_tokens,
                         writer=None, epoch=None, mode=1,
                         pattern=None, figure_size=(6, 6), seaborn_style='whitegrid'):
    """
    可视化融合前后patch token的相似度分布

    Args:
        pre_fusion_src_tokens: 融合前源图像patch token
        pre_fusion_tgt_tokens: 融合前目标图像patch token
        post_fusion_src_tokens: 融合后源图像patch token
        post_fusion_tgt_tokens: 融合后目标图像patch token
        writer: tensorboardX SummaryWriter
        epoch: epoch
        mode: 模式，1代表源图像，2代表目标图像
        pattern: 融合模式，r2t, r2n, n2t, n2r, t2r, t2n
        figure_size: 图像大小
        seaborn_style: seaborn风格

    Returns:
        None
    """

    # 计算融合前后patch token的相似度
    similarities_ori = _calculate_similarity(pre_fusion_src_tokens, pre_fusion_tgt_tokens)
    similarities = _calculate_similarity(post_fusion_src_tokens, post_fusion_tgt_tokens)

    # 设置seaborn风格
    sns.set(style=seaborn_style)

    # 创建画图对象
    fig, ax = plt.subplots(figsize=figure_size)

    # 绘制融合前后相似度分布图
    sns.kdeplot(similarities, color='b', label='Before MA', ax=ax, multiple="stack")
    sns.kdeplot(similarities_ori, color='g', label='After MA', ax=ax, multiple="stack")

    # 设置标题和标签
    if pattern == 'r2t':
        sign = 'R and T'
    elif pattern == 'r2n':
        sign = 'R and N'
    elif pattern == 'n2t':
        sign = 'N and T'
    elif pattern == 'n2r':
        sign = 'N and R'
    elif pattern == 't2r':
        sign = 'T and R'
    elif pattern == 't2n':
        sign = 'T and N'
    plt.title("Similarity Distribution between {}".format(sign), fontsize=18, fontweight='bold')
    plt.xlabel("Cosine Similarity", fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')
    # 设置x轴刻度标签大小
    plt.xticks(fontsize=15)

    # 设置y轴刻度标签大小
    plt.yticks(fontsize=15)
    # 添加图例
    plt.legend(loc='upper right', fontsize=17)

    # 显示图像
    plt.show()

    # 将图像写入tensorboard
    if writer is not None:
        writer.add_figure('Similarity_{}'.format(sign), plt.gcf(), epoch)
