#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09


"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (model tensor)
probFea: all feature vectors of the gallery set (model tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

import numpy as np
import torch


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # 获取查询样本数量和总样本数量
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)

    # 如果只使用局部距离矩阵，则直接使用传入的矩阵
    if only_local:
        original_dist = local_distmat
    else:
        # 拼接查询和数据库特征，计算全局距离矩阵
        feat = torch.cat([probFea, galFea])  # 拼接查询和数据库特征
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
                  torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1, -2, feat, feat.t())  # 计算欧式距离
        original_dist = distmat.cpu().numpy()  # 转换为 numpy 数组以便后续操作
        del feat  # 删除特征矩阵，以节省内存

        # 如果有局部距离矩阵，将其加到原始距离矩阵中
        if not local_distmat is None:
            original_dist = original_dist + local_distmat

    gallery_num = original_dist.shape[0]
    # 对原始距离进行归一化处理
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)  # 初始化加权相似度矩阵
    initial_rank = np.argsort(original_dist).astype(np.int32)  # 获取初步的排序结果

    # 使用 k-近邻和 k-互惠邻居进行重排序
    for i in range(all_num):
        # 计算 k-近邻
        forward_k_neigh_index = initial_rank[i, :k1 + 1]  # 获取前 k1 个近邻
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]  # 获取反向的 k1 个近邻
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  # 计算 k-互惠邻居
        k_reciprocal_expansion_index = k_reciprocal_index

        # 扩展 k-互惠邻居
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            # 如果互惠邻居的交集足够大，扩展邻居集合
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  # 去重
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])  # 计算权重
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)  # 归一化权重

    original_dist = original_dist[:query_num,]  # 保留查询样本的距离

    # 如果 k2 不是 1，则进一步进行扩展
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)  # 使用 k2 近邻计算平均值
        V = V_qe
        del V_qe

    del initial_rank  # 删除初始排序数组
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])  # 获取非零权重的索引

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)  # 初始化 Jaccard 距离矩阵

    # 计算 Jaccard 距离
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]  # 获取非零权重的索引
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)  # 计算 Jaccard 距离

    # 最终距离矩阵是 Jaccard 距离和原始距离的加权平均
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]  # 保留查询与数据库之间的距离
    return final_dist  # 返回最终的优化后的距离矩阵
