from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

# 随即去采样
class RandomIdentitySampler(Sampler):
    """
    在每个 batch 中，随机采样 N 个行人 ID，每个 ID 采样 K 张图像，
    因此一个 batch 的总大小是 N*K。

    参数:
    - data_source (list): 数据列表，格式为 (img_path, pid, camid, trackid)
    - num_instances (int): 每个 ID 在一个 batch 中采样的图像数量 K
    - batch_size (int): 一个 batch 的总大小 N*K
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        # 每个 batch 中包含多少个不同的 ID
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # 存储每个 ID 对应的图像索引

        # 遍历数据集，把每个 pid 对应的图像索引收集起来
        # 例如 {pid1: [img1_idx, img2_idx, ...], pid2: [...], ...}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())  # 所有 pid 列表

        # 估算一个 epoch 的总样本数
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                # 如果某个 ID 图像数不够 K 张，用重复采样补齐
                num = self.num_instances
            # 只保留能被 K 整除的部分
            self.length += num - num % self.num_instances

    def __iter__(self):
        """
        生成一个 epoch 的采样索引
        """
        batch_idxs_dict = defaultdict(list)  # 存每个 pid 分好批次的索引

        # 遍历每个 pid，把它的图像索引按 K 张一组分好批次
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                # 图像不足 K 张，随机重复采样补齐
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)  # 打乱顺序
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    # 每收集 K 张，就作为一个小 batch 存起来
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        # 所有可用的 pid
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []  # 最终输出的索引顺序

        # 不断随机选 N 个 pid，每个 pid 取一组 K 张图像，直到 pid 不够 N 个
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)  # 取该 pid 的一组 K 张图像
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    # 如果该 pid 的图像都用完了，就从候选列表移除
                    avai_pids.remove(pid)

        return iter(final_idxs)  # 返回一个迭代器

    def __len__(self):
        """
        返回一个 epoch 的总样本数
        """
        return self.length
