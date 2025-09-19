from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import numpy as np
import math
import torch.distributed as dist
import torch
import pickle

_LOCAL_PROCESS_GROUP = None


def _get_global_gloo_group():
    """
    获取一个 gloo 通信组，用于所有 rank 间的通信
    如果 backend 是 nccl，则创建一个新的 gloo 组，否则返回默认组
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    """
    将任意 Python 对象序列化为 Tensor，以便分布式通信
    """
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    # gloo 用 CPU，nccl 用 GPU
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    # 使用 pickle 序列化数据
    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        print(f"Rank {dist.get_rank()} trying to all-gather {len(buffer) / (1024 ** 3):.2f} GB of data on device {device}")
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    将所有 rank 的 tensor 填充到相同大小，方便 all_gather
    返回每个 rank 的原始大小，以及填充后的最大 tensor
    """
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "all_gather 必须在 group 内所有 rank 调用"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)

    # 收集所有 rank 的 tensor 大小
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 如果当前 tensor 小于最大大小，补 0 填充
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    在所有 rank 上收集任意 picklable 对象
    返回每个 rank 的数据列表
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # 收集所有 rank 的 tensor
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    # 反序列化为原始数据
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def shared_random_seed():
    """
    生成一个在所有 rank 上相同的随机种子
    方便分布式训练时同步随机性
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


class RandomIdentitySampler_DDP(Sampler):
    """
    分布式随机身份采样器：
    - 每个 batch 采样 N 个不同的身份 (pid)
    - 每个身份随机采样 K 张图像
    - 总 batch size = N * K，分给多个 rank

    参数:
    - data_source: 数据集列表 (img_path, pid, camid, trackid)
    - batch_size: 总 batch 大小
    - num_instances: 每个 pid 采样的图像数量 K
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        # 每个 rank 处理的 mini-batch 大小
        self.mini_batch_size = self.batch_size // self.world_size
        # 每个 rank 采样的 pid 数量
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        # 建立 pid 到图像索引的映射
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # 计算一个 epoch 的样本数量
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.rank = dist.get_rank()
        # 每个 rank 分到的样本数
        self.length //= self.world_size

    def __iter__(self):
        # 同步随机种子，保证所有 rank 采样一致
        seed = shared_random_seed()
        np.random.seed(seed)
        self._seed = int(seed)

        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        # 只取当前 rank 的索引
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __fetch_current_node_idxs(self, final_idxs, length):
        """
        获取当前 rank 负责的样本索引
        """
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(self.mini_batch_size * self.rank + self.mini_batch_size * i,
                          min(self.mini_batch_size * self.rank + self.mini_batch_size * (i + 1), total_num))
            index_target.extend(index)
        index_target_npy = np.array(index_target)
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs

    def sample_list(self):
        """
        采样所有 rank 共享的索引列表
        """
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}
        batch_indices = []

        # 不断随机选择 pid，每个 pid 采 K 张图像
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances:
                    avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length
