# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os.path as osp
from .bases import BaseImageDataset
# glob：批量匹配图像路径
# re：通过正则表达式解析 PID 和摄像头 ID
# BaseImageDataset：提供统计方法

class RGBNT100(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'RGBNT100/rgbir'

# ************ 初始化流程 ***************
    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(RGBNT100, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
    # 流程和 MSVR310 类似，主要区别是 RGBNT100 直接在一个目录下存所有 JPG 文件，而 MSVR310 是每个视频 ID 一个文件夹，里面有 vis/ni/th 三种模态。

    # 和 MSVR310 一样，先检查路径，避免后续报错
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))



    def _process_dir(self, dir_path, relabel=False):
        # 获取指定目录下所有 jpg 图片路径
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        # 定义正则表达式，提取文件名中的 pid（行人ID）和 camid（摄像头ID）
        # 文件名格式假设为：xxxx_cx.jpg 其中 xxxx 是 pid，cx 是摄像头 ID
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        # 用于收集所有出现过的 pid
        pid_container = set()
        for img_path in img_paths:
            # 从文件名中提取 pid 和 camid
            pid, _ = map(int, pattern.search(img_path).groups())
            # pid = -1 的图像是无效图像，跳过
            if pid == -1:
                continue
            pid_container.add(pid)

        # 将 pid 映射到连续的标签，例如 {10:0, 25:1, 42:2, ...}
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            # 再次解析 pid 和 camid
            pid, camid = map(int, pattern.search(img_path).groups())
            # 确保 pid 和 camid 在合理范围内
            assert 1 <= pid <= 600
            assert 1 <= camid <= 8
            trackid = -1  # 这里暂时没有轨迹ID，统一设为 -1
            camid -= 1  # camid 从 0 开始编号，原始数据可能从 1 开始
            # 如果是训练集，需要把 pid 映射到连续标签
            if relabel:
                pid = pid2label[pid]
            # 保存一条数据 (图片路径, 行人ID, 摄像头ID, 轨迹ID)
            dataset.append((img_path, pid, camid, trackid))

        # 返回处理好的数据列表
        return dataset
