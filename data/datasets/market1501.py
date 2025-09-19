# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob         # 用于批量获取文件路径，支持通配符
import re           # 正则表达式模块，用于解析文件名
import os.path as osp  # 文件路径操作模块

from .bases import BaseImageDataset  # 继承自 BaseImageDataset，提供统计与基础功能


class Market1501(BaseImageDataset):
    """
    Market1501 数据集类
    参考:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    数据集统计:
    - identities: 1501 (+1 background)
    - images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'   # 数据集目录名

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Market1501, self).__init__()
        # 设置各目录路径
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # 检查数据目录是否完整
        self._check_before_run()

        self.pid_begin = pid_begin  # pid 起始编号
        # 处理三个子集
        train = self._process_dir(self.train_dir, relabel=True)  # 训练集重新标号
        query = self._process_dir(self.query_dir, relabel=False)  # 查询集不重新标号
        gallery = self._process_dir(self.gallery_dir, relabel=False)  # 检索集不重新标号

        # 打印统计信息
        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        # 保存处理结果
        self.train = train
        self.query = query
        self.gallery = gallery

        # 分别统计 train/query/gallery 的 ID、图片数、摄像头数、视角数
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """运行前检查路径完整性"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

# ************ 处理数据目录 ************
    def _process_dir(self, dir_path, relabel=False):
        # 读取目录下所有 .jpg 文件
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # 文件名格式: '0002_c1_f0046182.jpg'
        # 0002 表示 pid，c1 表示摄像头编号
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        # 第一次遍历：收集所有 pid
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # pid=-1 表示无效图片，忽略
            pid_container.add(pid)

        # 生成 pid -> label 映射，保证 pid 连续编号
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        # 第二次遍历：处理每张图
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # 忽略无效图片
            assert 0 <= pid <= 1501  # pid=0 表示背景
            assert 1 <= camid <= 6  # 摄像头编号在 1~6 之间
            camid -= 1  # 从 0 开始编号

            # 需要重新标号则替换为新 pid
            if relabel: pid = pid2label[pid]

            # 每张图存成 (路径, pid, 摄像头id, 轨迹id)
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset


