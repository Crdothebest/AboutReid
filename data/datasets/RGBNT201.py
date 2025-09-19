from __future__ import division, print_function, absolute_import
import glob
import warnings
import os.path as osp
from .bases import BaseImageDataset


class RGBNT201(BaseImageDataset):
    # 数据集根目录名称
    dataset_dir = 'RGBNT201'

    def __init__(self, root='', verbose=True, **kwargs):
        super(RGBNT201, self).__init__()
        # 处理根目录路径，支持 ~ 用户目录
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # 兼容旧的数据集目录结构
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.')  # 警告旧目录结构

        # 训练、查询、测试集路径
        self.train_dir = osp.join(self.data_dir, 'train_171')
        self.query_dir = osp.join(self.data_dir, 'test')
        self.gallery_dir = osp.join(self.data_dir, 'test')

        # 检查数据路径是否存在
        self._check_before_run()

        # 处理各个数据集目录
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        # 如果 verbose=True，打印统计信息
        if verbose:
            print("=> RGBNT201 loaded")
            self.print_dataset_statistics(train, query, gallery)

        # 保存数据
        self.train = train
        self.query = query
        self.gallery = gallery

        # 获取各个数据集的统计信息：ID数、图像数、摄像头数、轨迹数
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """检查所有文件夹是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        处理单个目录，返回格式：
        [( [RGB图路径, NI图路径, TI图路径], 行人ID, 摄像头ID, 轨迹ID ), ...]
        """
        # 获取 RGB 图像路径
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))

        # 收集所有行人ID
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            jpg_name = img_path_RGB.split('/')[-1]  # 获取文件名
            pid = int(jpg_name.split('_')[0][0:6])  # 文件名前6位为行人ID
            pid_container.add(pid)
        # 行人ID映射为连续的label
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            img = []
            jpg_name = img_path_RGB.split('/')[-1]
            # 根据 RGB 图像名，找到对应的 NI、TI 图像
            img_path_NI = osp.join(dir_path, 'NI', jpg_name)
            img_path_TI = osp.join(dir_path, 'TI', jpg_name)
            img.append(img_path_RGB)
            img.append(img_path_NI)
            img.append(img_path_TI)

            pid = int(jpg_name.split('_')[0][0:6])      # 行人ID
            camid = int(jpg_name.split('_')[1][3])      # 摄像头ID
            trackid = -1                                # 轨迹ID，当前没有标注，设为 -1
            camid -= 1                                  # 摄像头ID从0开始
            if relabel:                                 # 如果是训练集，行人ID重新映射
                pid = pid2label[pid]
            data.append((img, pid, camid, trackid))     # 保存一条样本数据

        return data
