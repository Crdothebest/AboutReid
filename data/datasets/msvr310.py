# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""


import os
import os.path as osp
from .bases import BaseImageDataset


class MSVR310(BaseImageDataset):
    """
    Market1501
    参考: ICCV 2015, Market1501 论文
    数据集统计:
    # identities: 1501 (+1 背景)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'msvr310'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(MSVR310, self).__init__()
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        # 数据路径
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query3')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # 1. 检查路径
        self._check_before_run()

        # 2. 处理数据
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        # 3. 打印统计
        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(train, query, gallery)

        # 4. 保存到类属性
        self.train = train
        self.query = query
        self.gallery = gallery

        # 5. 统计信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
    # 主要区别在于：
    # 训练数据 需要 relabel，因为原始 ID 可能不连续。
    # query / gallery 保持原始 ID 不变。
    # 每个样本不仅有 RGB 图像，还包括 nir (近红外) 和 thermal (热红外) 图像。

    # 如果路径不存在直接报错，避免后续加载失败
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    # 处理目录
    def _process_dir(self, dir_path, relabel=False):
        vid_container = set()
        for vid in os.listdir(dir_path):
            vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for vid in os.listdir(dir_path):
            vid_path = osp.join(dir_path, vid)
            r_data = os.listdir(osp.join(vid_path, 'vis'))
            for img in r_data:
                r_img_path = osp.join(vid_path, 'vis', img)
                n_img_path = osp.join(vid_path, 'ni', img)
                t_img_path = osp.join(vid_path, 'th', img)
                vid = int(img[0:4])  # 前4位是vid
                camid = int(img[11])  # 第11位是摄像头id
                sceneid = int(img[6:9])  # 第6-8位是场景id
                assert 0 <= camid <= 7
                if relabel:
                    vid = vid2label[vid]
                dataset.append(((r_img_path, n_img_path, t_img_path), vid, camid, sceneid))
        return dataset
    # 关键点：
    # 每个 vid 目录下有多个模态：vis (可见光)、ni (近红外)、th (热红外)。
    # 每个样本保存三个模态路径 + vid + camid + sceneid。
    # relabel=True 时，重新给 PID 编号，保证从 0 开始连续。

    # def _process_dir2(self, dir_path, relabel=False):
    #     img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    #     pattern = re.compile(r'([-\d]+)_c([-\d]+)')

    #     pid_container = set()
    #     for img_path in img_paths:
    #         pid, _ = map(int, pattern.search(img_path).groups())
    #         if pid == -1: continue  # junk images are just ignored
    #         pid_container.add(pid)
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}

    #     dataset = []
    #     for img_path in img_paths:
    #         pid, camid = map(int, pattern.search(img_path).groups())
    #         #pdb.set_trace()
    #         #if pid == -1: continue  # junk images are just ignored
    #         assert 1 <= pid <= 600  # pid == 0 means background
    #         assert 1 <= camid <= 8
    #         camid -= 1  # index starts from 0
    #         if relabel: pid = pid2label[pid]
    #         dataset.append((img_path, pid, camid))
    #     return dataset
