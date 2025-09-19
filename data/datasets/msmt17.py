import os.path as osp
from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17 数据集
    参考:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: http://www.pkuvmc.com/publications/msmt17.html

    数据集统计:
    - identities: 4101
    - images: 32621 (train) + 11659 (query) + 82161 (gallery)
    - cameras: 15
    """
    dataset_dir = 'MSMT17'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)

        # 子目录 & 列表文件路径
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        # 1. 检查目录完整性
        self._check_before_run()

        # 2. 处理训练、验证、查询、检索集
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val  # 训练集 = 训练 + 验证
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)

        # 3. 打印统计信息
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        # 4. 保存到类属性
        self.train = train
        self.query = query
        self.gallery = gallery

        # 5. 统计 PID / 图像数 / 摄像头数
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """检查路径是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))


    def _process_dir(self, dir_path, list_path):
        # 读取 list_xxx.txt，里面每行：img_path pid
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        dataset = []
        pid_container = set()
        cam_container = set()

        # 遍历每行，处理图像路径、pid、摄像头id
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)
            camid = int(img_path.split('_')[2])  # 文件名第3段是摄像头id
            img_path = osp.join(dir_path, img_path)

            # 存成 (图像路径, pid, 摄像头id, 轨迹id)
            dataset.append((img_path, self.pid_begin + pid, camid-1, 1))
            pid_container.add(pid)
            cam_container.add(camid)

        print(cam_container, 'cam_container')  # 打印摄像头id集合

        # 确认 pid 从 0 开始连续编号
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "pid 必须从 0 开始连续编号"
        return dataset
