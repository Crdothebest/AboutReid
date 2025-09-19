"""
数据集基类与图像读取工具（中文说明）

组成：
- read_image: 稳健的图像读取函数，支持单路径切片为 RGB/NI/TI 或多路径列表
- BaseDataset / BaseImageDataset: ReID 数据集统计与打印工具基类
- ImageDataset: PyTorch Dataset 封装，用于 DataLoader 迭代

要点：
- 通过 PIL 读取图像，并在单图模式下按列裁切为 3 块多模态图
- 统计信息包括 id 数、图像数、相机数、视角/轨迹数
"""
from PIL import Image, ImageFile   # PIL 用于图像处理
from torch.utils.data import Dataset  # PyTorch 提供的 Dataset 基类，用于构建数据集
import os.path as osp               # 用于处理文件路径

ImageFile.LOAD_TRUNCATED_IMAGES = True
# 允许加载不完整的图像，防止图像损坏导致读取错误


# ***************** 用于读取图像的函数 ************************
# 作用总结：
# 1. 输入可以是单张图路径或路径列表
# 2. 单张图会被切成 3 块，多个图会直接按原样存入列表
# 3. 如果读取失败会一直重试，避免 IO 错误

def read_image(img_list):
    """持续尝试读取图像，直到成功为止，避免 IO 过程中的错误"""
    if type(img_list) == type("This is a str"):   # 判断输入是否为单个路径（字符串）
        img_path = img_list # 单张图路径
        got_img = False # 是否成功读取图像
        if not osp.exists(img_path):              # 检查路径是否存在
            raise IOError("{} does not exist".format(img_path))
        while not got_img: # 如果未成功读取图像             
            try:
                img = Image.open(img_path).convert('RGB')  # 打开并转为 RGB
                # 将图像切割成三部分：RGB、NI、TI
                RGB = img.crop((0, 0, 256, 128)) # 切割RGB图像
                # 思考是什么：RGB图像的宽度和高度是256和128
                NI = img.crop((256, 0, 512, 128)) # 切割NI图像
                TI = img.crop((512, 0, 768, 128)) # 切割TI图像
                img3 = [RGB, NI, TI] # 将RGB、NI、TI图像拼接在一起
                got_img = True
            except IOError:   # 读取出错时重试
                print(f"IOError incurred when reading '{img_path}'. Will redo. Don't worry. Just chill.") # 打印错误信息                    
                pass
    else:
        img3 = []
        for i in img_list:   # 多个路径依次处理
            img_path = i # 单张图路径       
            got_img = False # 是否成功读取图像
            if not osp.exists(img_path):
                raise IOError("{} does not exist".format(img_path))
            while not got_img: # 如果未成功读取图像             
                try:
                    img = Image.open(img_path).convert('RGB')
                    img3.append(img)    # 不切割，直接加入列表
                    got_img = True
                except IOError:
                    print(f"IOError incurred when reading '{img_path}'. Will redo. Don't worry. Just chill.")
                    pass
    return img3  # 返回图像列表


# *************** 数据集基类 ***************
class BaseDataset(object):
    """
    ReID（行人重识别）数据集的基类
    """

    def get_imagedata_info(self, data):
        # 提取数据集中所有 pid（行人 ID）、camid（摄像头 ID）、trackid（轨迹 ID）

        # 还数据集在这里改 ； 按新的内容格式加载
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]

        # 转为集合去重
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)

        # 统计数量
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)

        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        # 需要子类实现
        raise NotImplementedError


# ******************* 图像数据集基类 **********************
class BaseImageDataset(BaseDataset):
    """
    图像 ReID 数据集基类，增加统计功能
    """

    def print_dataset_statistics(self, train, query, gallery):
        # 分别统计训练集、查询集、检索集（被检索的）的信息   
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        # 打印统计结果
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

# ************ 图像数据集类 ********************
class ImageDataset(Dataset):  # 初始化 长度统计
    def __init__(self, dataset, transform=None):
        self.dataset = dataset       # 数据集（list，每个元素包含路径、pid、camid、trackid）
        self.transform = transform   # 图像变换（如 ToTensor, Normalize）

    def __len__(self):
        return len(self.dataset)      # 返回数据集大小

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img3 = read_image(img_path)   # 读取图像（可能是多块）

        # 如果有 transform，应用在每张图上
        if self.transform is not None:
            img = [self.transform(img) for img in img3]

        # 返回图像和相关信息，单图用路径字符串，列表用第一个路径
        if type(img_path) == type("This is a str"):
            return img, pid, camid, trackid, img_path.split('/')[-1]
        else:
            return img, pid, camid, trackid, img_path[0].split('/')[-1]

# read_image：负责安全地读取图像，支持切割或多图输入
# BaseDataset / BaseImageDataset：提供数据统计功能
# ImageDataset：继承 PyTorch Dataset，封装图像读取、transform 转换，供 DataLoader 迭代训练