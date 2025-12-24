"""
数据集基类与图像读取工具（中文说明）

组成：
- read_image: 稳健的图像读取函数，支持单路径切片为 RGB/NI/TI 或多路径列表
- BaseDataset / BaseImageDataset: ReID 数据集统计与打印工具基类
- ImageDataset: PyTorch Dataset 封装，用于 DataLoader 迭代
- tokenize: 文本tokenization函数（从IDEA项目复制）
- EnhancedImageDataset: 支持文本特征的增强版ImageDataset

要点：
- 通过 PIL 读取图像，并在单图模式下按列裁切为 3 块多模态图
- 统计信息包括 id 数、图像数、相机数、视角/轨迹数
- 支持CLIP风格的文本tokenization
"""
from PIL import Image, ImageFile   # PIL 用于图像处理
from torch.utils.data import Dataset  # PyTorch 提供的 Dataset 基类，用于构建数据集
import os.path as osp               # 用于处理文件路径
import torch                        # 用于tokenize函数
from modeling.clip.simple_tokenizer import SimpleTokenizer  # CLIP tokenizer

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
                    
                    # 🔥 关键修改：处理RGBNT100双模态数据
                    # 如果所有路径都相同（RGBNT100情况），说明是双模态数据集
                    if len(set(img_list)) == 1:  # 所有路径相同
                        # RGBNT100：RGB-IR双模态，需要从单张图中提取RGB和IR
                        # 假设图像是水平拼接的：左边RGB，右边IR
                        width, height = img.size
                        if width >= 256:  # 确保图像足够宽
                            RGB = img.crop((0, 0, width//2, height))  # 左半部分作为RGB
                            IR = img.crop((width//2, 0, width, height))  # 右半部分作为IR
                            # 为了兼容三模态模型，创建虚拟的TI（使用IR图像）
                            img3 = [RGB, IR, IR]  # RGB, IR, 虚拟TI
                        else:
                            # 如果图像不够宽，直接使用原图作为RGB，IR和TI都使用原图
                            img3 = [img, img, img]
                    else:
                        # RGBNT201：三模态数据集，直接使用三个路径
                        img3.append(img)    # 不切割，直接加入列表
                    got_img = True
                except IOError:
                    print(f"IOError incurred when reading '{img_path}'. Will redo. Don't worry. Just chill.")
                    pass
    return img3  # 返回图像列表


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    """
    将文本caption进行tokenization处理，返回固定长度的token张量。

    Args:
        caption (str): 输入的文本描述
        tokenizer: tokenizer对象（如SimpleTokenizer）
        text_length (int): 输出token序列的固定长度，默认77
        truncate (bool): 是否截断过长的文本，默认True

    Returns:
        torch.LongTensor: 形状为[text_length]的token张量
    """
    # 获取特殊token
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    # 对文本进行编码并添加特殊token
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    # 创建固定长度的结果张量
    result = torch.zeros(text_length, dtype=torch.long)

    if len(tokens) > text_length:
        if truncate:
            # 截断并确保结束token在最后
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )

    # 将tokens复制到结果张量
    result[:len(tokens)] = torch.tensor(tokens)
    return result


# *************** 数据集基类 ***************
class BaseDataset(object):
    """
    ReID（行人重识别）数据集的基类
    """

    def get_imagedata_info(self, data):
        # 提取数据集中所有 pid（行人 ID）、camid（摄像头 ID）、trackid（轨迹 ID）

        # 支持包含文本的数据格式（img, pid, camid, trackid, text_rgb, text_nir, text_tir）
        pids, cams, tracks = [], [], []
        for item in data:
            # 只取前4个元素，忽略可能的文本数据
            _, pid, camid, trackid = item[:4]
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
# EnhancedImageDataset：增强版ImageDataset，支持文本特征融合


class EnhancedImageDataset(Dataset):
    """
    增强版ImageDataset，支持文本特征融合

    返回格式：
    - 启用文本特征时：(img, pid, camid, trackid, img_path, text_features)
    - 禁用文本特征时：(img, pid, camid, trackid, img_path)
    """

    def __init__(self, dataset, transform=None, text_loader=None, use_text_features=False):
        """
        初始化增强版图像数据集

        Args:
            dataset: 基础数据集（包含图像路径和标签）
            transform: 图像变换
            text_loader: 文本特征加载器
            use_text_features: 是否使用文本特征
        """
        self.dataset = dataset
        self.transform = transform
        self.text_loader = text_loader
        self.use_text_features = use_text_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]

        # 读取图像
        img3 = read_image(img_path)

        # 应用图像变换
        if self.transform is not None:
            img = [self.transform(img) for img in img3]

        # 获取图像文件名（用于查找文本特征）
        if type(img_path) == type("This is a str"):
            img_filename = img_path.split('/')[-1]
        else:
            img_filename = img_path[0].split('/')[-1]

        # 如果启用文本特征，获取对应的文本特征
        if self.use_text_features and self.text_loader is not None:
            try:
                # 获取三种模态的文本特征
                rgb_text = self.text_loader.get_text_feature(img_filename, 'RGB')
                nir_text = self.text_loader.get_text_feature(img_filename, 'NIR')
                tir_text = self.text_loader.get_text_feature(img_filename, 'TIR')

                text_features = {
                    'rgb_text': rgb_text,
                    'nir_text': nir_text,
                    'tir_text': tir_text
                }

                return img, pid, camid, trackid, img_filename, text_features

            except Exception as e:
                print(f"⚠️  获取文本特征失败 {img_filename}: {e}")
                print("   将使用零向量作为默认值")

                # 返回零向量作为默认值
                zero_text = torch.zeros(512, dtype=torch.float32)
                text_features = {
                    'rgb_text': zero_text,
                    'nir_text': zero_text,
                    'tir_text': zero_text
                }

                return img, pid, camid, trackid, img_filename, text_features
        else:
            # 不使用文本特征，返回标准格式
            return img, pid, camid, trackid, img_filename


class IDEATextImageDataset(Dataset):
    """
    IDEA风格的图像数据集类 - 完全复制IDEA项目的实现

    处理包含文本的数据集，返回经过tokenize的文本特征。
    输入：7元组 (img_path, pid, camid, trackid, r_text, n_text, t_text)
    输出：8元组 (img, pid, camid, trackid, img_filename, r_tokens, n_tokens, t_tokens)
    """

    def __init__(self, dataset, transform=None, text_length: int = 77,
                 truncate: bool = True, mask_ratio: float = 0.):
        """
        初始化IDEA风格的图像数据集

        Args:
            dataset: 数据集列表，每个元素为7元组
            transform: 图像变换
            text_length: 文本token的最大长度
            truncate: 是否截断过长的文本
            mask_ratio: 文本mask比例（预留）
        """
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 解包7元组数据：图像路径 + 行人ID + 相机ID + 轨迹ID + 三种模态的文本
        img_path, pid, camid, trackid, r_text, n_text, t_text = self.dataset[index]

        # 读取并处理图像
        img3 = read_image(img_path)

        # 对三种模态的文本进行tokenize
        r_tokens = tokenize(r_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        n_tokens = tokenize(n_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        t_tokens = tokenize(t_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        # 应用图像变换
        if self.transform is not None:
            img = [self.transform(img) for img in img3]

        # 返回图像和所有信息
        if type(img_path) == type("This is a str"):
            return img, pid, camid, trackid, img_path.split('/')[-1], r_tokens, n_tokens, t_tokens
        else:
            return img, pid, camid, trackid, img_path[0].split('/')[-1], r_tokens, n_tokens, t_tokens


# read_image：负责安全地读取图像，支持切割或多图输入
# BaseDataset / BaseImageDataset：提供数据统计功能
# ImageDataset：继承 PyTorch Dataset，封装图像读取、transform 转换，供 DataLoader 迭代训练
# EnhancedImageDataset：增强版ImageDataset，支持文本特征融合
# IDEATextImageDataset：IDEA风格的文本图像数据集，完全复制IDEA实现