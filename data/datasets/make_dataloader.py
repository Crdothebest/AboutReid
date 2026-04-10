import os
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from .bases import ImageDataset, EnhancedImageDataset, read_image
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .msvr310 import MSVR310
from .RGBNT201 import RGBNT201
from .RGBNT100 import RGBNT100
from .sampler_ddp import RandomIdentitySampler_DDP
from .qwen_vl_loader import get_text_loader, QwenVLTextLoader  # 新增文本特征加载器
import torch.distributed as dist
def _tokenize_texts(texts, context_length=77):
    from modeling.clip.simple_tokenizer import SimpleTokenizer
    import torch as _torch
    tokenizer = SimpleTokenizer()
    result = []
    for text in texts:
        tokens = tokenizer.encode(text or '')[:context_length - 2]
        tokens = [49406] + tokens + [49407]
        tokens = tokens + [0] * (context_length - len(tokens))
        result.append(tokens)
    return _torch.tensor(result, dtype=_torch.long)


# 数据集字典：存数据集的名称 需要加载新的就往这里写
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'RGBNT201': RGBNT201,
    'RGBNT100': RGBNT100,
    'MSVR310': MSVR310
}

""" Random Erasing (Cutout)

Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2019, Ross Wightman

Random Erasing 是一种数据增强技术，它通过随机擦除图像中的一部分区域来增强模型的鲁棒性。
"""
import random
import math

import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5,
            min_area=0.02,
            max_area=1 / 3,
            min_aspect=0.3,
            max_aspect=None,
            mode='const',
            min_count=1,
            max_count=None,
            num_splits=0,
            device='cuda',
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

# 该方法实现了实际的擦除操作。在每次调用时，方法会尝试随机选择一个矩形区域，然后擦除该区域的像素。
    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, viewids,_


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, camids_batch, viewids, img_paths


def val_collate_fn_idea_style(batch):
    """
    IDEA风格验证数据的collate函数 - 处理IDEATextImageDataset的8元素数据

    IDEATextImageDataset返回格式:
    (img, pid, camid, trackid, img_filename, r_tokens, n_tokens, t_tokens)

    输出格式:
    (imgs, pids, camids, camids_batch, viewids, img_paths, text_features)
    兼容processor.py的7元素解析逻辑
    """
    # 解包8元素数据
    imgs, pids, camids, trackids, img_filenames, r_tokens, n_tokens, t_tokens = zip(*batch)

    # 处理图像数据（已经是变换后的张量列表）
    RGB_list, NI_list, TI_list = [], [], []
    for img in imgs:
        RGB_list.append(img[0])  # RGB图像
        NI_list.append(img[1])   # NI图像
        TI_list.append(img[2])   # TI图像

    # 堆叠成批次张量
    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, 'NI': NI, 'TI': TI}

    # 处理标签数据
    pids = torch.tensor(pids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(trackids, dtype=torch.int64)  # 使用trackid作为viewid

    # 组织文本特征
    text_features = {
        'RGB': torch.stack(r_tokens, dim=0),
        'NIR': torch.stack(n_tokens, dim=0),
        'TIR': torch.stack(t_tokens, dim=0)
    }

    # 返回7元素格式：imgs, pids, camids, camids_batch, viewids, img_paths, text_features
    return imgs, pids, camids_batch, camids_batch, viewids, img_filenames, text_features


# ============ 增强版collate函数 (支持文本特征开关) ============

def train_collate_fn_with_text(batch):
    """
    增强版训练collate函数 - 支持文本特征

    当use_text_features=True时，batch中的每个样本包含：
    (img_list, pid, camid, viewid, img_path, text_features)

    当use_text_features=False时，batch中的每个样本为：
    (img_list, pid, camid, viewid, img_path)
    """
    # 检查batch中是否包含文本特征
    sample = batch[0]
    has_text = len(sample) > 4  # 5个元素表示包含文本特征

    if has_text:
        imgs, pids, camids, viewids, img_paths, text_features = zip(*batch)
        # 解包文本特征
        rgb_texts, nir_texts, tir_texts = [], [], []
        for text_dict in text_features:
            rgb_texts.append(text_dict['rgb_text'])
            nir_texts.append(text_dict['nir_text'])
            tir_texts.append(text_dict['tir_text'])

        rgb_texts = torch.stack(rgb_texts, dim=0)
        nir_texts = torch.stack(nir_texts, dim=0)
        tir_texts = torch.stack(tir_texts, dim=0)
        text_features = {'RGB': rgb_texts, 'NIR': nir_texts, 'TIR': tir_texts}
    else:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        text_features = None

    # 处理基本数据
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    # 处理图像数据
    RGB_list, NI_list, TI_list = [], [], []
    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}

    return imgs, pids, camids, viewids, img_paths, text_features


def train_collate_fn_idea_style(batch):
    """
    IDEA风格数据集的collate函数 - 完全复制IDEA的文本处理逻辑
    batch中每个样本格式: (img, pid, camid, trackid, _, r_text, n_text, t_text)
    """
    imgs, pids, camids, trackids, _, r_text, n_text, t_text = zip(*batch)

    # 处理图像数据 - 复制IDEA的图像处理逻辑
    RGB_list, NI_list, TI_list = [], [], []
    for img in imgs:
        RGB_list.append(img[0])  # RGB 图像
        NI_list.append(img[1])   # NIR 图像
        TI_list.append(img[2])   # TIR 图像

    # 将图像列表堆叠成张量
    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)

    # 组织成字典格式
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}

    # 将文本特征组织成字典（key与make_model.py一致：'RGB'/'NIR'/'TIR'）
    text = {'RGB': torch.stack(r_text),
            'NIR': torch.stack(n_text),
            'TIR': torch.stack(t_text)}

    # 将标签转换为张量
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    trackids = torch.tensor(trackids, dtype=torch.int64)

    return imgs, pids, camids, trackids, text

def val_collate_fn_with_text(batch):
    """
    增强版验证collate函数 - 支持文本特征
    """
    # 检查batch中是否包含文本特征
    sample = batch[0]
    has_text = len(sample) > 5  # 6个元素表示包含文本特征

    if has_text:
        imgs, pids, camids, viewids, img_paths, text_features = zip(*batch)
        # 解包文本特征
        rgb_texts, nir_texts, tir_texts = [], [], []
        for text_dict in text_features:
            rgb_texts.append(text_dict['rgb_text'])
            nir_texts.append(text_dict['nir_text'])
            tir_texts.append(text_dict['tir_text'])

        rgb_texts = torch.stack(rgb_texts, dim=0)
        nir_texts = torch.stack(nir_texts, dim=0)
        tir_texts = torch.stack(tir_texts, dim=0)
        text_features = {'RGB': rgb_texts, 'NIR': nir_texts, 'TIR': tir_texts}
    else:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        text_features = None

    # 处理基本数据
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)

    # 处理图像数据
    RGB_list, NI_list, TI_list = [], [], []
    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}

    return imgs, pids, camids, camids_batch, viewids, img_paths, text_features


# 该函数根据配置文件 cfg 构建训练和验证数据加载器（dataloader）
def make_dataloader(cfg):
    import os
    # Initialize collate_fn variables with module-level defaults to avoid UnboundLocalError
    # when dataset-specific branches don't execute (e.g., non-RGBNT201 datasets)
    import sys as _sys
    _m = _sys.modules[__name__]
    _train_collate_fn_default = getattr(_m, 'train_collate_fn')
    _val_collate_fn_default = getattr(_m, 'val_collate_fn')
    train_collate_fn = _train_collate_fn_default
    val_collate_fn = _val_collate_fn_default
    re_prob = getattr(cfg.INPUT, 'RE_PROB', 0.5)

    # ============ IDEA风格离线预编码（完全按照IDEA项目的方式） ============
    dataset_name = getattr(cfg.DATASETS, 'NAMES', 'RGBNT201')
    use_idea_style_dataset = True  # 强制使用IDEA风格数据集

    # 获取文本特征开关
    use_text_features = getattr(cfg.DATASETS, 'USE_TEXT_FEATURES', True)

    # 动态构建文本特征路径：根据数据集名称自动选择
    if use_text_features:
        # 智能路径构建逻辑
        configured_path = getattr(cfg.DATASETS, 'QWEN_VL_ANNO_DIR', None)
        default_paths = [None, "./QwenVL_Anno"]

        if configured_path in default_paths:
            # 使用默认路径，自动构建完整路径
            qwen_vl_anno_dir = f"data/datasets/QwenVL_Anno/{dataset_name}/text"
            print(f"📁 自动构建文本特征路径: {qwen_vl_anno_dir}")
        else:
            # 使用配置文件指定的路径
            configured_full_path = os.path.join(configured_path, dataset_name, "text")

            # 检查完整路径是否存在
            if os.path.exists(configured_full_path):
                qwen_vl_anno_dir = configured_full_path
                print(f"📁 使用完整配置路径: {qwen_vl_anno_dir}")
            elif os.path.exists(os.path.join(configured_path, dataset_name)):
                # 数据集目录存在，添加text子目录
                qwen_vl_anno_dir = os.path.join(configured_path, dataset_name, "text")
                print(f"📁 自动补全路径: {qwen_vl_anno_dir}")
            else:
                # 使用配置的路径作为基础目录
                qwen_vl_anno_dir = os.path.join(configured_path, dataset_name, "text")
                print(f"📁 使用配置基础路径构建: {qwen_vl_anno_dir}")
    else:
        qwen_vl_anno_dir = getattr(cfg.DATASETS, 'QWEN_VL_ANNO_DIR', './QwenVL_Anno')

    # 初始化IDEA风格文本加载器（完全按照IDEA项目的方式）
    idea_style_text_loader = None

    if dataset_name == 'RGBNT201' and use_text_features:
        # IDEA风格：使用离线预编码的文本加载器
        from .qwen_vl_loader import get_text_loader
        clip_model_name = getattr(cfg.MODEL, 'TRANSFORMER_TYPE', 'ViT-B-16').split('_')[-1]
        idea_style_text_loader = get_text_loader(
            anno_dir=qwen_vl_anno_dir,
            use_clip=False,  # IDEA风格：强制使用预编码模式
            clip_model_name=clip_model_name,
            cfg=cfg
        )
        print("✅ 已启用IDEA风格离线预编码文本加载器（完全复制IDEA方式）")

    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=re_prob, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # Handle both tuple and string formats for DATASETS.NAMES
    dataset_name = cfg.DATASETS.NAMES
    if isinstance(dataset_name, tuple):
        dataset_name = dataset_name[0]
    
    # Handle both tuple and string formats for DATASETS.ROOT_DIR
    root_dir = cfg.DATASETS.ROOT_DIR
    if isinstance(root_dir, tuple):
        root_dir = root_dir[0]
        
    
    # Convert relative path to absolute path to avoid path issues
    if not os.path.isabs(root_dir):
        root_dir = os.path.abspath(root_dir)

    # 支持多种数据集模式
    if dataset_name == 'RGBNT201_IDEA':
        # 使用完全复制IDEA项目的RGBNT201_IDEA_Text数据集
        from .RGBNT201_IDEA_Text import RGBNT201_IDEA_Text
        dataset = RGBNT201_IDEA_Text(root=root_dir, cfg=cfg)
        print("✅ 使用完全复制IDEA项目的RGBNT201_IDEA_Text数据集")
    elif dataset_name == 'RGBNT201':
        from .RGBNT201 import RGBNT201
        dataset = RGBNT201(root=root_dir, cfg=cfg)
        print("✅ 使用AboutReid风格的RGBNT201数据集")
    else:
        # 对于其他数据集，保持原有逻辑
        dataset = __factory[dataset_name](root=root_dir)
        print("✅ 使用标准数据集（非RGBNT201）")

    # 根据数据集类型选择不同的处理方式
    if dataset_name == 'RGBNT201_IDEA':
        # 使用IDEA风格数据集 - 完全复制IDEA的处理方式
        from .bases import IDEATextImageDataset

        train_set = IDEATextImageDataset(dataset.train, train_transforms)
        train_set_normal = IDEATextImageDataset(dataset.train, val_transforms)

        # 创建IDEA风格的collate函数 - 完全复制IDEA的文本处理
        def train_collate_fn_idea_style(batch):
            """
            IDEA风格数据集的collate函数 - 完全复制IDEA的文本处理逻辑
            batch中每个样本格式: (img, pid, camid, trackid, img_filename, r_tokens, n_tokens, t_tokens)
            """
            imgs, pids, camids, trackids, img_filenames, r_tokens, n_tokens, t_tokens = zip(*batch)

            # 处理图像数据 - 复制IDEA的图像处理逻辑
            RGB_list, NI_list, TI_list = [], [], []
            for img in imgs:
                RGB_list.append(img[0])
                NI_list.append(img[1])
                TI_list.append(img[2])

            RGB = torch.stack(RGB_list, dim=0)
            NI = torch.stack(NI_list, dim=0)
            TI = torch.stack(TI_list, dim=0)
            imgs = {'RGB': RGB, "NI": NI, "TI": TI}

            # 处理基本数据 - 复制IDEA的数据处理逻辑
            pids = torch.tensor(pids, dtype=torch.int64)
            camids = torch.tensor(camids, dtype=torch.int64)
            trackids = torch.tensor(trackids, dtype=torch.int64)

            # 处理tokenized文本 - 直接使用token tensors
            text_tokens = {
                'rgb_text': torch.stack(r_tokens),
                'ni_text': torch.stack(n_tokens),
                'ti_text': torch.stack(t_tokens)
            }

            return imgs, pids, camids, trackids, text_tokens

        train_collate_fn = train_collate_fn_idea_style
        val_collate_fn = train_collate_fn_idea_style

        print("✅ 使用完全复制IDEA项目的collate函数（预处理文本）")

    elif dataset_name == 'RGBNT201':
        # 创建自定义的数据集包装类，支持动态文本功能
        class RGBNT201DatasetWrapper(Dataset):
            """RGBNT201数据集包装器，支持预编码文本特征"""

            def __init__(self, dataset, transform=None, use_text_features=False, feat_dir=None):
                self.dataset = dataset
                self.transform = transform
                self.use_text_features = use_text_features
                # 加载预编码的 512 维文本向量
                self._feat = {}
                if use_text_features and feat_dir and os.path.isdir(feat_dir):
                    import glob
                    for pt_file in glob.glob(os.path.join(feat_dir, "*_feat.pt")):
                        name = os.path.basename(pt_file)          # e.g. train_RGB_feat.pt
                        key = name.replace("_feat.pt", "")        # e.g. train_RGB
                        self._feat[key] = torch.load(pt_file, map_location="cpu")
                    print(f"✅ 已加载预编码文本特征: {list(self._feat.keys())}")
                elif use_text_features:
                    print("⚠️  未找到预编码文本特征目录，文本功能将降级为 token 模式")

            def __len__(self):
                return len(self.dataset)

            def _get_feat(self, split_modal, jpg_name):
                """根据 split(train/test) + modal(RGB/NI/TI) + 文件名 返回 512 维向量"""
                key = f"{split_modal}"   # e.g. "train_RGB"
                if key in self._feat and jpg_name in self._feat[key]:
                    return self._feat[key][jpg_name]   # tensor [512]
                return None

            def __getitem__(self, index):
                data = self.dataset[index]

                if self.use_text_features:
                    img_paths, pid, camid, trackid, text_rgb, text_nir, text_tir = data
                    jpg_name = os.path.basename(img_paths[0])

                    # 优先用预编码向量，降级用字符串
                    # split 由 dataset 决定（train 或 test）
                    for split in ("train", "test"):
                        rgb_feat = self._get_feat(f"{split}_RGB", jpg_name)
                        if rgb_feat is not None:
                            nir_feat = self._get_feat(f"{split}_NI", jpg_name)
                            tir_feat = self._get_feat(f"{split}_TI", jpg_name)
                            text_rgb = rgb_feat if rgb_feat is not None else text_rgb
                            text_nir = nir_feat if nir_feat is not None else text_nir
                            text_tir = tir_feat if tir_feat is not None else text_tir
                            break

                    img3 = read_image(img_paths)
                    if self.transform is not None:
                        img = [self.transform(img) for img in img3]
                    return img, pid, camid, trackid, text_rgb, text_nir, text_tir
                else:
                    img_paths, pid, camid, trackid, _, _, _ = data
                    img3 = read_image(img_paths)
                    if self.transform is not None:
                        img = [self.transform(img) for img in img3]
                    return img, pid, camid, trackid

        if use_text_features:
            # ✅ 启用文本功能：使用包装器 + 文本collate函数
            _feat_dir = os.path.join(root_dir, 'datasets', 'RGBNT201', 'text')
            train_set = RGBNT201DatasetWrapper(dataset.train, train_transforms, use_text_features=True, feat_dir=_feat_dir)
            train_set_normal = RGBNT201DatasetWrapper(dataset.train, val_transforms, use_text_features=True, feat_dir=_feat_dir)

            # 创建包含文本的collate函数
            def train_collate_fn_with_text(batch):
                """
                包含文本特征的collate函数
                batch格式: (img, pid, camid, trackid, text_rgb, text_nir, text_tir)
                """
                imgs, pids, camids, trackids, texts_rgb, texts_nir, texts_tir = zip(*batch)

                # 处理图像数据
                RGB_list, NI_list, TI_list = [], [], []
                for img in imgs:
                    RGB_list.append(img[0])
                    NI_list.append(img[1])
                    TI_list.append(img[2])

                RGB = torch.stack(RGB_list, dim=0)
                NI = torch.stack(NI_list, dim=0)
                TI = torch.stack(TI_list, dim=0)
                imgs = {'RGB': RGB, "NI": NI, "TI": TI}

                # 处理基本数据
                pids = torch.tensor(pids, dtype=torch.int64)
                camids = torch.tensor(camids, dtype=torch.int64)
                trackids = torch.tensor(trackids, dtype=torch.int64)

                # 处理文本特征 - 使用CLIP编码或预编码
                # 文本已在 __getitem__ 中以预编码 [512] 向量返回；降级时为字符串
                def _stack_or_tokenize(texts):
                    if isinstance(texts[0], torch.Tensor):
                        return torch.stack(texts, dim=0)   # [B, 512] 预编码向量
                    return _tokenize_texts(list(texts))    # [B, 77] token 降级

                text_features = {
                    'RGB': _stack_or_tokenize(texts_rgb),
                    'NIR': _stack_or_tokenize(texts_nir),
                    'TIR': _stack_or_tokenize(texts_tir),
                }

                return imgs, pids, camids, trackids, text_features

            train_collate_fn = train_collate_fn_with_text
            val_collate_fn = train_collate_fn_with_text

            print("✅ RGBNT201数据集启用文本功能（动态包装器 + 文本编码）")

        else:
            # ❌ 禁用文本功能：使用包装器 + 普通collate函数
            train_set = RGBNT201DatasetWrapper(dataset.train, train_transforms, use_text_features=False)
            train_set_normal = RGBNT201DatasetWrapper(dataset.train, val_transforms, use_text_features=False)

            # 创建不包含文本的collate函数
            def train_collate_fn_no_text(batch):
                """
                不包含文本特征的collate函数
                batch格式: (img, pid, camid, trackid)
                """
                imgs, pids, camids, trackids = zip(*batch)

                # 处理图像数据
                RGB_list, NI_list, TI_list = [], [], []
                for img in imgs:
                    RGB_list.append(img[0])
                    NI_list.append(img[1])
                    TI_list.append(img[2])

                RGB = torch.stack(RGB_list, dim=0)
                NI = torch.stack(NI_list, dim=0)
                TI = torch.stack(TI_list, dim=0)
                imgs = {'RGB': RGB, "NI": NI, "TI": TI}

                # 处理基本数据
                pids = torch.tensor(pids, dtype=torch.int64)
                camids = torch.tensor(camids, dtype=torch.int64)
                trackids = torch.tensor(trackids, dtype=torch.int64)

                # 不返回文本特征
                # 返回格式：img, vid, target_cam, target_view, _
                return imgs, pids, camids, trackids, None

            train_collate_fn = train_collate_fn_no_text
            val_collate_fn = train_collate_fn_no_text

            print("✅ RGBNT201数据集禁用文本功能（动态包装器 + 无文本）")

    else:
        # 对于其他数据集，使用标准方式
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        print("✅ 使用标准数据集（非RGBNT201）")
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    # ============ 选择collate函数（根据数据集类型和文本特征开关） ============
    if dataset_name == 'RGBNT201_IDEA':
        # RGBNT201_IDEA数据集已经设置了IDEA风格的collate函数，保持不变
        print("✅ RGBNT201_IDEA数据集使用IDEA风格collate函数（已设置）")
    elif use_text_features:
        train_collate_fn = train_collate_fn_with_text
        val_collate_fn = val_collate_fn_with_text
        print("✅ 使用增强版collate函数（支持文本特征）")
    else:
        # 使用原始collate函数
        pass  # uses defaults set at function start
        print("✅ 使用原始collate函数（无文本特征）")

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH,
                                                     cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            if dataset_name == 'RGBNT201_IDEA':
                # 对于RGBNT201_IDEA，使用shuffle而不是RandomIdentitySampler
                train_loader = DataLoader(
                    train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn,
                )
            else:
                train_loader = DataLoader(
                    train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                    num_workers=num_workers, collate_fn=train_collate_fn,
                )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Only use for grad-cam when fixed samples need for different modalities
    # train_loader = DataLoader(
    #     train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
    #     collate_fn=train_collate_fn
    # )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 验证集：对于ReID任务，验证集需要query和gallery，所以使用测试集（query + gallery）
    # 注意：从训练集中划分的验证集不能用于ReID评估（因为需要query-gallery配对）
    # 训练集中划分的验证集可以用于其他监控（如loss），但mAP评估必须使用测试集
    # 因此，val_loader始终使用测试集（query + gallery）进行mAP评估

    # 处理验证数据集（强制使用IDEA风格）
    if dataset_name == 'RGBNT201_IDEA':
        # IDEA风格数据集 - 使用包含文本特征的验证数据集
        from .bases import IDEATextImageDataset
        # 合并query和gallery数据
        val_data = dataset.query + dataset.gallery
        val_set = IDEATextImageDataset(val_data, val_transforms)
        print("✅ 使用IDEA风格验证数据集（包含文本特征）")
        # 使用专门的collate函数处理8元素数据
        val_collate_fn_to_use = val_collate_fn_idea_style
    elif dataset_name == 'RGBNT201':
        # RGBNT201数据集 - 需要特殊处理，支持文本和非文本模式
        val_data = dataset.query + dataset.gallery
        _val_feat_dir = os.path.join(root_dir, 'datasets', 'RGBNT201', 'text') if use_text_features else None
        val_set = RGBNT201DatasetWrapper(val_data, val_transforms, use_text_features=use_text_features, feat_dir=_val_feat_dir)
        print(f"✅ 使用RGBNT201验证数据集（{'包含' if use_text_features else '不包含'}文本特征）")
        # 为RGBNT201创建专门的验证collate函数
        def val_collate_fn_rgbnt201(batch):
            """
            RGBNT201验证数据集的collate函数
            batch格式: (img, pid, camid, trackid) 或 (img, pid, camid, trackid, text_rgb, text_nir, text_tir)
            返回格式: (imgs, pids, camids, camids_batch, viewids, img_paths, text_features)
            """
            if len(batch) == 0:
                raise ValueError("Empty batch")

            # 检查第一个样本的格式来确定是否有文本
            sample = batch[0]
            has_text = len(sample) == 7  # (img, pid, camid, trackid, text_rgb, text_nir, text_tir)

            if has_text or use_text_features:
                # 启用文本：(img, pid, camid, trackid, text_rgb, text_nir, text_tir)
                try:
                    imgs, pids, camids, trackids, text_rgbs, text_nirs, text_tirs = zip(*batch)
                    # 处理文本特征
                    def _stack_val(items):
                        if items and isinstance(items[0], __import__('torch').Tensor):
                            return __import__('torch').stack(list(items), dim=0)
                        return list(items)
                    text_features = {
                        'RGB': _stack_val(text_rgbs),
                        'NIR': _stack_val(text_nirs),
                        'TIR': _stack_val(text_tirs)
                    }
                except ValueError:
                    # 如果解包失败，可能是格式不匹配
                    print(f"Batch format error. Expected 7 elements, got {len(sample)} in first sample")
                    raise
            else:
                # 禁用文本：(img, pid, camid, trackid)
                try:
                    imgs, pids, camids, trackids = zip(*batch)
                    text_features = None
                except ValueError:
                    print(f"Batch format error. Expected 4 elements, got {len(sample)} in first sample")
                    raise

            # 处理图像数据
            RGB_list, NI_list, TI_list = [], [], []
            for img in imgs:
                if len(img) >= 3:
                    RGB_list.append(img[0])
                    NI_list.append(img[1])
                    TI_list.append(img[2])
                else:
                    raise ValueError(f"Image tuple must have at least 3 elements, got {len(img)}")

            RGB = torch.stack(RGB_list, dim=0)
            NI = torch.stack(NI_list, dim=0)
            TI = torch.stack(TI_list, dim=0)
            imgs = {'RGB': RGB, "NI": NI, "TI": TI}

            # 处理基本数据，确保都不为None
            pids = [pid if pid is not None else 0 for pid in pids]
            camids = [cid if cid is not None else 0 for cid in camids]
            trackids = [tid if tid is not None else 0 for tid in trackids]

            pids = torch.tensor(pids, dtype=torch.int64)
            camids = torch.tensor(camids, dtype=torch.int64)
            trackids = torch.tensor(trackids, dtype=torch.int64)

            # 返回验证格式：(imgs, pids, camids, camids_batch, viewids, img_paths, text_features)
            result = (imgs, pids, camids, camids, trackids, None, text_features)

            # 最终检查
            if len(result) != 7:
                raise ValueError(f"Collate function must return 7 elements, got {len(result)}")
            if result[4] is None:  # target_view
                raise ValueError("target_view cannot be None")

            return result

        val_collate_fn_to_use = val_collate_fn_rgbnt201
    else:
        # 对于其他数据集，使用标准方式
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        print("✅ 使用标准验证数据集")
        val_collate_fn_to_use = val_collate_fn

    # 如果存在验证集，打印信息但不使用（因为ReID评估需要query-gallery配对）
    if hasattr(dataset, 'val') and len(dataset.val) > 0:
        print("ℹ️  验证集存在（{} 张图像），但ReID评估使用测试集（query + gallery）".format(len(dataset.val)))
        print("   验证集可用于其他监控，但mAP评估必须使用测试集")

    print("✅ 使用测试集（query + gallery）进行训练监控和mAP评估")
    print(f"   查询集: {len(dataset.query)} 张图像")
    print(f"   图库集: {len(dataset.gallery)} 张图像")

    print(f"🔧 val_loader使用collate_fn: {val_collate_fn_to_use.__name__ if hasattr(val_collate_fn_to_use, '__name__') else 'unknown'}")
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_to_use
    )
    # 根据数据集类型选择合适的collate函数
    if dataset_name == 'RGBNT201':
        train_loader_normal_collate = train_collate_fn_with_text if use_text_features else train_collate_fn_no_text
    else:
        train_loader_normal_collate = val_collate_fn

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=train_loader_normal_collate
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
