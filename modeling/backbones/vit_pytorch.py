""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small resnest, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs


# From PyTorch internals
# _ntuple 是一个辅助函数，用于将输入 x 转换为大小为 n 的元组
def _ntuple(n):
    def parse(x):
        # 如果输入 x 是一个可迭代对象（如列表、元组等），直接返回它
        if isinstance(x, container_abcs.Iterable):
            return x
        # 否则将 x 重复 n 次并转换为元组
        return tuple(repeat(x, n))

    return parse


# ImageNet 数据集的均值和标准差，用于图像标准化
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 用于将输入转换为大小为 2 的元组（通常用于图像尺寸）
to_2tuple = _ntuple(2)


# 随机深度（Stochastic Depth）函数：按给定的 drop_prob 概率丢弃路径
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    这个函数实现了随机深度（Stochastic Depth）技术，通常用于残差网络中，
    按照给定的丢弃概率 `drop_prob` 来丢弃部分路径（即跳过一些层），
    从而提高训练过程中的多样性和鲁棒性。
    """
    # 如果 drop_prob 为 0 或者不在训练模式下，直接返回输入 x
    if drop_prob == 0. or not training:
        return x

    # 计算保持路径的概率
    keep_prob = 1 - drop_prob
    # 创建一个与输入 x 形状匹配的随机张量
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适应不同维度的张量（不仅限于二维卷积）
    # 生成一个随机张量，值在 [keep_prob, 1] 范围内
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 将随机张量二值化，保留路径或者丢弃路径
    random_tensor.floor_()  # floor 操作将小于 1 的数变为 0，大于等于 1 的数变为 1
    # 缩放输出，保证丢弃路径时不会影响输入的均值
    output = x.div(keep_prob) * random_tensor
    return output


# DropPath 模块类：将随机深度（Stochastic Depth）应用于网络中的残差路径
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    该模块在前向传播过程中应用随机深度技术，允许在训练过程中
    按概率丢弃部分残差连接的路径，从而减轻过拟合并提高模型的泛化能力。
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob  # 设置丢弃路径的概率

    def forward(self, x):
        # 在前向传播过程中应用 drop_path 函数
        return drop_path(x, self.drop_prob, self.training)  # 使用训练模式时才丢弃路径


# 配置函数：用于返回模型的默认配置字典
def _cfg(url='', **kwargs):
    return {
        'url': url,  # 模型的下载链接
        'num_classes': 1000,  # 类别数（通常是 ImageNet 的 1000 类）
        'input_size': (3, 224, 224),  # 输入图像的尺寸（3 通道，224x224 像素）
        'pool_size': None,  # 池化层大小（默认为 None）
        'crop_pct': .9,  # 图像裁剪的比例（默认裁剪 90% 的图像）
        'interpolation': 'bicubic',  # 图像插值方式（默认为双三次插值）
        'mean': IMAGENET_DEFAULT_MEAN,  # 图像标准化的均值（默认为 ImageNet 的均值）
        'std': IMAGENET_DEFAULT_STD,  # 图像标准化的标准差（默认为 ImageNet 的标准差）
        'first_conv': 'patch_embed.proj',  # 第一个卷积层的名称
        'classifier': 'head',  # 分类器层的名称
        **kwargs  # 可以通过关键字参数传递其他配置
    }


default_cfgs = {
    # patch resnest
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid resnest
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}

import torch
import torch.nn as nn


class Mlp(nn.Module):
    """
    MLP（多层感知机）模块，用于实现基本的全连接神经网络。
    该模块包含了两层全连接层，并支持激活函数和 Dropout 操作。

    Args:
        in_features (int): 输入特征的维度。
        hidden_features (int, optional): 隐藏层特征的维度。如果为 None，则默认设置为 in_features。
        out_features (int, optional): 输出特征的维度。如果为 None，则默认设置为 in_features。
        act_layer (nn.Module, optional): 激活函数类型，默认为 GELU 激活函数。
        drop (float, optional): Dropout 概率，默认为 0.0（即不使用 Dropout）。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        # out_features 默认为 in_features，hidden_features 默认为 in_features
        out_features = out_features or in_features  # 如果没有指定 out_features，使用 in_features
        hidden_features = hidden_features or in_features  # 如果没有指定 hidden_features，使用 in_features

        # 第一层全连接层，将输入特征映射到隐藏特征空间
        self.fc1 = nn.Linear(in_features, hidden_features)  # 创建一个全连接层 fc1，输入为 in_features，输出为 hidden_features

        # 激活函数层，默认使用 GELU 激活函数
        self.act = act_layer()  # 激活函数默认使用 GELU

        # 第二层全连接层，将隐藏特征映射到输出特征空间
        self.fc2 = nn.Linear(hidden_features, out_features)  # 创建一个全连接层 fc2，输入为 hidden_features，输出为 out_features

        # Dropout 层，用于在训练期间随机丢弃一部分神经元，防止过拟合
        self.drop = nn.Dropout(drop)  # 使用指定的丢弃概率 drop 创建 Dropout 层

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, in_features)。

        Returns:
            Tensor: 经过 MLP 处理后的输出张量，形状为 (batch_size, out_features)。
        """
        # 第一层全连接
        x = self.fc1(x)  # 输入 x 通过第一层全连接 fc1

        # 激活函数
        x = self.act(x)  # 对第一层的输出应用激活函数 act_layer（默认为 GELU）

        # Dropout 层
        x = self.drop(x)  # 对经过激活函数的输出应用 Dropout 层

        # 第二层全连接
        x = self.fc2(x)  # 输入经过第一层全连接和 Dropout 后，传入第二层全连接 fc2

        # 再次应用 Dropout
        x = self.drop(x)  # 对第二层的输出再次应用 Dropout 层（通常可以在输出层上也使用 Dropout）

        return x  # 返回最终输出

# 自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, get_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# 这部分代码实现的是一个 Transformer Block，并且集成了 适配器（Adapter） 和 提示词（Prompt） 机制，主要用于多模态学习和适配任务。这段代码的核心思想是利用不同的输入模态（RGB、NIR、TIR）通过不同的提示词和适配器，使得模型能够更好地处理多模态数据。
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        d_model = dim
        self.k = 4
        self.begin = -1
        dropout = 0.0
        self.adapter_prompt_rgb = nn.Parameter(torch.zeros(self.k, d_model))
        self.adapter_prompt_nir = nn.Parameter(torch.zeros(self.k, d_model))
        self.adapter_prompt_tir = nn.Parameter(torch.zeros(self.k, d_model))
        self.adapter_transfer = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                              QuickGELU(),
                                              nn.Dropout(dropout),
                                              nn.Linear(int(d_model // 2), int(d_model)))
        self.adapter_r = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                       QuickGELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(int(d_model // 2), int(d_model)))
        self.adapter_n = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                       QuickGELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(int(d_model // 2), int(d_model)))
        self.adapter_t = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                       QuickGELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(int(d_model // 2), int(d_model)))

        self.adapter_ffn = nn.Sequential(nn.Linear(d_model, int(d_model * 2)),
                                         QuickGELU(),
                                         nn.Linear(int(d_model * 2), int(d_model)))

    def forward_ori(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_with_adapter(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        adapter_ffn = self.adapter_ffn(x)
        x = x + self.drop_path(self.mlp(self.norm2(x))) + adapter_ffn
        return x

    def forward_with_prompt_only_first_layer(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            if index == 0:
                n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                    self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
                t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                    self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                r = last_prompt + self.adapter_transfer(last_prompt)
                n2r = last_prompt
                t2r = last_prompt
            else:
                r = last_prompt
                n2r = last_prompt
                t2r = last_prompt
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            if index == 0:
                r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                    self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
                t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                    self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                n = last_prompt + self.adapter_transfer(last_prompt)
                r2n = last_prompt
                t2n = last_prompt
            else:
                n = last_prompt
                r2n = last_prompt
                t2n = last_prompt
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            if index == 0:
                r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                    self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
                n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                    self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                t = last_prompt + self.adapter_transfer(last_prompt)
                r2t = last_prompt
                n2t = last_prompt
            else:
                t = last_prompt
                r2t = last_prompt
                n2t = last_prompt
            x = torch.cat([x, r2t, n2t, t], dim=0)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward_with_prompt(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                r = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_rgb.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                n = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_nir.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                t = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_tir.unsqueeze(
                    1).expand(-1, x.shape[1], -1))
            else:
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2t, n2t, t], dim=0)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward_with_prompt_adapter(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                r = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_rgb.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                n = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_nir.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                t = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_tir.unsqueeze(
                    1).expand(-1, x.shape[1], -1))
            else:
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2t, n2t, t], dim=0)

        x = x + self.drop_path(self.attn(self.norm1(x)))
        adapter_ffn = self.adapter_ffn(x)
        x = x + self.drop_path(self.mlp(self.norm2(x))) + adapter_ffn
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward(self, x: torch.Tensor, modality=None, index=None, last_prompt=None, prompt_sign=True,
                adapter_sign=True):
        if prompt_sign and adapter_sign:
            return self.forward_with_prompt_adapter(x, modality, index, last_prompt)
        elif prompt_sign and not adapter_sign:
            if index > self.begin:
                return self.forward_with_prompt(x, modality, index, last_prompt)
                # return self.forward_with_prompt_only_first_layer(x, modality, index, last_prompt)
            else:
                return self.forward_ori(x), None
        elif not prompt_sign and adapter_sign:
            if index > self.begin:
                return self.forward_with_adapter(x)
            else:
                return self.forward_ori(x)
        else:
            return self.forward_ori(x)


class ReAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        v = v.permute(0, 2, 1, 3).flatten(-2)
        return x, v


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# 这段代码定义了一个 HybridEmbed 类，用于从卷积神经网络（CNN）中提取特征并将其映射到一个嵌入维度。这通常用于将CNN提取的特征进一步用于其他网络（如Transformer）中。类的功能主要是将CNN的特征图（feature map）通过一定的转换映射为一个固定维度的嵌入向量，方便后续处理。
class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class Trans(nn.Module):
    """ Transformer-based Object Re-Identification
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0,
                 view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu=1.0,
                 cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other resnest
        self.local_feature = local_feature
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.prompt_sign = cfg.MODEL.PROMPT
        self.adapter_sign = cfg.MODEL.ADAPTER
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.fc = nn.Linear(embed_dim, 1000) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        scale = embed_dim ** -0.5
        self.proj_special_vit = nn.Parameter(scale * torch.randn(embed_dim, 512))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id, modality=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.cam_num > 1 and self.view_num > 1:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 1:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 1:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for i in range(len(self.blocks)):
            if self.prompt_sign and self.adapter_sign:
                if i == 0:
                    x, last_prompt = self.blocks[i](x, modality, i, None, prompt_sign=True,
                                                    adapter_sign=True)
                else:
                    x, last_prompt = self.blocks[i](x, modality, i, last_prompt, prompt_sign=True,
                                                    adapter_sign=True)
            elif self.prompt_sign and not self.adapter_sign:
                if i == 0:
                    x, last_prompt = self.blocks[i](x, modality, i, None, prompt_sign=True,
                                                    adapter_sign=False)
                else:
                    x, last_prompt = self.blocks[i](x, modality, i, last_prompt, prompt_sign=True,
                                                    adapter_sign=False)
            elif not self.prompt_sign and self.adapter_sign:
                x = self.blocks[i](x, modality, i, None, prompt_sign=False, adapter_sign=True)
            else:
                x = self.blocks[i](x, modality, i, None, prompt_sign=False, adapter_sign=False)

        x = self.norm(x)
        return x

    def forward(self, x, cam_label=None, view_label=None, modality=None):
        x = self.forward_features(x, cam_label, view_label, modality)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old resnest that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base_patch16_224(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0,
                         drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5,cfg=None, **kwargs):
    model = Trans(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, \
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu, local_feature=local_feature,cfg=cfg,**kwargs)

    return model


def vit_small_patch16_224(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0.,
                          drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5,cfg=None,
                          **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = Trans(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,
        qkv_bias=False, drop_path_rate=drop_path_rate, \
        camera=camera, view=view, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu, local_feature=local_feature,cfg=cfg, **kwargs)

    return model


def deit_small_patch16_224(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                           attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5,cfg=None,
                           **kwargs):
    model = Trans(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view,
        sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),cfg=cfg, **kwargs)

    return model


def swin_small_patch16_224(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                           attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5,cfg=None,
                           **kwargs):
    model = Trans(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view,
        sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),cfg=cfg, **kwargs)

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `model.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = model.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
