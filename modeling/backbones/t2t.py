# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

"""
T2T-ViT (Tokens-to-Token Vision Transformer) 实现

功能概述：
- 实现基于Tokens-to-Token机制的Vision Transformer架构
- 通过渐进式token重组减少token数量，提高计算效率
- 支持多种token编码器：Transformer、Performer、Convolution
- 集成SIE（Side Information Embedding）用于相机和视角信息
- 提供多种模型变体：不同深度、宽度和注意力头数配置

主要创新：
1. T2T模块：通过unfold操作和轻量级注意力逐步重组tokens
2. 多尺度特征提取：支持不同尺寸的滑动窗口
3. 位置编码：使用正弦位置编码和可学习的SIE嵌入
4. 模型变体：提供从7层到24层的多种配置
"""
import torch  # 引入 PyTorch 主库
import torch.nn as nn  # 引入神经网络模块
import torch.nn.functional as F  # 引入常用函数接口（如插值、激活等）
from timm.models.helpers import load_pretrained  # 从 timm 导入预训练加载工具
from timm.models.registry import register_model  # 从 timm 导入模型注册装饰器
from timm.models.layers import trunc_normal_  # 从 timm 导入截断正态初始化

from modeling.backbones.token_transformer import Token_transformer  # 轻量 token 级 transformer 编码器
from modeling.backbones.token_performer import Token_performer  # Performer 注意力变体（线性注意力）
from modeling.backbones.transformer_block import Block, get_sinusoid_encoding  # 主体 Transformer 块与位置编码
import math  # 数学工具库


def _cfg(url='', **kwargs):
    """
    构造timm风格的模型配置字典
    
    功能：
    - 为不同T2T-ViT变体提供标准配置参数
    - 包含预训练权重URL、输入尺寸、数据预处理参数等
    - 支持通过kwargs覆盖默认配置
    
    参数：
    - url: 预训练权重下载地址
    - **kwargs: 额外配置参数，可覆盖默认值
    
    返回：
    - dict: 包含模型配置的字典
    """
    return {
        'url': url,  # 预训练权重下载地址
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,  # 分类数与输入尺寸
        'crop_pct': .9, 'interpolation': 'bicubic',  # 默认裁剪比例与插值方式
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),  # 归一化均值与方差
        'classifier': 'head',  # 分类头字段名
        **kwargs  # 允许额外键覆盖
    }


# 不同T2T-ViT模型变体的配置字典
# 注意：新增模型变体时需要同步修改forward函数中的处理逻辑
default_cfgs = {
    # Performer-based T2T-ViT variants (使用Performer作为token编码器)
    'T2t_vit_7': _cfg(),      # 深度7层，轻量级版本
    'T2t_vit_10': _cfg(),     # 深度10层，中等规模
    'T2t_vit_12': _cfg(),     # 深度12层，标准配置
    'T2t_vit_14': _cfg(),     # 深度14层，较大规模
    'T2t_vit_19': _cfg(),     # 深度19层，大规模
    'T2t_vit_24': _cfg(),     # 深度24层，最大规模
    
    # Transformer-based T2T-ViT variants (使用Transformer作为token编码器)
    'T2t_vit_t_14': _cfg(),   # Transformer tokens版本，深度14
    'T2t_vit_t_19': _cfg(),   # Transformer tokens版本，深度19
    'T2t_vit_t_24': _cfg(),   # Transformer tokens版本，深度24
    
    # Specialized variants (特殊变体)
    'T2t_vit_14_resnext': _cfg(),  # ResNeXt风格的注意力头配置
    'T2t_vit_14_wide': _cfg(),     # 宽版本，更多通道数
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token编码模块
    
    功能概述：
    - 实现渐进式token重组机制，通过unfold操作和轻量级注意力逐步减少token数量
    - 支持三种token编码器：Transformer、Performer、Convolution
    - 通过多阶段处理将图像patches转换为更紧凑的token表示
    
    核心机制：
    1. 软切分(Soft Split)：使用unfold操作将图像分割为重叠的patches
    2. 重组(Re-structurization)：通过轻量级注意力机制重新组织tokens
    3. 投影(Projection)：将最终tokens投影到目标嵌入维度
    
    参数：
    - img_size: 输入图像尺寸
    - tokens_type: token编码器类型 ('performer', 'transformer', 'convolution')
    - in_chans: 输入通道数
    - embed_dim: 最终嵌入维度
    - token_dim: 中间token维度
    """
    
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()
        
        # 根据token编码器类型初始化不同的网络结构
        if tokens_type == 'transformer':
            """
            使用Transformer作为T2T编码器
            - 三次软切分：7x7->3x3->3x3，步长分别为4,2,2
            - 两次Transformer注意力重组
            - 最终线性投影到目标维度
            """
            print('adopt transformer encoder for tokens-to-token')
            # 三次软切分操作，逐步减少token数量
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # 第一次：7x7窗口，步长4
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 第二次：3x3窗口，步长2
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 第三次：3x3窗口，步长2

            # 两次Transformer注意力重组
            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)  # 最终投影

        elif tokens_type == 'performer':
            """
            使用Performer作为T2T编码器（推荐）
            - Performer使用线性注意力，计算复杂度更低
            - 相同的三次软切分策略
            - 两次Performer注意力重组
            """
            print('adopt performer encoder for tokens-to-token')
            # 三次软切分操作
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # 两次Performer注意力重组（线性复杂度）
            self.attention1 = Token_performer(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':
            """
            使用卷积作为T2T编码器（对照组）
            - 仅用于与卷积方法对比，不是主要模型
            - 需要修改forward函数为三次卷积操作
            """
            print('adopt convolution layers for tokens-to-token')
            # 三次卷积操作替代unfold
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # 计算最终token数量：三次软切分的总步长为4*2*2=16
        self.num_patches = (img_size[0] // (4 * 2 * 2)) * (img_size[1] // (4 * 2 * 2))

    def forward(self, x):
        """
        T2T模块前向传播：三段式渐进式token重组
        
        处理流程：
        1. 第一次软切分：将图像分割为7x7的patches
        2. 第一次重组：通过注意力机制重新组织tokens
        3. 第二次软切分：将重组后的特征分割为3x3的patches
        4. 第二次重组：再次通过注意力机制重组tokens
        5. 第三次软切分：最终分割为3x3的patches
        6. 投影：线性投影到目标嵌入维度
        
        参数：
        - x: 输入图像张量 [B, C, H, W]
        
        返回：
        - x: 处理后的token序列 [B, N, embed_dim]
        """
        # 第一阶段：第一次软切分 (7x7窗口，步长4)
        x = self.soft_split0(x).transpose(1, 2)  # [B, C*7*7, H'*W'] -> [B, N, C*49]

        # 第一次重组：通过注意力机制重新组织tokens
        x = self.attention1(x)  # [B, N, token_dim]
        B, new_HW, C = x.shape
        # 恢复为2D特征图以便进行下一次软切分
        x = x.transpose(1, 2).reshape(B, C, 64, 32)  # 假设输入为256x128
        
        # 第二阶段：第二次软切分 (3x3窗口，步长2)
        x = self.soft_split1(x).transpose(1, 2)  # [B, C*9, H''*W''] -> [B, N', C*9]

        # 第二次重组：再次通过注意力机制重组tokens
        x = self.attention2(x)  # [B, N', token_dim]
        B, new_HW, C = x.shape
        # 再次恢复为2D特征图
        x = x.transpose(1, 2).reshape(B, C, 32, 16)  # 进一步降维
        
        # 第三阶段：第三次软切分 (3x3窗口，步长2)
        x = self.soft_split2(x).transpose(1, 2)  # [B, C*9, H'''*W'''] -> [B, N'', C*9]

        # 最终投影：线性投影到目标嵌入维度
        x = self.project(x)  # [B, N'', embed_dim]

        return x


class T2T_ViT(nn.Module):
    """
    T2T-ViT主体模型：结合T2T编码和ViT编码的完整架构
    
    功能概述：
    - 集成T2T模块进行渐进式token重组
    - 使用标准ViT Transformer块进行特征编码
    - 支持SIE（Side Information Embedding）用于相机和视角信息
    - 提供完整的分类头用于下游任务
    
    主要组件：
    1. T2T_module: 渐进式token重组模块
    2. 位置编码: 正弦位置编码 + 可学习SIE嵌入
    3. Transformer块: 多层标准ViT编码器
    4. 分类头: 用于最终分类任务
    
    参数：
    - img_size: 输入图像尺寸
    - tokens_type: T2T编码器类型
    - num_classes: 分类类别数
    - embed_dim: 嵌入维度
    - depth: Transformer块层数
    - num_heads: 注意力头数
    - camera/view: 相机/视角数量（用于SIE）
    - sie_xishu: SIE嵌入系数
    """
    
    def __init__(self, img_size=(256, 128), tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., camera=0, view=0, sie_xishu=3.0, 
                 norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # 初始化T2T编码模块
        self.tokens_to_token = T2T_module(
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, 
            embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        # 初始化分类token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 可学习的分类token
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim),
                                      requires_grad=False)  # 固定的正弦位置编码
        self.pos_drop = nn.Dropout(p=drop_rate)  # 位置编码后的dropout
        
        # SIE（Side Information Embedding）相关参数
        self.cam_num = camera  # 相机数量
        self.view_num = view   # 视角数量
        self.sie_xishu = sie_xishu  # SIE嵌入系数
        
        # 初始化SIE嵌入
        if camera > 1 and view > 1:  # 同时使用相机和视角嵌入
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:  # 仅使用相机嵌入
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:  # 仅使用视角嵌入
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        # 打印训练配置信息
        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))
        
        # 初始化Transformer块列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减率
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                  drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  # 最终层归一化

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


        # 初始化权重
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        功能：
        - 为不同类型的层设置合适的初始化策略
        - 线性层使用截断正态分布初始化
        - LayerNorm层使用标准初始化
        
        参数：
        - m: 待初始化的模块
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 截断正态分布初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)      # 偏置初始化为0
            nn.init.constant_(m.weight, 1.0)  # 权重初始化为1

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        指定不参与权重衰减的参数
        
        功能：
        - 返回不需要权重衰减的参数名称列表
        - cls_token通常不需要权重衰减
        
        返回：
        - set: 不参与权重衰减的参数名称集合
        """
        return {'cls_token'}

    def get_classifier(self):
        """
        获取分类头模块
        
        返回：
        - nn.Module: 分类头模块
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """
        重置分类头
        
        功能：
        - 根据新的类别数重新创建分类头
        - 用于迁移学习或改变任务
        
        参数：
        - num_classes: 新的类别数
        - global_pool: 全局池化方式（未使用）
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # 创新点01: multi_scale 多尺度的特征提取，应该添加在forward_features方法中
    def forward_features(self, x, camera_id, view_id):
        """
        特征提取前向传播（包含SIE嵌入）
        
        功能：
        - 通过T2T模块将图像转换为tokens
        - 添加分类token和位置编码
        - 根据相机和视角信息添加SIE嵌入
        - 通过多层Transformer块进行特征编码
        
        参数：
        - x: 输入图像张量 [B, C, H, W]
        - camera_id: 相机ID [B]
        - view_id: 视角ID [B]
        
        返回：
        - x: 编码后的特征tokens [B, N+1, embed_dim] (包含cls_token)
        """
        B = x.shape[0]
        
        # 通过T2T模块将图像转换为tokens
        x = self.tokens_to_token(x)  # [B, N, embed_dim]

        # 添加分类token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, embed_dim]
        
        # 添加位置编码和SIE嵌入
        if self.cam_num > 0 and self.view_num > 0:  # 同时使用相机和视角嵌入
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:  # 仅使用相机嵌入
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:  # 仅使用视角嵌入
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:  # 仅使用位置编码
            x = x + self.pos_embed

        # 应用dropout
        x = self.pos_drop(x)

        # 通过多层Transformer块进行特征编码
        for blk in self.blocks:
            x = blk(x)

        # 最终层归一化
        x = self.norm(x)
        return x

    def forward(self, x, cam_label=None, view_label=None):
        """
        模型前向传播主函数
        
        功能：
        - 调用forward_features进行特征提取
        - 返回特征列表用于后续处理
        - 支持相机和视角标签输入
        
        参数：
        - x: 输入图像张量 [B, C, H, W]
        - cam_label: 相机标签 [B] (可选)
        - view_label: 视角标签 [B] (可选)
        
        返回：
        - list: 包含特征tokens的列表 [x] (用于多尺度处理)
        
        注意：
        - 此处返回列表格式，便于后续多尺度滑动窗口处理
        - 可以在此处添加不同尺度的滑动窗口集成
        """
        # 提取特征
        x = self.forward_features(x, camera_id=cam_label, view_id=view_label)
        
        # 返回特征列表（便于多尺度处理）
        cash_x = []
        cash_x.append(x)
        return cash_x
        
        # TODO: 在此处可以添加多尺度滑动窗口处理逻辑

    def load_param(self, model_path):
        """
        加载预训练权重并进行必要适配
        
        功能：
        - 从指定路径加载预训练模型权重
        - 处理不同权重字典格式的兼容性
        - 适配位置编码尺寸不匹配的情况
        - 跳过分类头和蒸馏分支的权重
        
        参数：
        - model_path: 预训练权重文件路径
        
        处理逻辑：
        1. 加载权重字典并适配不同格式
        2. 跳过分类头和蒸馏分支权重
        3. 处理老版本权重的形状适配
        4. 处理位置编码尺寸不匹配的情况
        5. 复制匹配的权重参数
        """
        param_dict = torch.load(model_path)
        
        # 适配不同的权重字典格式
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'state_dict_ema' in param_dict:
            param_dict = param_dict['state_dict_ema']
            
        for k, v in param_dict.items():
            # 跳过分类头和蒸馏分支权重
            if 'head' in k or 'dist' in k:
                continue
                
            # 处理老版本权重的形状适配
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # 适配老版本ResNeSt的卷积权重格式
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
                
            # 处理位置编码尺寸不匹配
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # 处理蒸馏模型的双cls token
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                # 根据目标尺寸调整位置编码
                v = resize_pos_embed(v, self.pos_embed, 16, 8)
                
            # 复制匹配的权重参数
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(
                    k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    """
    调整位置编码大小以适配不同的输入尺寸
    
    功能：
    - 当使用不同尺寸的预训练权重时，调整位置编码的尺寸
    - 分离cls token和网格位置编码
    - 使用双线性插值调整网格位置编码尺寸
    - 重新组合cls token和调整后的网格编码
    
    参数：
    - posemb: 原始位置编码 [1, N+1, D]
    - posemb_new: 目标位置编码 [1, M+1, D]
    - hight: 目标高度
    - width: 目标宽度
    
    返回：
    - posemb: 调整后的位置编码 [1, M+1, D]
    
    参考：
    - https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]

    # 分离cls token和网格位置编码
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]  # cls token和网格编码
    ntok_new -= 1  # 去除cls token

    # 计算原始网格尺寸
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(
        posemb.shape, posemb_new.shape, hight, width))
    
    # 重塑为2D网格并进行双线性插值
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, D, H, W]
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')  # 插值到目标尺寸
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)  # 还原为序列格式
    
    # 重新组合cls token和调整后的网格编码
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


@register_model
def t2t_vit_7(pretrained=False, **kwargs):
    """
    T2T-ViT-7模型工厂函数
    
    功能：
    - 创建深度为7层的T2T-ViT模型
    - 使用Performer作为token编码器
    - 轻量级配置，适合资源受限环境
    
    配置参数：
    - embed_dim: 256 (嵌入维度)
    - depth: 7 (Transformer层数)
    - num_heads: 4 (注意力头数)
    - mlp_ratio: 2.0 (MLP扩展比例)
    - tokens_type: 'performer' (使用Performer编码器)
    
    参数：
    - pretrained: 是否加载预训练权重
    - **kwargs: 其他配置参数
    
    返回：
    - T2T_ViT: 配置好的模型实例
    """
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_10(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_12(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_19(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_24(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
               attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, pretrained=False,
               **kwargs):
    """
    T2T-ViT-24模型工厂函数（最大规模版本）
    
    功能：
    - 创建深度为24层的T2T-ViT模型
    - 使用Performer作为token编码器
    - 最大规模配置，提供最佳性能
    
    配置参数：
    - embed_dim: 512 (嵌入维度)
    - depth: 24 (Transformer层数)
    - num_heads: 8 (注意力头数)
    - mlp_ratio: 3.0 (MLP扩展比例)
    - tokens_type: 'performer' (使用Performer编码器)
    - img_size: (256, 128) (输入图像尺寸，适合ReID任务)
    
    特殊参数：
    - camera: 相机数量（用于SIE嵌入）
    - view: 视角数量（用于SIE嵌入）
    - sie_xishu: SIE嵌入系数
    - drop_path_rate: 随机深度衰减率
    
    参数：
    - pretrained: 是否加载预训练权重
    - **kwargs: 其他配置参数
    
    返回：
    - T2T_ViT: 配置好的模型实例
    """
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_14(num_classes=1051, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                 attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, pretrained=False,
                 **kwargs):
    """
    T2T-ViT-T-14模型工厂函数（Transformer版本）
    
    功能：
    - 创建深度为14层的T2T-ViT模型
    - 使用Transformer作为token编码器（而非Performer）
    - 提供与Performer版本的对比实验
    
    配置参数：
    - embed_dim: 384 (嵌入维度)
    - depth: 14 (Transformer层数)
    - num_heads: 6 (注意力头数)
    - mlp_ratio: 3.0 (MLP扩展比例)
    - tokens_type: 'transformer' (使用Transformer编码器)
    - num_classes: 1051 (默认类别数，适合特定数据集)
    
    特殊参数：
    - camera: 相机数量（用于SIE嵌入）
    - view: 视角数量（用于SIE嵌入）
    - sie_xishu: SIE嵌入系数
    
    参数：
    - pretrained: 是否加载预训练权重
    - **kwargs: 其他配置参数
    
    返回：
    - T2T_ViT: 配置好的模型实例
    
    注意：
    - 此版本使用标准Transformer而非Performer，计算复杂度更高
    - 主要用于与Performer版本进行性能对比
    """
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(num_classes=num_classes, img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_24(num_classes=1051, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                 attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, pretrained=False,
                 **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(num_classes=num_classes, img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


# rexnext and wide structure
@register_model
def t2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# 测试代码示例（已注释）
# 用于验证模型创建和前向传播的正确性
# model = t2t_vit_t_24()  # 创建T2T-ViT-T-24模型
# data = torch.randn(2, 3, 256, 128)  # 创建测试数据 [batch_size, channels, height, width]
# model.load_param('/15127306268/wyh/UIS/pth/82.6_T2T_ViTt_24_pth.tar')  # 加载预训练权重
# model(data)  # 执行前向传播测试
