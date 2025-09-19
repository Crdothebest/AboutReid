# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

## 提取数据特征
"""
T2T-ViT   # 文件描述：Tokens-to-Token Vision Transformer 实现
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


def _cfg(url='', **kwargs):  # 构造 timm 默认配置字典
    return {
        'url': url,  # 预训练权重下载地址
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,  # 分类数与输入尺寸
        'crop_pct': .9, 'interpolation': 'bicubic',  # 默认裁剪比例与插值方式
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),  # 归一化均值与方差
        'classifier': 'head',  # 分类头字段名
        **kwargs  # 允许额外键覆盖
    }


default_cfgs = {  # timm 风格的不同规模配置占位
    # 这里可以加新的模型，但是需要改 forward 里的  # 提示：新增变体需同步 forward 适配
    'T2t_vit_7': _cfg(),  # 深度7
    'T2t_vit_10': _cfg(),  # 深度10
    'T2t_vit_12': _cfg(),  # 深度12
    'T2t_vit_14': _cfg(),  # 深度14
    'T2t_vit_19': _cfg(),  # 深度19
    'T2t_vit_24': _cfg(),  # 深度24
    'T2t_vit_t_14': _cfg(),  # transformer tokens 版 14
    'T2t_vit_t_19': _cfg(),  # transformer tokens 版 19
    'T2t_vit_t_24': _cfg(),  # transformer tokens 版 24
    'T2t_vit_14_resnext': _cfg(),  # resnext 头
    'T2t_vit_14_wide': _cfg(),  # 宽版
}


class T2T_module(nn.Module):  # Tokens-to-Token 编码模块
    """
    Tokens-to-Token encoding module  # 通过 unfold + 轻量注意力分层聚合 tokens
    """

   ## 在下面这个函数里面去写滑动窗口的代码
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):  # 初始化参数
        super().__init__()  # 父类构造

        if tokens_type == 'transformer':  # 使用 Transformer 作为 T2T 编码器
            print('adopt transformer encoder for tokens-to-token')  # 日志提示
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # 第一次软切分
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 第二次软切分
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 第三次软切分

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)  # 第1层 token 编码
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)  # 第2层 token 编码
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)  # 线性映射到最终嵌入维度

        elif tokens_type == 'performer':  # 使用 Performer 作为 T2T 编码器
            print('adopt performer encoder for tokens-to-token')  # 日志提示
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # 第一次软切分
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 第二次软切分
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 第三次软切分

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)  # 备选实现
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)  # 备选实现
            self.attention1 = Token_performer(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)  # 第1层 performer
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)  # 第2层 performer
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)  # 投影到嵌入维度

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model  # 卷积对照组
            # for this tokens type, you need change forward as three convolution operation  # 需要改 forward 为三次卷积
            print('adopt convolution layers for tokens-to-token')  # 日志提示
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution  # 第一次卷积替代 unfold
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1))  # the 2nd convolution  # 第二次卷积
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1))  # the 3rd convolution  # 第三次卷积作为投影

        self.num_patches = (img_size[0] // (4 * 2 * 2)) * (  # 计算最终 token 数
                img_size[1] // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately  # 3 次软切分的总步长

## 重点需要改的就是 forward 里的
    def forward(self, x):  # 前向传播：三段式 T2T 编码
        # step0: soft split  # 第一次软切分为 patch tokens
        x = self.soft_split0(x).transpose(1, 2)  # [B, C*k*k, H'*W'] -> [B, N, Ck^2]

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)  # 第一次 token 编码（重组）
        B, new_HW, C = x.shape  # 记录形状
        x = x.transpose(1, 2).reshape(B, C, 64, 32)  # 恢复为 2D 特征图（依赖输入尺寸设定）
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)  # 第二次软切分

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)  # 第二次 token 编码（重组）
        B, new_HW, C = x.shape  # 更新形状
        x = x.transpose(1, 2).reshape(B, C, 32, 16)  # 再次恢复为 2D 特征图
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)  # 第三次软切分

        # final tokens
        x = self.project(x)  # 最终线性投影得到 tokens

        return x  # 返回 tokens 序列


class T2T_ViT(nn.Module):  # 主体模型：T2T 编码 + ViT 编码
    def __init__(self, img_size=(256, 128), tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., camera=0,
                 view=0, sie_xishu=3.0, norm_layer=nn.LayerNorm, token_dim=64):  # 初始化各项配置
        super().__init__()  # 父类构造
        self.num_classes = num_classes  # 类别数
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models  # 记录特征维度

        self.tokens_to_token = T2T_module(
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)  # T2T 编码器
        num_patches = self.tokens_to_token.num_patches  # 最终 token 数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类 token 参数
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim),
                                      requires_grad=False)  # 正弦位置编码（固定）
        self.pos_drop = nn.Dropout(p=drop_rate)  # 位置后 dropout
        self.cam_num = camera  # 相机数量
        self.view_num = view  # 视角数量
        self.sie_xishu = sie_xishu  # SIE 系数
        # Initialize SIE Embedding
        if camera > 1 and view > 1:  # 同时使用相机与视角嵌入
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))  # 嵌入表
            trunc_normal_(self.sie_embed, std=.02)  # 截断正态初始化
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))  # 打印信息
            print('using SIE_Lambda is : {}'.format(sie_xishu))  # 打印 SIE 系数
        elif camera > 1:  # 仅相机嵌入
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))  # 相机嵌入
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:  # 仅视角嵌入
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))  # 视角嵌入
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))  # 打印 Dropout
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))  # 打印 注意力 Dropout
        print('using drop_path rate is : {}'.format(drop_path_rate))  # 打印 DropPath
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule  # 分层 DropPath 率
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])  # ViT 主干 block 列表
        self.norm = norm_layer(embed_dim)  # 最后层归一化

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # 分类头或恒等

        trunc_normal_(self.cls_token, std=.02)  # 初始化 cls token
        self.apply(self._init_weights)  # 递归初始化权重

    def _init_weights(self, m):  # 通用初始化
        if isinstance(m, nn.Linear):  # 线性层
            trunc_normal_(m.weight, std=.02)  # 截断正态
            if isinstance(m, nn.Linear) and m.bias is not None:  # 若有偏置
                nn.init.constant_(m.bias, 0)  # 偏置置零
        elif isinstance(m, nn.LayerNorm):  # 层归一化
            nn.init.constant_(m.bias, 0)  # 偏置置零
            nn.init.constant_(m.weight, 1.0)  # 权重置1

    @torch.jit.ignore
    def no_weight_decay(self):  # 指定不参与权重衰减的参数名
        return {'cls_token'}  # cls_token 不做 weight decay

    def get_classifier(self):  # 取分类头
        return self.head  # 返回分类头

    def reset_classifier(self, num_classes, global_pool=''):  # 重设分类头
        self.num_classes = num_classes  # 更新类别数
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # 新建头

## x是用来提取特
    def forward_features(self, x, camera_id, view_id):  # 提取特征（含 SIE）
        B = x.shape[0]  # batch 大小
        x = self.tokens_to_token(x)  # T2T 编码为 tokens

        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展 cls token
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接到 tokens 前
        if self.cam_num > 0 and self.view_num > 0:  # 同时加相机与视角嵌入
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]  # 加位置与SIE
        elif self.cam_num > 0:  # 仅相机
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]  # 加位置与相机
        elif self.view_num > 0:  # 仅视角
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]  # 加位置与视角
        else:
            x = x + self.pos_embed  # 仅加位置

        x = self.pos_drop(x)  # dropout

        for blk in self.blocks:  # 逐层 Transformer 编码
            x = blk(x)  # 通过一个 Block

        x = self.norm(x)  # 最后一层归一化
        return x  # 返回 tokens（含 cls）

# x是用来提取特征的  此处需要打断点；换一个不同尺度的滑动窗口集成起来
    def forward(self, x, cam_label=None, view_label=None):  # 前向：返回中间缓存
        x = self.forward_features(x, camera_id=cam_label, view_id=view_label)  # 取特征
        cash_x = []  # 建立列表
        cash_x.append(x)  # 追加特征
        return cash_x  # 返回列表

        # ************* 在这里加 滑动窗口

    def load_param(self, model_path):  # 加载预训练权重并做必要适配
        param_dict = torch.load(model_path)  # 读取权重
        if 'model' in param_dict:  # 适配常见权重字典结构
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'state_dict_ema' in param_dict:
            param_dict = param_dict['state_dict_ema']
        for k, v in param_dict.items():  # 遍历参数
            if 'head' in k or 'dist' in k:  # 跳过分类头/蒸馏分支
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:  # 老版权重适配
                # For old resnest that I trained prior to conv based patchification  # 注释保留
                O, I, H, W = self.patch_embed.proj.weight.shape  # 读取目标形状
                v = v.reshape(O, -1, H, W)  # 重塑为卷积权重
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:  # 位置编码尺寸不匹配
                # To resize pos embedding when using model at different size from pretrained weights  # 说明
                if 'distilled' in model_path:  # 处理蒸馏模型的双 cls token
                    print('distill need to choose right cls token in the pth')  # 提示
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)  # 取其一
                v = resize_pos_embed(v, self.pos_embed, 16, 8)  # 依据目标尺寸插值
            try:
                self.state_dict()[k].copy_(v)  # 复制参数
            except:
                print('===========================ERROR=========================')  # 错误提示
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))  # 打印形状不匹配


def resize_pos_embed(posemb, posemb_new, hight, width):  # 调整位置编码大小
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from  # 注释保留
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224  # 参考链接
    ntok_new = posemb_new.shape[1]  # 新的 token 数

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]  # 分离 cls 与网格
    ntok_new -= 1  # 去除 cls

    gs_old = int(math.sqrt(len(posemb_grid)))  # 旧网格边长
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))  # 打印信息
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # 重塑为 NCHW
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')  # 双线性插值到目标大小
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)  # 还原为序列
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)  # 拼回 cls
    return posemb  # 返回新位置编码


@register_model
def t2t_vit_7(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
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
               **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_14(num_classes=1051, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                 attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, pretrained=False,
                 **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(num_classes=num_classes, img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
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

# model = t2t_vit_t_24()
# data = torch.randn(2, 3, 256, 128)
# model.load_param('/15127306268/wyh/UIS/pth/82.6_T2T_ViTt_24_pth.tar')
# model(data)
