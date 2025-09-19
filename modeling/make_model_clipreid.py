import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import numpy as np  # 导入数值计算库
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer  # 引入简单分词器
_tokenizer = _Tokenizer()  # 实例化分词器
from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # 从 timm 引入通用层与初始化

## 重点注释这一篇文件  # 文件注释提示

def weights_init_kaiming(m): # 初始化权重
    classname = m.__class__.__name__  # 获取模块类名
    if classname.find('Linear') != -1:  # 线性层权重初始化
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')  # Kaiming 正态初始化
        nn.init.constant_(m.bias, 0.0)  # 偏置置零

    elif classname.find('Conv') != -1:  # 卷积层权重初始化
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')  # Kaiming 正态（fan_in）
        if m.bias is not None:  # 若存在偏置
            nn.init.constant_(m.bias, 0.0)  # 偏置置零
    elif classname.find('BatchNorm') != -1:  # BN 层初始化
        if m.affine:  # 仅当有可学习仿射参数
            nn.init.constant_(m.weight, 1.0)  # 权重置 1
            nn.init.constant_(m.bias, 0.0)  # 偏置置 0

def weights_init_classifier(m): # 初始化分类器
    classname = m.__class__.__name__  # 获取类名
    if classname.find('Linear') != -1:  # 线性层
        nn.init.normal_(m.weight, std=0.001)  # 权重正态分布初始化
        if m.bias:  # 若有偏置
            nn.init.constant_(m.bias, 0.0)  # 偏置置零


class TextEncoder(nn.Module):  # 文本编码器，用于将提示转换为文本特征
    def __init__(self, clip_model): # 文本编码器
        super().__init__()  # 调用父类构造
        self.transformer = clip_model.transformer  # 文本 Transformer 编码器
        self.positional_embedding = clip_model.positional_embedding  # 文本位置嵌入
        self.ln_final = clip_model.ln_final  # 最后层归一化
        self.text_projection = clip_model.text_projection  # 文本到共同空间的投影矩阵
        self.dtype = clip_model.dtype  # 数据类型（float32/float16）

    def forward(self, prompts, tokenized_prompts): # 前向传播
        x = prompts + self.positional_embedding.type(self.dtype)  # 加上位置嵌入
        x = x.permute(1, 0, 2)  # NLD -> LND  # 变换维度以适配 transformer 输入
        x = self.transformer(x)  # 通过 transformer 编码
        x = x.permute(1, 0, 2)  # LND -> NLD  # 变回 batch 第一维
        x = self.ln_final(x).type(self.dtype)  # 最后层归一化并转 dtype

        # x.shape = [batch_size, n_ctx, transformer.width]  # 特征维度说明
        # take features from the eot embedding (eot_token is the highest number in each sequence)  # 取 EOT 位置的特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # 选取 EOT 特征并线性投影
        return x  # 返回文本特征

class build_transformer(nn.Module): # 构建Transformer
    def __init__(self, num_classes, camera_num, view_num, cfg): # 初始化
        super(build_transformer, self).__init__()  # 父类构造
        self.model_name = cfg.MODEL.NAME  # 模型名称（ViT-B-16 或 RN50）
        self.cos_layer = cfg.MODEL.COS_LAYER  # 是否使用余弦层（若有）
        self.neck = cfg.MODEL.NECK  # 颈部结构类型
        self.neck_feat = cfg.TEST.NECK_FEAT  # 测试时使用 neck 前/后特征
        if self.model_name == 'ViT-B-16':  # CLIP ViT-B/16 配置
            self.in_planes = 768  # 主干输出维度
            self.in_planes_proj = 768  # 投影空间维度
        elif self.model_name == 'RN50':  # CLIP ResNet50 配置
            self.in_planes = 2048  # 主干输出维度
            self.in_planes_proj = 1024  # 投影空间维度
        self.num_classes = num_classes  # 类别数
        self.camera_num = camera_num  # 相机数量
        self.view_num = view_num  # 视角数量
        self.sie_coe = cfg.MODEL.SIE_COE   # SIE 系数

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)  # 主特征分类头
        self.classifier.apply(weights_init_classifier)  # 初始化分类头
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)  # 投影特征分类头
        self.classifier_proj.apply(weights_init_classifier)  # 初始化分类头

        self.bottleneck = nn.BatchNorm1d(self.in_planes)  # BNNeck（主特征）
        self.bottleneck.bias.requires_grad_(False)  # 冻结偏置
        self.bottleneck.apply(weights_init_kaiming)  # 初始化 BN
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)  # BNNeck（投影特征）
        self.bottleneck_proj.bias.requires_grad_(False)  # 冻结偏置
        self.bottleneck_proj.apply(weights_init_kaiming)  # 初始化 BN

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)  # 视觉网格高
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)  # 视觉网格宽
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]  # 步长（patch stride）
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)  # 加载 CLIP
        clip_model.to("cuda")  # 移至 GPU

        self.image_encoder = clip_model.visual  # 视觉编码器（CLIP视觉部分）

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:  # 同时按相机与视角
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))  # 嵌入表
            trunc_normal_(self.cv_embed, std=.02)  # 截断正态初始化
            print('camera number is : {}'.format(camera_num))  # 打印相机数
        elif cfg.MODEL.SIE_CAMERA:  # 仅相机
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))  # 相机嵌入
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:  # 仅视角
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))  # 视角嵌入
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES  # 数据集名称
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)  # 提示学习器
        self.text_encoder = TextEncoder(clip_model)  # 文本编码器

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None): # 前向传播
        if get_text == True:  # 仅提取文本特征
            prompts = self.prompt_learner(label)  # 基于标签生成提示
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)  # 文本编码
            return text_features  # 返回文本特征

        if get_image == True:  # 仅提取图像特征
            image_features_last, image_features, image_features_proj = self.image_encoder(x)  # 前向提取
            if self.model_name == 'RN50':  # ResNet50 分支
                return image_features_proj[0]  # 返回投影特征（展平）
            elif self.model_name == 'ViT-B-16':  # ViT 分支
                return image_features_proj[:,0]  # 返回 CLS 投影特征
        
        if self.model_name == 'RN50':  # ResNet50 完整路径
            image_features_last, image_features, image_features_proj = self.image_encoder(x)  # 前向
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)  # GAP 后展平
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)  # GAP 主特征
            img_feature_proj = image_features_proj[0]  # 投影特征

        elif self.model_name == 'ViT-B-16':  # ViT 完整路径
            if cam_label != None and view_label!=None:  # 同时使用相机与视角
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]  # 获取嵌入
            elif cam_label != None:  # 仅相机
                cv_embed = self.sie_coe * self.cv_embed[cam_label]  # 相机嵌入
            elif view_label!=None:  # 仅视角
                cv_embed = self.sie_coe * self.cv_embed[view_label]  # 视角嵌入
            else:
                cv_embed = None  # 不使用嵌入
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)  # 前向带嵌入
            img_feature_last = image_features_last[:,0]  # 最后层 CLS
            img_feature = image_features[:,0]  # 主特征 CLS
            img_feature_proj = image_features_proj[:,0]  # 投影 CLS

        feat = self.bottleneck(img_feature)  # BNNeck 后特征
        feat_proj = self.bottleneck_proj(img_feature_proj)  # 投影 BNNeck 后特征
        
        if self.training:  # 训练分支
            cls_score = self.classifier(feat)  # 主特征分类分数
            cls_score_proj = self.classifier_proj(feat_proj)  # 投影特征分类分数
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj  # 返回损失所需

        else:  # 测试分支
            if self.neck_feat == 'after':  # 使用 BN 后特征
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)  # 拼接主+投影
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)  # 使用 BN 前特征


    def load_param(self, trained_path): # 加载参数
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path): # 加载参数
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num): # 构建模型
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip   # 从当前包导入 clip 模块（通常是 OpenAI CLIP 的封装实现）


def load_clip_to_cpu(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size):  
    """
    加载 CLIP 模型到 CPU 上，并返回构建好的模型
    参数:
        cfg: 配置文件
        backbone_name: 主干网络名称（此处未使用）
        h_resolution, w_resolution: 输入图像的分辨率
        vision_stride_size: 图像编码器的步幅大小
    """
    model_path = '/home/zubuntu/workspace/yzy/MambaPro/pths/ViT-B-16.pt'  # 预训练 CLIP 模型的路径

    try:
        # 优先尝试加载 JIT 编译好的 TorchScript 模型
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError: 
        # 如果 JIT 加载失败，就直接加载普通的 state_dict 参数
        state_dict = torch.load(model_path, map_location="cpu")

    # 使用 clip.build_model() 构建 CLIP 模型
    # 如果 state_dict 存在，加载权重；否则用 JIT 模型的权重
    model = clip.build_model(cfg, state_dict or model.state_dict(), 
                             h_resolution, w_resolution, vision_stride_size)

    return model



class PromptLearner(nn.Module):  
    """
    提示学习器 (Prompt Learner)
    用于为 CLIP 模型生成可学习的文本提示（prompts），
    不同类别会学习不同的上下文向量，从而提升分类性能。
    """
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        """
        参数:
            num_class: 数据集中的类别数
            dataset_name: 数据集名称（决定提示模板是 vehicle 还是 person）
            dtype: 数据类型 (如 torch.float32 / float16)
            token_embedding: CLIP 的词向量嵌入层
        """
        super().__init__()
        
        # 根据数据集类型选择提示模板
        if dataset_name == "VehicleID" or dataset_name == "veri": 
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512       # 上下文向量维度（CLIP 默认文本编码维度为 512）
        n_ctx = 4           # 可学习的上下文 token 数量

        # 预处理提示文本
        ctx_init = ctx_init.replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).cuda()  # 将模板转为 token
        with torch.no_grad():
            # 将 token 转换为嵌入向量，作为上下文初始化
            embedding = token_embedding(tokenized_prompts).type(dtype)  

        self.tokenized_prompts = tokenized_prompts  # 保存原始 token（不可训练）

        # 为每个类别创建独立的上下文向量
        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)      # 正态分布初始化
        self.cls_ctx = nn.Parameter(cls_vectors)    # 作为可训练参数

        # 处理 token embedding 的前缀（class-independent）和后缀
        # 前缀：模板中不可学习的前部分 (e.g., "A photo of a")
        # 后缀：模板中不可学习的后部分 (e.g., "person.")
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx


    def forward(self, label):  
        """
        前向传播: 根据类别标签生成对应的 prompts 向量
        参数:
            label: 类别标签 (batch_size,)
        返回:
            prompts: 拼接好的 token 向量 (batch_size, token_len, dim)
        """
        cls_ctx = self.cls_ctx[label]          # 取出对应类别的上下文向量
        b = label.shape[0]                     # batch 大小
        prefix = self.token_prefix.expand(b, -1, -1)  # 扩展前缀到 batch
        suffix = self.token_suffix.expand(b, -1, -1)  # 扩展后缀到 batch
            
        # 拼接 [前缀 + 类别上下文 + 后缀]
        prompts = torch.cat(
            [
                prefix,      # (batch, 前缀长度, dim)
                cls_ctx,     # (batch, n_cls_ctx, dim)
                suffix,      # (batch, 后缀长度, dim)
            ],
            dim=1,
        ) 

        return prompts


