"""
增强版CLIP ReID模型 - 支持文本融合开关控制

基于原有的CLIP ReID模型，添加文本特征融合功能：
- 开关控制：可选择启用/禁用文本融合
- 多融合策略：注意力融合、拼接融合、残差融合
- 向下兼容：当开关关闭时完全保持原有功能

作者：AboutReid项目组
"""

import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .fusion_part.cross_modal_attention import create_text_fusion_module
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_tokenizer = _Tokenizer()


def weights_init_kaiming(m):
    """Kaiming权重初始化"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """分类器权重初始化"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """提示学习器"""
    def __init__(self, num_classes, dataset_name, dtype, token_embedding):
        super().__init__()
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.dtype = dtype
        self.token_embedding = token_embedding

        # 基于数据集设置提示模板
        if 'person' in dataset_name.lower():
            ctx_init = "A photo of a X X X X person."
        elif 'vehicle' in dataset_name.lower():
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X object."

        # 分词并创建提示
        ctx_init = ctx_init.replace(" X", " ").replace("  ", " ").strip()
        n_ctx = 4  # number of context tokens

        prompt_prefix = ctx_init.replace("X", "").strip()
        prompt_suffix = ""

        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.n_ctx = n_ctx

        # 创建可学习参数
        ctx_vectors = torch.empty(n_ctx, 512, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # 类别特定提示
        self.meta_net = nn.Sequential(
            nn.Linear(512, num_classes * n_ctx, bias=False),
            nn.LayerNorm(num_classes * n_ctx)
        )
        self.meta_net.half()

        # 固定提示部分
        self.register_buffer("token_prefix", torch.tensor(_tokenizer.encode(prompt_prefix), dtype=torch.long))
        self.register_buffer("token_suffix", torch.tensor(_tokenizer.encode(prompt_suffix), dtype=torch.long))
        self.tokenized_prompts = torch.cat([
            self.token_prefix,
            torch.zeros(n_ctx, dtype=torch.long),
            self.token_suffix
        ]).unsqueeze(0)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            # 类别特定提示
            prefix = prefix.unsqueeze(0).expand(label.shape[0], -1, -1)
            suffix = suffix.unsqueeze(0).expand(label.shape[0], -1, -1)
            ctx = ctx.unsqueeze(0).expand(label.shape[0], -1, -1)

            # 类别特定上下文
            class_ctx = self.meta_net(self.ctx.view(-1)).view(label.shape[0], self.n_ctx, -1)
            prompts = torch.cat([prefix, class_ctx, suffix], dim=1)
        else:
            # 通用提示
            prompts = torch.cat([prefix, ctx, suffix], dim=0).unsqueeze(0)

        return prompts

    def forward(self, label=None):
        # 固定提示编码
        prefix = self.token_embedding(self.token_prefix)
        suffix = self.token_embedding(self.token_suffix)

        # 上下文编码
        ctx = self.ctx

        # 构造提示
        prompts = self.construct_prompts(ctx, prefix, suffix, label)

        return prompts


class EnhancedCLIPReID(nn.Module):
    """
    增强版CLIP ReID模型 - 支持文本融合开关控制

    开关说明：
    - use_text_fusion=False: 保持原有CLIP ReID功能
    - use_text_fusion=True: 启用文本融合增强
    """

    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()

        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.NAME

        # ============ 开关控制参数 ============
        self.use_text_fusion = getattr(cfg.MODEL, 'USE_TEXT_FUSION', False)
        self.text_fusion_method = getattr(cfg.MODEL, 'TEXT_FUSION_METHOD', 'attention')
        self.text_fusion_weight = getattr(cfg.MODEL, 'TEXT_FUSION_WEIGHT', 0.3)

        # 模型参数
        self.in_planes = 768 if 'ViT' in self.model_name else 2048
        self.in_planes_proj = 512

        # SIE参数
        self.sie_camera = cfg.MODEL.SIE_CAMERA
        self.sie_view = cfg.MODEL.SIE_VIEW
        self.sie_coe = cfg.MODEL.SIE_COE

        # 分类器
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        # BNNeck
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # 加载CLIP模型
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        # SIE嵌入
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)

        # ============ 文本相关组件（条件创建） ============
        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

        # 文本融合模块（仅在启用时创建）
        if self.use_text_fusion:
            self.text_fusion = create_text_fusion_module(
                method=self.text_fusion_method,
                embed_dim=self.in_planes_proj  # 使用投影维度
            )
            print(f"✅ 已启用文本融合: {self.text_fusion_method}模式")
        else:
            self.text_fusion = None
            print("✅ 使用原版CLIP ReID（无文本融合）")

    def forward(self, x=None, label=None, get_image=False, get_text=False,
                cam_label=None, view_label=None, text_features=None):
        """
        增强版前向传播

        Args:
            x: 图像输入
            label: 标签
            get_image: 是否仅返回图像特征
            get_text: 是否仅返回文本特征
            cam_label: 相机标签
            view_label: 视角标签
            text_features: 外部文本特征（QwenVL）
        """

        # 仅返回文本特征
        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        # 仅返回图像特征
        if get_image:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        # ============ 图像编码（始终执行） ============
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            # SIE嵌入
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        # ============ 文本融合开关控制 ============
        if self.use_text_fusion and self.text_fusion is not None and text_features is not None:
            # ✅ 启用文本融合: 使用外部QwenVL文本特征进行融合
            enhanced_proj = self.text_fusion(img_feature_proj, text_features)
            # 可选择是否替换原始特征
            if self.text_fusion_method == "residual":
                # 残差融合：保留原始视觉特征为主
                img_feature_proj = img_feature_proj + self.text_fusion_weight * (enhanced_proj - img_feature_proj)
            else:
                # 其他融合：使用融合结果
                img_feature_proj = enhanced_proj

        # BNNeck处理
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        """加载预训练参数"""
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        """微调加载参数"""
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning {}'.format(model_path))


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    """加载CLIP模型到CPU"""
    # 使用原本项目的CLIP加载逻辑
    from .make_model_clipreid import load_clip_to_cpu as original_load_clip

    # 创建一个基本的cfg对象用于CLIP构建
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import cfg as default_cfg

    # 使用默认配置，保持原始的模型路径设置
    cfg = default_cfg.clone()

    model = original_load_clip(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size)
    return model