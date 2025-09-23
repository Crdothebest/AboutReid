"""
MambaPro 模型构建（中文说明）

职责：
- 定义视觉骨干包装类 build_transformer（支持 ViT/CLIP/T2T 等）
- 定义整体模型 MambaPro（多模态 RGB/NI/TI 特征提取与融合，支持 AAM/Mamba 分支）
- 提供 make_model 工厂函数按配置实例化模型

要点：
- 通过 cfg 切换是否使用 CLIP、相机/视角嵌入（SIE）、LoRA 冻结等
- 训练返回多头 logits/特征以支持多损失；测试返回拼接或融合特征
"""
import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from timm.models.layers import trunc_normal_
from modeling.make_model_clipreid import load_clip_to_cpu
from modeling.clip.LoRA import mark_only_lora_as_trainable as lora_train
from modeling.fusion_part.AAM import AAM


def weights_init_kaiming(m):  # 定义一个函数，用 Kaiming 初始化方法对模型层进行初始化
    classname = m.__class__.__name__  # 获取当前层的类名（如 'Linear'、'Conv2d'、'BatchNorm2d' 等）
    
    if classname.find('Linear') != -1:  # 如果是全连接层（Linear）
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')  # 使用 Kaiming 正态分布初始化权重，适合 ReLU 激活
        nn.init.constant_(m.bias, 0.0)  # 将偏置初始化为 0

    elif classname.find('Conv') != -1:  # 如果是卷积层（Conv）
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')  # 使用 Kaiming 正态分布初始化卷积核权重
        if m.bias is not None:  # 如果卷积层有偏置项
            nn.init.constant_(m.bias, 0.0)  # 将偏置初始化为 0

    elif classname.find('BatchNorm') != -1:  # 如果是批归一化层（BatchNorm）
        if m.affine:  # 如果 BatchNorm 层有可学习参数（weight 和 bias）
            nn.init.constant_(m.weight, 1.0)  # 将缩放因子 gamma 初始化为 1
            nn.init.constant_(m.bias, 0.0)   # 将平移因子 beta 初始化为 0


def weights_init_classifier(m):  # 定义一个函数，用于初始化分类器层（通常是最后一层 Linear）
    classname = m.__class__.__name__  # 获取层的类名
    
    if classname.find('Linear') != -1:  # 如果是全连接层
        nn.init.normal_(m.weight, std=0.001)  # 使用均值为 0，标准差为 0.001 的正态分布初始化权重
        if m.bias:  # 如果存在偏置
            nn.init.constant_(m.bias, 0.0)  # 将偏置初始化为 0



class build_transformer(nn.Module):  # 视觉骨干封装（兼容 ViT/CLIP/T2T 等）
    def __init__(self, num_classes, cfg, camera_num, view_num, factory,feat_dim):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T  # 预训练权重路径（ImageNet/自定义）
        self.in_planes = feat_dim  # 特征维度（线性分类器/BNNeck输入）
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA  # 是否启用相机/视角嵌入
        self.neck = cfg.MODEL.NECK  # 颈部结构类型（如 bnneck）
        self.neck_feat = cfg.TEST.NECK_FEAT  # 测试阶段返回 neck 前/后特征
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE  # 骨干类型名
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE  # 同上
        self.flops_test = cfg.MODEL.FLOPS_TEST  # FLOPs 测试标志
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))  # 打印骨干类型

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num  # 相机数量（用于 SIE）
        else:
            self.camera_num = 0
        # No view
        self.view_num = 0  # 视角数此处固定为0（如需可扩展）
        
        # 🔥 新增：CLIP多尺度滑动窗口配置
        # 功能：从配置文件读取CLIP多尺度滑动窗口设置
        # 默认值：False（不启用多尺度处理）
        self.use_clip_multi_scale = getattr(cfg.MODEL, 'USE_CLIP_MULTI_SCALE', False)
        
        if cfg.MODEL.TRANSFORMER_TYPE == 'vit_base_patch16_224':
            # 标准ViT分支（保持原有功能）
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                            num_classes=num_classes,
                                                            camera=self.camera_num, view=self.view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                            cfg = cfg)  # 从工厂构建 ViT
            self.clip = 0  # 标记非 CLIP 分支
            self.base.load_param(model_path)  # 加载 ImageNet 预训练
            print('Loading pretrained model from ImageNet')  # 提示信息
            if cfg.MODEL.FROZEN:
                lora_train(self.base)  # 仅训练 LoRA 参数（其余冻结）
        elif cfg.MODEL.TRANSFORMER_TYPE == 't2t_vit_t_24':
            # 新增：T2T-ViT-24模型处理
            # 功能：创建T2T-ViT-24模型，支持多尺度滑动窗口
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate=cfg.MODEL.DROP_RATE,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                camera=self.camera_num,
                view=self.view_num,
                sie_xishu=cfg.MODEL.SIE_COE,
                use_multi_scale=self.use_multi_scale  # 传递多尺度参数
            )
            self.clip = 0  # 标记非 CLIP 分支
            self.base.load_param(model_path)  # 加载预训练权重
            print('Loading pretrained T2T-ViT-24 model')  # 提示信息
            if cfg.MODEL.FROZEN:
                lora_train(self.base)  # 仅训练 LoRA 参数（其余冻结）
        elif cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            # 恢复原作者的设计：ViT-B-16走CLIP分支
            # 功能：保持原作者的CLIP实现，并添加多尺度滑动窗口支持
            self.clip = 1  # 标记走 CLIP 分支
            self.sie_xishu = cfg.MODEL.SIE_COE  # SIE 系数
            clip_model = load_clip_to_cpu(cfg, self.model_name, cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                                          cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                                          cfg.MODEL.STRIDE_SIZE)  # 加载 CLIP 模型
            print('Loading pretrained model from CLIP')  # 提示信息
            clip_model.to("cuda")  # 将 CLIP 模型移至 GPU
            self.base = clip_model.visual  # 使用视觉编码器作为骨干
            if cfg.MODEL.FROZEN:
                lora_train(self.base)  # 仅训练 LoRA

            # 🔥 新增：CLIP多尺度滑动窗口初始化
            # 功能：在CLIP分支基础上添加多尺度滑动窗口特征提取
            if self.use_clip_multi_scale:
                from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor
                # 初始化多尺度特征提取器：512维输入，4x4/8x8/16x16滑动窗口
                self.clip_multi_scale_extractor = CLIPMultiScaleFeatureExtractor(feat_dim=512, scales=[4, 8, 16])
                print('✅ 为CLIP启用多尺度滑动窗口特征提取模块')
                print(f'   - 滑动窗口尺度: [4, 8, 16]')
                print(f'   - 特征维度: 512 (CLIP投影维度)')
            
            # 🔥 新增：多尺度MoE配置和初始化
            # 功能：从配置文件读取MoE设置，初始化MoE模块
            self.use_multi_scale_moe = getattr(cfg.MODEL, 'USE_MULTI_SCALE_MOE', False)
            self.moe_scales = getattr(cfg.MODEL, 'MOE_SCALES', [4, 8, 16])
            
            if self.use_multi_scale_moe:
                from modeling.fusion_part.multi_scale_moe import CLIPMultiScaleMoE
                # 初始化多尺度MoE模块：512维输入，4x4/8x8/16x16滑动窗口，专家网络
                self.clip_multi_scale_moe = CLIPMultiScaleMoE(
                    feat_dim=512, 
                    scales=self.moe_scales,
                    expert_hidden_dim=1024,
                    temperature=1.0
                )
                # 初始化专家权重历史记录（用于分析）
                self.expert_weights_history = []
                print('🔥 为CLIP启用多尺度MoE特征融合模块')
                print(f'   - 滑动窗口尺度: {self.moe_scales}')
                print(f'   - 特征维度: 512 (CLIP投影维度)')
                print(f'   - 专家隐藏层维度: 1024')
                print(f'   - 门控网络温度: 1.0')

            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 768))  # 相机×视角嵌入（CLIP实际维度）
                trunc_normal_(self.cv_embed, std=.02)  # 截断正态初始化
                print('camera number is : {}'.format(camera_num))  # 打印相机数
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 768))  # 仅相机嵌入（CLIP实际维度）
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 768))  # 仅视角嵌入（CLIP实际维度）
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)  # 线性分类头
        self.classifier.apply(weights_init_classifier)  # 分类头初始化

        self.bottleneck = nn.BatchNorm1d(self.in_planes)  # BNNeck
        self.bottleneck.bias.requires_grad_(False)  # 冻结偏置
        self.bottleneck.apply(weights_init_kaiming)  # BN 初始化

    def forward(self, x, label=None, cam_label=None, view_label=None, modality=None):
        if self.clip == 0:
            x = self.base(x, cam_label=cam_label, view_label=view_label, modality=modality)  # ViT/T2T 前向
                
        else:
            # CLIP分支 - 保持原有逻辑
            if self.cv_embed_sign:
                if self.flops_test:
                    cam_label = 0  # FLOPs 测试时统一相机索引
                cv_embed = self.sie_xishu * self.cv_embed[cam_label]  # 取相机/视角嵌入
            else:
                cv_embed = None  # 不使用嵌入
            x = self.base(x, cv_embed, modality)  # CLIP 前向
            
            # 🔥 新增：CLIP多尺度滑动窗口处理
            # 功能：在CLIP特征提取后，添加多尺度滑动窗口处理
            # 处理流程：CLIP输出 → 分离tokens → 多尺度处理 → 特征融合 → 重新组合
            if hasattr(self, 'use_clip_multi_scale') and self.use_clip_multi_scale and hasattr(self, 'clip_multi_scale_extractor'):
                # 🔥 分离CLS token和patch tokens
                # CLIP输出格式：[CLS_token, patch_token1, patch_token2, ...]
                cls_token = x[:, 0:1, :]  # [B, 1, 512] - CLIP的CLS token
                patch_tokens = x[:, 1:, :]  # [B, N, 512] - CLIP的patch tokens
                
                # 🔥 检查是否使用MoE融合
                if hasattr(self, 'use_multi_scale_moe') and self.use_multi_scale_moe and hasattr(self, 'clip_multi_scale_moe'):
                    # 🔥 使用MoE融合多尺度特征
                    # 核心算法：4x4/8x8/16x16滑动窗口 → MoE专家网络 → 动态权重融合
                    multi_scale_feature, expert_weights = self.clip_multi_scale_moe(patch_tokens)  # [B, 512], [B, 3]
                    
                    # 🔥 MoE融合完成提示（仅在第一次调用时显示）
                    if not hasattr(self, '_moe_fusion_called'):
                        print(f"✅ MoE多尺度特征融合完成！")
                        print(f"   - 输出特征形状: {multi_scale_feature.shape}")
                        print(f"   - 专家权重形状: {expert_weights.shape}")
                        print(f"   - 专家权重分布: {expert_weights[0].detach().cpu().numpy()}")
                        self._moe_fusion_called = True
                    
                    # 保存专家权重用于分析（可选）
                    if hasattr(self, 'expert_weights_history'):
                        self.expert_weights_history.append(expert_weights.detach().cpu())
                else:
                    # 🔥 使用传统MLP融合多尺度特征
                    # 核心算法：4x4/8x8/16x16滑动窗口 → MLP特征融合
                    multi_scale_feature = self.clip_multi_scale_extractor(patch_tokens)  # [B, 512]
                    
                    # 🔥 滑动窗口融合完成提示（仅在第一次调用时显示）
                    if not hasattr(self, '_sliding_window_fusion_called'):
                        print(f"✅ 多尺度滑动窗口特征融合完成！")
                        print(f"   - 输出特征形状: {multi_scale_feature.shape}")
                        self._sliding_window_fusion_called = True
                
                # 🔥 将多尺度特征与CLS token结合（残差连接）
                # 增强CLS token：原始CLS + 多尺度特征
                enhanced_cls = cls_token + multi_scale_feature.unsqueeze(1)  # [B, 1, 512]
                
                # 🔥 重新组合tokens：增强的CLS token + 原始patch tokens
                # 保持原始序列结构，但CLS token被多尺度特征增强
                x = torch.cat([enhanced_cls, patch_tokens], dim=1)  # [B, N+1, 512]

        global_feat = x[:, 0]  # 取CLS token 作为全局特征
        feat = self.bottleneck(global_feat)  # 过 BNNeck（训练常用）

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)  # 特殊 margin 类头（需要 label）
            else:
                cls_score = self.classifier(feat)  # 普通线性分类
            return x, cls_score, global_feat  # 返回缓存、分类分数、全局特征
        else:
            if self.neck_feat == 'after':
                return x, feat  # 测试返回 BN 后特征
            else:
                return x, global_feat  # 测试返回 BN 前特征

    def load_param(self, trained_path):  # 从权重文件加载参数（兼容DP/DDP前缀）
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))  # 打印来源

    def load_param_finetune(self, model_path):  # 精调：严格按键拷贝
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class MambaPro(nn.Module):  # 三模态组装与融合 head
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(MambaPro, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768  # ViT 基本维度
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512  # CLIP ViT-B/16 维度
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory,feat_dim=self.feat_dim)  # 共享骨干
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE  # 每ID样本数（采样策略用）
        self.camera = camera_num  # 相机数
        self.view = view_num  # 视角数
        self.direct = cfg.MODEL.DIRECT  # 是否直接拼接分类
        self.neck = cfg.MODEL.NECK  # 颈部类型
        self.neck_feat = cfg.TEST.NECK_FEAT  # 测试特征选择
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE  # 分类头类型
        self.mamba = cfg.MODEL.MAMBA  # 是否启用 Mamba 融合
        
        # 使用原始AAM融合模块
        self.AAM = AAM(self.feat_dim, n_layers=2, cfg=cfg)
        self.miss_type = cfg.TEST.MISS  # 测试缺失模态策略
        self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)  # 原始三模态拼接分类头
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)  # 原始拼接 BNNeck
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier_fuse = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)  # 融合特征分类头
        self.classifier_fuse.apply(weights_init_classifier)
        self.bottleneck_fuse = nn.BatchNorm1d(3 * self.feat_dim)  # 融合 BNNeck
        self.bottleneck_fuse.bias.requires_grad_(False)
        self.bottleneck_fuse.apply(weights_init_kaiming)

    def load_param(self, trained_path):  # 精确加载（不去掉 module 前缀）
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, label=None, cam_label=None, view_label=None):  # 训练/测试两条路径
        if self.training:
            RGB = x['RGB']  # 可见光
            NI = x['NI']  # 近红外
            TI = x['TI']  # 热红外

            RGB_cash, RGB_score, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label,
                                                            modality='rgb')
            NI_cash, NI_score, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label, modality='nir')
            TI_cash, TI_score, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label, modality='tir')

            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)  # 三模态拼接
            ori_global = self.bottleneck(ori)  # BNNeck
            ori_score = self.classifier(ori_global)  # 原始拼接分类

            if self.mamba:
                fuse = self.AAM(RGB_cash, NI_cash, TI_cash)  # 融合序列（如 Mamba）
                fuse_global = self.bottleneck_fuse(fuse)  # BNNeck 融合
                fuse_score = self.classifier_fuse(fuse_global)  # 融合分类

            if self.direct:  # 直接输出拼接/融合用于分类（简化 heads）
                if self.mamba:
                    return ori_score, ori, fuse_score, fuse  # 原始与融合并行输出
                else:
                    return ori_score, ori 
            else:
                if self.mamba: 
                    return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, fuse_score, fuse  # 多头多尺度损失
                else:
                    return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']  # 测试路径
            NI = x['NI']    
            TI = x['TI']
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label, modality='rgb')
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label, modality='nir')
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label, modality='tir')

            if self.mamba:
                fuse = self.AAM(RGB_cash, NI_cash, TI_cash)  # 输出融合特征
                return fuse
            else:
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)  # 输出拼接特征
                return ori

# 作用：把人类好记的字符串名字，翻译成代码里真正可调用的模型构造函数
__factory_T_type = {  # 骨干工厂映射
    'vit_base_patch16_224': vit_base_patch16_224, 
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):  # 模型工厂
    model = MambaPro(num_class, cfg, camera_num, view_num, __factory_T_type)  # 实例化 MambaPro
    print('===========Building MambaPro===========')  # 构建提示
    return model  # 返回模型
