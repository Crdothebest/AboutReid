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
        # 原代码：
        # self.neck = cfg.MODEL.NECK  # 颈部结构类型（如 bnneck）

        # 修改为：
        self.neck = getattr(cfg.MODEL, 'NECK', 'bnneck')  # 默认使用bnneck
        self.neck_feat = cfg.TEST.NECK_FEAT  # 测试阶段返回 neck 前/后特征
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE  # 骨干类型名
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE  # 同上
        self.flops_test = cfg.MODEL.FLOPS_TEST  # FLOPs 测试标志

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
        
        # 🔥 新增：T2T-ViT多尺度滑动窗口配置
        # 功能：从配置文件读取T2T-ViT多尺度滑动窗口设置
        # 默认值：False（不启用多尺度处理）
        self.use_multi_scale = getattr(cfg.MODEL, 'USE_MULTI_SCALE', False)
        
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
            if cfg.MODEL.FROZEN:
                lora_train(self.base)  # 仅训练 LoRA 参数（其余冻结）
        elif cfg.MODEL.TRANSFORMER_TYPE == 't2t_vit_t_24':
            # 新增：T2T-ViT-24模型处理
            # 功能：创建T2T-ViT-24模型，支持多尺度滑动窗口
            # 使用getattr获取配置参数，如果不存在则使用默认值
            drop_path_rate = getattr(cfg.MODEL, 'DROP_PATH', 0.1)  # 默认0.1
            drop_rate = getattr(cfg.MODEL, 'DROP_RATE', 0.0)  # 默认0.0
            attn_drop_rate = getattr(cfg.MODEL, 'ATT_DROP_RATE', 0.0)  # 默认0.0
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=drop_path_rate,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                camera=self.camera_num,
                view=self.view_num,
                sie_xishu=cfg.MODEL.SIE_COE,
                use_multi_scale=self.use_multi_scale  # 传递多尺度参数
            )
            self.clip = 0  # 标记非 CLIP 分支
            self.base.load_param(model_path)  # 加载预训练权重
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
            clip_model.to("cuda")  # 将 CLIP 模型移至 GPU
            self.base = clip_model.visual  # 使用视觉编码器作为骨干
            if cfg.MODEL.FROZEN:
                lora_train(self.base)  # 仅训练 LoRA

            # 🔥 新增：CLIP多尺度滑动窗口初始化
            # 功能：在CLIP分支基础上添加多尺度滑动窗口特征提取
            if self.use_clip_multi_scale:
                from modeling.fusion_part.clip_multi_scale_sliding_window import CLIPMultiScaleFeatureExtractor
                clip_scales = getattr(cfg.MODEL, 'CLIP_MULTI_SCALE_SCALES', [4, 8, 16])
                self.clip_multi_scale_extractor = CLIPMultiScaleFeatureExtractor(feat_dim=512, scales=clip_scales)
            
            # 🔥 新增：多尺度MoE配置和初始化
            # 功能：从配置文件读取MoE设置，初始化MoE模块
            self.use_multi_scale_moe = getattr(cfg.MODEL, 'USE_MULTI_SCALE_MOE', False)
            self.moe_scales = getattr(cfg.MODEL, 'MOE_SCALES', [4, 8, 16])
            
            # 门控融合配置
            use_gate_fusion_raw = getattr(cfg.MODEL, 'USE_GATE_FUSION', False)
            if isinstance(use_gate_fusion_raw, str):
                self.use_gate_fusion = use_gate_fusion_raw.lower() in ('true', '1', 'yes')
            else:
                self.use_gate_fusion = bool(use_gate_fusion_raw)
            self.gate_dropout = getattr(cfg.MODEL, 'GATE_DROPOUT', 0.1)
            
            # 注意力融合配置
            use_attention_fusion_raw = getattr(cfg.MODEL, 'USE_ATTENTION_FUSION', False)
            if isinstance(use_attention_fusion_raw, str):
                self.use_attention_fusion = use_attention_fusion_raw.lower() in ('true', '1', 'yes')
            else:
                self.use_attention_fusion = bool(use_attention_fusion_raw)
            
            self.attention_num_heads = getattr(cfg.MODEL, 'ATTENTION_NUM_HEADS', 8)
            self.attention_dropout = getattr(cfg.MODEL, 'ATTENTION_DROPOUT', 0.1)
            self.attention_dim = getattr(cfg.MODEL, 'ATTENTION_DIM', 512)
            
            if self.use_multi_scale_moe:
                from modeling.fusion_part.multi_scale_moe import CLIPMultiScaleMoE
                # 🔥 修复：从配置文件读取所有MoE参数，替代硬编码
                expert_hidden_dim = getattr(cfg.MODEL, 'MOE_EXPERT_HIDDEN_DIM', 1024)
                temperature = getattr(cfg.MODEL, 'MOE_TEMPERATURE', 1.0)
                expert_dropout = getattr(cfg.MODEL, 'MOE_EXPERT_DROPOUT', 0.1)
                gate_dropout = getattr(cfg.MODEL, 'MOE_GATE_DROPOUT', 0.1)
                expert_layers = getattr(cfg.MODEL, 'MOE_EXPERT_LAYERS', 2)
                gate_layers = getattr(cfg.MODEL, 'MOE_GATE_LAYERS', 2)
                expert_threshold = getattr(cfg.MODEL, 'MOE_EXPERT_THRESHOLD', 0.1)
                residual_weight = getattr(cfg.MODEL, 'MOE_RESIDUAL_WEIGHT', 1.0)
                init_weights = getattr(cfg.MODEL, 'MOE_INIT_WEIGHTS', None)
                
                # ========== 固定权重模式参数读取 ==========
                # 功能：从配置文件读取固定权重相关参数
                #
                # MOE_USE_FIXED_WEIGHTS:
                #   - 类型：bool
                #   - 默认值：False（使用动态门控网络）
                #   - 说明：控制是否使用固定权重模式
                #   - 命令行示例：MODEL.MOE_USE_FIXED_WEIGHTS True
                #
                # MOE_FIXED_WEIGHTS:
                #   - 类型：list of float
                #   - 默认值：[0.33, 0.33, 0.34]（三个专家均等权重）
                #   - 说明：固定权重值，仅在USE_FIXED_WEIGHTS=True时生效
                #   - 格式要求：
                #     * 长度必须等于专家数量（MOE_NUM_EXPERTS）
                #     * 权重会自动归一化，无需手动确保和为1.0
                #     * 示例：[0.33, 0.33, 0.34] 或 [0.5, 0.3, 0.2]
                #   - 命令行示例：MODEL.MOE_FIXED_WEIGHTS "[0.33,0.33,0.34]"
                #
                # 使用场景：
                #   1. 调试实验：固定权重可以排除门控网络影响，专注于专家网络性能
                #   2. 性能对比：对比固定权重 vs 动态权重的效果差异
                #   3. 跨域鲁棒性：固定权重可能在跨域场景下更稳定
                #
                # 注意事项：
                #   - 当USE_FIXED_WEIGHTS=True时，门控网络将被禁用，不参与训练
                #   - 固定权重不会随训练改变，始终保持预设值
                #   - 建议同时禁用MoE辅助Loss（BALANCE_LOSS_WEIGHT=0.0等）
                #
                # 固定权重模式参数
                if hasattr(cfg.MODEL, 'MOE_USE_FIXED_WEIGHTS'):
                    use_fixed_weights_raw = cfg.MODEL.MOE_USE_FIXED_WEIGHTS
                    if isinstance(use_fixed_weights_raw, str):
                        use_fixed_weights = use_fixed_weights_raw.lower() in ('true', '1', 'yes')
                    else:
                        use_fixed_weights = bool(use_fixed_weights_raw)
                else:
                    use_fixed_weights = False
                
                if hasattr(cfg.MODEL, 'MOE_FIXED_WEIGHTS'):
                    fixed_weights_raw = cfg.MODEL.MOE_FIXED_WEIGHTS
                    if isinstance(fixed_weights_raw, str):
                        import ast
                        try:
                            fixed_weights = ast.literal_eval(fixed_weights_raw)
                        except:
                            fixed_weights = [0.33, 0.33, 0.34]
                    else:
                        fixed_weights = fixed_weights_raw
                else:
                    fixed_weights = [0.33, 0.33, 0.34]
                
                # Top-k 路由参数
                use_top_k_routing = getattr(cfg.MODEL, 'MOE_USE_TOP_K_ROUTING', False)
                top_k = getattr(cfg.MODEL, 'MOE_TOP_K', 2)
                top_k_mode = getattr(cfg.MODEL, 'MOE_TOP_K_MODE', 'soft')
                
                if isinstance(use_top_k_routing, str):
                    use_top_k_routing = use_top_k_routing.lower() in ['true', '1', 'yes']
                else:
                    use_top_k_routing = bool(use_top_k_routing)
                
                if isinstance(top_k_mode, str):
                    top_k_mode = top_k_mode.lower()
                    if top_k_mode not in ['soft', 'hard']:
                        top_k_mode = 'soft'
                
                # 初始化多尺度MoE模块：使用所有配置参数
                self.clip_multi_scale_moe = CLIPMultiScaleMoE(
                    feat_dim=512, 
                    scales=self.moe_scales,
                    expert_hidden_dim=expert_hidden_dim,
                    temperature=temperature,
                    expert_dropout=expert_dropout,
                    gate_dropout=gate_dropout,
                    expert_layers=expert_layers,
                    gate_layers=gate_layers,
                    expert_threshold=expert_threshold,
                    residual_weight=residual_weight,
                    use_gate_fusion=self.use_gate_fusion,
                    use_attention_fusion=self.use_attention_fusion,
                    attention_num_heads=self.attention_num_heads,
                    attention_dropout=self.attention_dropout,
                    attention_dim=self.attention_dim,
                    init_weights=init_weights,
                    use_fixed_weights=use_fixed_weights,
                    fixed_weights=fixed_weights,
                    use_top_k_routing=use_top_k_routing,
                    top_k=top_k,
                    top_k_mode=top_k_mode
                )
                # 初始化专家权重历史记录（用于分析）
                self.expert_weights_history = []

            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 768))  # 相机×视角嵌入（CLIP实际维度）
                trunc_normal_(self.cv_embed, std=.02)  # 截断正态初始化
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 768))  # 仅相机嵌入（CLIP实际维度）
                trunc_normal_(self.cv_embed, std=.02)
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 768))  # 仅视角嵌入（CLIP实际维度）
                trunc_normal_(self.cv_embed, std=.02)

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
                    # 🎯 Step 4: 传递epoch给MoE模块，激活温度调度
                    multi_scale_feature, expert_weights = self.clip_multi_scale_moe(patch_tokens)  # [B, 512], [B, 3]
                    
                    # 保存专家权重用于分析（可选）
                    if hasattr(self, 'expert_weights_history'):
                        self.expert_weights_history.append(expert_weights.detach().cpu())
                    
                    # 🔥 保存专家权重用于MoE损失计算
                    # 注意：必须保留梯度，否则MoE损失无法反向传播更新门控网络
                    self.current_expert_weights = expert_weights  # 保留梯度，不detach
                else:
                    # 🔥 使用传统MLP融合多尺度特征
                    # 核心算法：4x4/8x8/16x16滑动窗口 → MLP特征融合
                    multi_scale_feature = self.clip_multi_scale_extractor(patch_tokens)  # [B, 512]
                
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
        """
        加载预训练权重，兼容 DataParallel/DistributedDataParallel 前缀
        """
        param_dict = torch.load(trained_path, map_location='cpu')
        
        # 适配不同的权重字典格式
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'state_dict_ema' in param_dict:
            param_dict = param_dict['state_dict_ema']
        
        loaded_params = 0
        skipped_params = 0
        
        for i in param_dict:
            # 移除 'module.' 前缀（兼容 DP/DDP）
            key = i.replace('module.', '')
            
            # 检查参数是否存在于当前模型中
            if key not in self.state_dict():
                skipped_params += 1
                continue
            
            # 检查尺寸是否匹配
            param_shape = param_dict[i].shape
            model_shape = self.state_dict()[key].shape
            
            if param_shape != model_shape:
                skipped_params += 1
                continue
            
            # 尺寸匹配，加载参数
            try:
                self.state_dict()[key].copy_(param_dict[i])
                loaded_params += 1
            except Exception as e:
                print(f"❌ 加载参数失败: {key}, 错误: {e}")
                skipped_params += 1
        
        print(f"✅ 参数加载完成: 成功加载 {loaded_params} 个参数, 跳过 {skipped_params} 个参数")

    def load_param_finetune(self, model_path):  # 精调：严格按键拷贝
        """
        精调模式：严格按键拷贝，不处理前缀
        """
        param_dict = torch.load(model_path, map_location='cpu')
        
        # 适配不同的权重字典格式
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'state_dict_ema' in param_dict:
            param_dict = param_dict['state_dict_ema']
        
        loaded_params = 0
        skipped_params = 0
        
        for i in param_dict:
            # 检查参数是否存在于当前模型中
            if i not in self.state_dict():
                skipped_params += 1
                continue
            
            # 检查尺寸是否匹配
            param_shape = param_dict[i].shape
            model_shape = self.state_dict()[i].shape
            
            if param_shape != model_shape:
                skipped_params += 1
                continue
            
            # 尺寸匹配，加载参数
            try:
                self.state_dict()[i].copy_(param_dict[i])
                loaded_params += 1
            except Exception as e:
                print(f"❌ 加载参数失败: {i}, 错误: {e}")
                skipped_params += 1
        
        print(f"✅ 参数加载完成: 成功加载 {loaded_params} 个参数, 跳过 {skipped_params} 个参数")


class MambaPro(nn.Module):  # 三模态组装与融合 head
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(MambaPro, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768  # ViT 基本维度
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512  # CLIP ViT-B/16 维度
        elif 't2t_vit_t_24' in cfg.MODEL.TRANSFORMER_TYPE or 't2t_vit_t_14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512  # T2T-ViT 维度（embed_dim=512）
        else:
            # 默认值，如果都不匹配则使用512
            self.feat_dim = 512
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
        """
        加载预训练权重并进行必要适配
        
        功能：
        - 从指定路径加载预训练模型权重
        - 处理不同权重字典格式的兼容性（model/state_dict/state_dict_ema）
        - 适配参数尺寸不匹配的情况
        - 跳过不存在的参数
        
        参数：
        - trained_path: 预训练权重文件路径
        """
        param_dict = torch.load(trained_path, map_location='cpu')
        
        # 适配不同的权重字典格式
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'state_dict_ema' in param_dict:
            param_dict = param_dict['state_dict_ema']
        
        loaded_params = 0
        skipped_params = 0
        size_mismatch_params = []
        
        for i in param_dict:
            # 检查参数是否存在于当前模型中
            if i not in self.state_dict():
                skipped_params += 1
                continue
            
            # 检查尺寸是否匹配
            param_shape = param_dict[i].shape
            model_shape = self.state_dict()[i].shape
            
            if param_shape != model_shape:
                size_mismatch_params.append((i, param_shape, model_shape))
                skipped_params += 1
                continue
            
            # 尺寸匹配，加载参数
            try:
                self.state_dict()[i].copy_(param_dict[i])
                loaded_params += 1
            except Exception as e:
                print(f"❌ 加载参数失败: {i}, 错误: {e}")
                skipped_params += 1
        
        # 打印加载统计信息
        print(f"✅ 参数加载完成: 成功加载 {loaded_params} 个参数, 跳过 {skipped_params} 个参数")
        
        # 如果有尺寸不匹配的参数，打印详细信息
        if size_mismatch_params:
            print(f"\n⚠️  发现 {len(size_mismatch_params)} 个尺寸不匹配的参数:")
            for param_name, param_shape, model_shape in size_mismatch_params[:10]:  # 只显示前10个
                print(f"   - {param_name}: 权重文件 {param_shape} vs 模型期望 {model_shape}")
            if len(size_mismatch_params) > 10:
                print(f"   ... 还有 {len(size_mismatch_params) - 10} 个参数未显示")

    def forward(self, x, label=None, cam_label=None, view_label=None):  # 训练/测试两条路径
        if self.training:
            RGB = x['RGB']  # 可见光
            NI = x['NI']  # 近红外
            TI = x['TI']  # 热红外

            RGB_tokens, RGB_score, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label,
                                                            modality='rgb')
            NI_tokens, NI_score, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label, modality='nir')
            TI_tokens, TI_score, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label, modality='tir')
            
            # 为了保持兼容性，将tokens作为cash使用
            RGB_cash = RGB_tokens
            NI_cash = NI_tokens
            TI_cash = TI_tokens

            # 🔥 关键修改：检测数据集类型，支持双模态和三模态
            # 通过检查TI特征是否与NI特征相同来判断是否为双模态数据集
            is_dual_modal = torch.allclose(NI_global, TI_global, atol=1e-6)
            
            if is_dual_modal:
                # 🔥 RGBNT100双模态数据集：只使用RGB和IR（NI），忽略TI
                ori = torch.cat([RGB_global, NI_global], dim=-1)  # 双模态拼接
                # 调整bottleneck和classifier的输入维度
                ori_global = self.bottleneck(ori)  # BNNeck
                ori_score = self.classifier(ori_global)  # 原始拼接分类
                
                if self.mamba:
                    # 双模态融合：只使用RGB和IR特征
                    fuse = self.AAM(RGB_cash, NI_cash, None)  # 传入None作为TI
                    fuse_global = self.bottleneck_fuse(fuse)  # BNNeck 融合
                    fuse_score = self.classifier_fuse(fuse_global)  # 融合分类
            else:
                # 🔥 RGBNT201三模态数据集：使用RGB、NI、TI
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
                    if is_dual_modal:
                        # 双模态：只返回RGB和IR的特征
                        return RGB_score, RGB_global, NI_score, NI_global, fuse_score, fuse
                    else:
                        # 三模态：返回所有特征
                        return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, fuse_score, fuse
                else:
                    if is_dual_modal:
                        # 双模态：只返回RGB和IR的特征
                        return RGB_score, RGB_global, NI_score, NI_global
                    else:
                        # 三模态：返回所有特征
                        return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']  # 测试路径
            NI = x['NI']    
            TI = x['TI']
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label, modality='rgb')
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label, modality='nir')
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label, modality='tir')

            # 🔥 测试时也检测数据集类型
            is_dual_modal = torch.allclose(NI_global, TI_global, atol=1e-6)
            
            if self.mamba:
                if is_dual_modal:
                    # 双模态融合
                    fuse = self.AAM(RGB_cash, NI_cash, None)  # 传入None作为TI
                    return fuse
                else:
                    # 三模态融合
                    fuse = self.AAM(RGB_cash, NI_cash, TI_cash)  # 输出融合特征
                    return fuse
            else:
                if is_dual_modal:
                    # 双模态拼接
                    ori = torch.cat([RGB_global, NI_global], dim=-1)  # 输出拼接特征
                    return ori
                else:
                    # 三模态拼接
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
    return model  # 返回模型
