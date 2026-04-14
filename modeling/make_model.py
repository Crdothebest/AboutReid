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
            self._clip_model = clip_model  # 保留 CLIP 模型引用，供 IDEA02 文本编码使用（encode_text 用 no_grad 调用，无需冻结）

            # 获取CLIP模型的实际vision_width（通常是768而不是512）
            # VisionTransformer没有直接的width属性，需要从transformer中获取
            self.vision_width = self.base.transformer.width

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

            # SIE (Spatial Identity Embedding) 初始化
            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                # 相机和视角都启用：创建 camera_num * view_num 个嵌入
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.vision_width))
                trunc_normal_(self.cv_embed, std=.02)
            elif cfg.MODEL.SIE_CAMERA:
                # 仅相机启用：创建 camera_num 个嵌入
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.vision_width))
                trunc_normal_(self.cv_embed, std=.02)
            elif cfg.MODEL.SIE_VIEW:
                # 仅视角启用：创建 view_num 个嵌入
                self.cv_embed = nn.Parameter(torch.zeros(view_num, self.vision_width))
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

        if loaded_params > 0:
            print("=" * 80)
            print(f"🎉 ✅ 预训练权重加载成功: 成功加载 {loaded_params} 个参数, 跳过 {skipped_params} 个参数")
            print(f"📁 预训练权重路径: {trained_path}")
            print("=" * 80)
        else:
            print("=" * 80)
            print(f"⚠️ 参数加载完成: 成功加载 {loaded_params} 个参数, 跳过 {skipped_params} 个参数")
            print("=" * 80)

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
        
        # ============ 文本融合配置 ============
        self.use_text_fusion = getattr(cfg.MODEL, 'USE_TEXT_FUSION', False)
        self.text_fusion_method = getattr(cfg.MODEL, 'TEXT_FUSION_METHOD', 'attention')
        self.text_fusion_weight = getattr(cfg.MODEL, 'TEXT_FUSION_WEIGHT', 0.3)

        # 读取文本融合维度配置
        self.text_fusion_embed_dim = getattr(cfg.MODEL, 'TEXT_FUSION_EMBED_DIM', self.feat_dim)
        self.text_fusion_input_dim = getattr(cfg.MODEL, 'TEXT_FUSION_INPUT_DIM', self.feat_dim * 3)
        self.text_fusion_text_dim = getattr(cfg.MODEL, 'TEXT_FUSION_TEXT_DIM', self.feat_dim)

        # ============ 模态内引导配置 ============
        self.use_modal_guidance = getattr(cfg.MODEL, 'USE_MODAL_GUIDANCE', True)  # 默认启用模态内引导
        self.guidance_residual = getattr(cfg.MODEL, 'GUIDANCE_RESIDUAL', True)   # 使用残差结构避免特征丢失
        self.guidance_scale = getattr(cfg.MODEL, 'GUIDANCE_SCALE', 0.1)          # 引导增强幅度

        # 如果启用文本融合，创建融合模块
        if self.use_text_fusion:
            from .fusion_part.cross_modal_attention import create_text_fusion_module
            self.text_fusion = create_text_fusion_module(
                method=self.text_fusion_method,
                embed_dim=self.text_fusion_embed_dim,     # 从配置读取
                input_dim=self.text_fusion_input_dim,     # 从配置读取
                text_dim=self.text_fusion_text_dim,       # 从配置读取
            )
            print(f"✅ MambaPro已启用文本融合: {self.text_fusion_method}模式 (embed_dim: {self.text_fusion_embed_dim})")
            # 预创建 residual 模式所需的文本适配器，确保参数被 optimizer 注册、设备一致
            if self.text_fusion_method == "residual":
                _half = self.text_fusion_input_dim // 2
                self.text_adapters = nn.ModuleDict({
                    'RGB': nn.Sequential(
                        nn.Linear(512, _half), nn.GELU(),
                        nn.Linear(_half, self.text_fusion_input_dim),
                        nn.LayerNorm(self.text_fusion_input_dim)
                    ),
                    'NIR': nn.Sequential(
                        nn.Linear(512, _half), nn.GELU(),
                        nn.Linear(_half, self.text_fusion_input_dim),
                        nn.LayerNorm(self.text_fusion_input_dim)
                    ),
                    'TIR': nn.Sequential(
                        nn.Linear(512, _half), nn.GELU(),
                        nn.Linear(_half, self.text_fusion_input_dim),
                        nn.LayerNorm(self.text_fusion_input_dim)
                    ),
                })
            else:
                self.text_adapters = None
            # 预创建上采样投影层（确保参数被 optimizer 注册、设备一致）
            # attention/concat 方法输出 embed_dim(512)，需要投影回 1536
            if self.text_fusion_method in ("attention", "concat"):
                if self.text_fusion_embed_dim != 1536:
                    self.attention_upsampler = nn.Linear(self.text_fusion_embed_dim, 1536)
                else:
                    self.attention_upsampler = None
                self.concat_upsampler = self.attention_upsampler  # 同一个投影层复用
            else:
                self.attention_upsampler = None
                self.concat_upsampler = None
        else:
            self.text_fusion = None
            self.text_adapters = None
            self.attention_upsampler = None
            self.concat_upsampler = None

        # 如果启用模态内引导，创建门控网络
        if self.use_modal_guidance:
            self.modal_guidance = self._create_modal_guidance()
            print(f"✅ MambaPro已启用模态内引导: 残差结构防止特征丢失 (scale={self.guidance_scale})")
        else:
            self.modal_guidance = None

        # 使用原始AAM融合模块
        self.AAM = AAM(self.feat_dim, n_layers=2, cfg=cfg)
        self.miss_type = cfg.TEST.MISS  # 测试缺失模态策略
        self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)  # 原始三模态拼接分类头
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)  # 原始拼接 BNNeck
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # ============ 维度守恒设计：始终保持1536维输出 ============
        # 无论是否使用文本融合，输出维度始终为3 * feat_dim (1536)
        # 这样可以完美兼容预训练的BatchNorm和分类器权重
        output_dim = 3 * self.feat_dim  # 固定1536维

        self.classifier_fuse = nn.Linear(output_dim, self.num_classes, bias=False)  # 融合特征分类头
        self.classifier_fuse.apply(weights_init_classifier)
        self.bottleneck_fuse = nn.BatchNorm1d(output_dim)  # 融合 BNNeck (1536维)
        self.bottleneck_fuse.bias.requires_grad_(False)
        self.bottleneck_fuse.apply(weights_init_kaiming)

        print(f"✅ MambaPro输出维度: {output_dim} (维度守恒拼接，兼容预训练权重)")

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
        if loaded_params > 0:
            print("=" * 80)
            print(f"🎉 ✅ 预训练权重加载成功: 成功加载 {loaded_params} 个参数, 跳过 {skipped_params} 个参数")
            print(f"📁 预训练权重路径: {trained_path}")
            print("=" * 80)
        else:
            print("=" * 80)
            print(f"⚠️ 参数加载完成: 成功加载 {loaded_params} 个参数, 跳过 {skipped_params} 个参数")
            print("=" * 80)

        # 如果有尺寸不匹配的参数，打印详细信息
        if size_mismatch_params:
            print(f"\n⚠️  发现 {len(size_mismatch_params)} 个尺寸不匹配的参数:")
            for param_name, param_shape, model_shape in size_mismatch_params[:10]:  # 只显示前10个
                print(f"   - {param_name}: 权重文件 {param_shape} vs 模型期望 {model_shape}")
            if len(size_mismatch_params) > 10:
                print(f"   ... 还有 {len(size_mismatch_params) - 10} 个参数未显示")

    def _create_modal_guidance(self):
        """创建安全的模态内引导网络"""
        class SafeModalGuidance(nn.Module):
            """安全的模态内引导：残差结构避免特征丢失"""

            def __init__(self, feat_dim=512, text_dim=512, use_residual=True, scale_init=0.1):
                super().__init__()
                self.feat_dim = feat_dim
                self.use_residual = use_residual

                # 分布对齐层
                self.visual_norm = nn.LayerNorm(feat_dim)
                self.text_norm = nn.LayerNorm(text_dim)
                self.text_adapter = nn.Linear(text_dim, feat_dim)

                # 安全的门控网络
                self.gate_network = nn.Sequential(
                    nn.Linear(feat_dim * 2, feat_dim),
                    nn.LayerNorm(feat_dim),
                    nn.GELU(),
                    nn.Linear(feat_dim, feat_dim),
                    nn.Sigmoid()  # 输出[0,1]门控信号
                )

                # 增强幅度控制器 (可配置初始值)
                self.enhancement_scale = nn.Parameter(torch.tensor(scale_init))

            def forward(self, visual_feat, text_feat=None):
                """安全的模态内引导"""
                if text_feat is None:
                    return visual_feat

                # 支持 [B, seq, D] 和 [B, D] 两种输入：取 CLS token（第0个）做引导
                is_seq = visual_feat.dim() == 3
                if is_seq:
                    cls_feat = visual_feat[:, 0]  # [B, D]
                else:
                    cls_feat = visual_feat

                # 分布对齐（均基于 [B, D]）
                visual_normed = self.visual_norm(cls_feat)
                text_normed = self.text_norm(text_feat)
                text_aligned = self.text_adapter(text_normed)

                # 生成门控信号 [B, D]
                combined = torch.cat([visual_normed, text_aligned], dim=-1)
                guidance = self.gate_network(combined)

                if is_seq:
                    # 把 guidance 广播到整个序列
                    guidance = guidance.unsqueeze(1)  # [B, 1, D]

                if self.use_residual:
                    # 安全的残差增强：原始 + 增强
                    enhancement = visual_feat * guidance * self.enhancement_scale
                    enhanced_visual = visual_feat + enhancement
                else:
                    # 传统方式（有风险）
                    enhanced_visual = visual_feat * guidance

                # 数值稳定性保护
                enhanced_visual = torch.clamp(enhanced_visual, -10, 10)

                return enhanced_visual

        return SafeModalGuidance(
            feat_dim=self.feat_dim,
            text_dim=self.feat_dim,  # 假设文本维度与视觉一致
            use_residual=self.guidance_residual,
            scale_init=self.guidance_scale
        )

    def forward(self, x, label=None, cam_label=None, view_label=None, text_features=None):  # 训练/测试两条路径（固定三模态，与MambaPro一致）
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

            # ============ 文本-视觉对齐 Loss (InfoNCE) ============
            # 批内对比损失：同一行人的(视觉, 文本)为正样本，批内其他行人为负样本
            # 比逐样本cosine距离更强：利用批内所有负样本，强迫不同身份特征分开
            self.text_align_loss = None
            if text_features is not None and self.use_text_fusion:
                import torch.nn.functional as F

                def _info_nce(v, t, temperature=0.07):
                    B = v.size(0)
                    v = F.normalize(v, dim=-1)
                    t = F.normalize(t, dim=-1)
                    logits = v @ t.T / temperature          # [B, B]
                    labels = torch.arange(B, device=v.device)
                    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

                temperature = getattr(self.cfg.MODEL, 'TEXT_ALIGN_TEMPERATURE', 0.07)
                align_loss = 0.0
                count = 0
                for feat, key in [(RGB_global, 'RGB'), (NI_global, 'NIR'), (TI_global, 'TIR')]:
                    if key in text_features:
                        align_loss = align_loss + _info_nce(feat, text_features[key], temperature)
                        count += 1
                if count > 0:
                    self.text_align_loss = align_loss / count

            # 🔥 固定三模态拼接（与MambaPro完全一致，移除智能检测）
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)  # 三模态拼接 [B, 1536]
            ori_global = self.bottleneck(ori)  # BNNeck
            ori_score = self.classifier(ori_global)  # 原始拼接分类

            if self.mamba:
                fuse = self.AAM(RGB_cash, NI_cash, TI_cash)  # 三模态融合

                # ============ 文本融合 ============
                if self.use_text_fusion and self.text_fusion is not None and text_features is not None:
                    # 准备文本特征：分模态处理，避免语义稀释
                    text_rgb = text_features['RGB']  # [B, 512] 预编码向量
                    text_nir = text_features['NIR']  # [B, 512]
                    text_tir = text_features['TIR']  # [B, 512]

                    # 应用文本融合 - 分模态引导策略
                    if self.text_fusion_method == "residual":
                        # ============ 残差融合：文本投影到视觉维度 ============
                        # 将三个模态的文本特征分别投影到1536维，然后进行门控增强
                        original_fuse = fuse.clone()  # [B, 1536] 保存原始AAM融合结果

                        # 分别处理每个模态的文本引导（text_adapters 在 __init__ 中预创建）
                        rgb_modulator = self.text_adapters['RGB'](text_rgb)  # [B, 512] -> [B, 1536]
                        nir_modulator = self.text_adapters['NIR'](text_nir)  # [B, 512] -> [B, 1536]
                        tir_modulator = self.text_adapters['TIR'](text_tir)  # [B, 512] -> [B, 1536]

                        # 组合三个模态的文本调制器（加权平均）
                        text_modulator = (rgb_modulator + nir_modulator + tir_modulator) / 3.0

                        # 门控相乘：使用sigmoid确保数值稳定性
                        gated_fuse = original_fuse * torch.sigmoid(text_modulator)

                        # 残差相加：保留原始视觉信息 + 文本引导增强
                        fuse = original_fuse + self.text_fusion_weight * gated_fuse

                    elif self.text_fusion_method == "attention":
                        # ============ 注意力融合：聚合三模态文本特征后再融合 ============
                        text_combined = (text_rgb + text_nir + text_tir) / 3.0  # [B, 512]
                        fuse = self.text_fusion(fuse, text_combined)  # [B, embed_dim]

                        # 确保输出维度为1536（通过预创建的上采样投影）
                        if self.attention_upsampler is not None:
                            fuse = self.attention_upsampler(fuse)

                    else:
                        # ============ 拼接融合：聚合三模态文本特征后再融合 ============
                        text_combined = (text_rgb + text_nir + text_tir) / 3.0  # [B, 512]
                        fuse = self.text_fusion(fuse, text_combined)  # [B, embed_dim]

                        # 确保输出维度为1536
                        if self.concat_upsampler is not None:
                            fuse = self.concat_upsampler(fuse)

                fuse_global = self.bottleneck_fuse(fuse)  # BNNeck 融合
                fuse_score = self.classifier_fuse(fuse_global)  # 融合分类

            if self.direct:  # 直接输出拼接/融合用于分类（简化 heads）
                if self.mamba:
                    return ori_score, ori, fuse_score, fuse  # 原始与融合并行输出
                else:
                    return ori_score, ori
            else:
                if self.mamba:
                    # 固定返回三模态特征（与MambaPro一致）
                    return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, fuse_score, fuse
                else:
                    # 固定返回三模态特征（与MambaPro一致）
                    return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']  # 测试路径
            NI = x['NI']
            TI = x['TI']
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label, modality='rgb')
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label, modality='nir')
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label, modality='tir')

            if self.mamba:
                # ============ 阶段1：模态内引导 (In-Modal Guidance) ============
                # 对每个模态单独进行文本引导增强
                if self.use_modal_guidance and text_features is not None:
                    # 应用模态内引导
                    RGB_enhanced = self.modal_guidance(RGB_cash, text_features.get('RGB'))
                    NI_enhanced = self.modal_guidance(NI_cash, text_features.get('NIR'))
                    TI_enhanced = self.modal_guidance(TI_cash, text_features.get('TIR'))
                else:
                    # 无文本引导时直接使用原始特征
                    RGB_enhanced, NI_enhanced, TI_enhanced = RGB_cash, NI_cash, TI_cash

                # ============ 阶段2：维度守恒拼接 (Dimension-Invariant Concatenation) ============
                # IMSG 可能返回 [B, seq, D]，拼接前统一取 CLS token (index 0)
                def _get_global(feat):
                    return feat[:, 0] if feat.dim() == 3 else feat
                fuse = torch.cat([_get_global(RGB_enhanced), _get_global(NI_enhanced), _get_global(TI_enhanced)], dim=-1)  # [B, 1536]

                # ============ 兼容性：保留原有文本融合逻辑 (可选) ============
                if self.use_text_fusion and self.text_fusion is not None and text_features is not None:
                    # 准备文本特征：聚合三模态文本
                    text_rgb = text_features['RGB']  # [B, 512]
                    text_nir = text_features['NIR']  # [B, 512]
                    text_tir = text_features['TIR']  # [B, 512]

                    # 聚合文本特征
                    text_combined = (text_rgb + text_nir + text_tir) / 3.0  # [B, 512]

                    # 应用全局文本融合 (可选增强)
                    if self.text_fusion_method == "residual":
                        # residual 分支：用 text_adapters 将文本投影到 1536 维后做残差增强
                        # (避免 TextResidualFusion 输出 embed_dim=512 与 fuse [B,1536] 不匹配)
                        if self.text_adapters is not None:
                            rgb_mod = self.text_adapters['RGB'](text_features['RGB'])
                            nir_mod = self.text_adapters['NIR'](text_features['NIR'])
                            tir_mod = self.text_adapters['TIR'](text_features['TIR'])
                            text_modulator = (rgb_mod + nir_mod + tir_mod) / 3.0
                            fuse = fuse + self.text_fusion_weight * fuse * torch.sigmoid(text_modulator)
                        # 否则跳过文本融合（text_adapters 未初始化）
                    else:
                        # attention/concat 分支：fusion 输出 embed_dim(512)，需上采样回 1536
                        fuse_text = self.text_fusion(fuse, text_combined)  # [B, embed_dim]
                        if self.attention_upsampler is not None:
                            fuse_text = self.attention_upsampler(fuse_text)  # [B, 1536]
                        fuse = fuse + self.text_fusion_weight * fuse_text  # 残差相加保留原始信息

                return fuse
            else:
                # 固定三模态拼接（与MambaPro一致）
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)  # [B, 1536]
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
    # ============ 开关控制：文本融合选择 ============
    use_text_fusion = getattr(cfg.MODEL, 'USE_TEXT_FUSION', False)

    # 始终使用MambaPro作为基础模型
    print("🎯 使用MambaPro模型（AboutReid核心架构）")
    if use_text_fusion:
        print("✅ 已启用文本融合功能")
    else:
        print("❌ 文本融合功能已禁用")
    model = MambaPro(num_class, cfg, camera_num, view_num, __factory_T_type)

    return model  # 返回模型
