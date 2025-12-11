from utils.logger import setup_logger
from utils.config_printer import print_final_config
from data import make_dataloader
from modeling import make_model
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from layers.make_loss import make_loss
from engine.processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg


def set_seed(seed):
    """
    设置随机种子，确保实验可复现
    
    Args:
        seed (int): 随机种子值
    
    【注意】
    - deterministic=True 和 benchmark=True 不能同时为True
    - 如果追求完全可复现：设置 deterministic=True, benchmark=False（会降低性能）
    - 如果追求性能：设置 deterministic=False, benchmark=True（可能不完全可复现）
    - 当前设置：deterministic=True, benchmark=False（优先可复现性）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 🔧 修复：deterministic 和 benchmark 不能同时为 True
    # 如果 deterministic=True，则 benchmark 必须为 False 才能确保完全可复现
    torch.backends.cudnn.deterministic = True  # 确保cuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False     # 禁用自动选择算法（与deterministic互斥）

# 训练主函数
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MambaPro Training") # 创建命令行解析器
    parser.add_argument( # 添加配置文件路径参数
        "--config_file", default="/home/zubuntu/workspace/yzy/MambaPro/configs/MSVR310/MambaPro.yml", help="path to config file", type=str
    )# 默认配置文件路径
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int) # 添加特征选择参数
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER) # 添加命令行参数
    parser.add_argument("--local_rank", default=0, type=int) # 添加本地排名参数
    # 🔥 新增：多尺度滑动窗口控制参数
    parser.add_argument("--use_multi_scale", action="store_true", help="Enable multi-scale sliding window (default: False)")
    parser.add_argument("--no_multi_scale", action="store_true", help="Disable multi-scale sliding window (default: False)")
    # 🔥 新增：MoE控制参数（从tools/train.py移植）
    parser.add_argument("--use_moe", action="store_true", 
                       help="启用多尺度MoE特征融合模块 (默认: False)")
    parser.add_argument("--disable_moe", action="store_true", 
                       help="强制禁用多尺度MoE特征融合模块 (默认: False)")
    parser.add_argument("--no_moe", action="store_true", help="Disable Multi-scale MoE fusion (default: False)")
    
    # 🔥 新增：门控融合控制参数
    parser.add_argument("--use_attention", action="store_true", 
                       help="启用门控融合机制 (默认: False)")
    parser.add_argument("--disable_attention", action="store_true", 
                       help="强制禁用门控融合机制 (默认: False)")
    parser.add_argument("--attention_heads", type=int, default=8, 
                       help="设置门控网络头数 (默认: 8)")
    parser.add_argument("--attention_dropout", type=float, default=0.1, 
                       help="设置门控网络Dropout比例 (默认: 0.1)")
    args = parser.parse_args() # 解析参数

    if args.config_file != "":
        cfg.merge_from_file(args.config_file) # 从配置文件合并配置
    
    # 配置加载优先级：默认值 < YAML文件 < 命令行参数（--opts）
    if args.opts:
        try:
            cfg.merge_from_list(args.opts)
            
            # 修正布尔值参数（YACS可能将"True"/"False"解析为字符串）
            bool_params = [
                ('MODEL', 'USE_ATTENTION_FUSION'),
                ('MODEL', 'USE_GATE_FUSION'),
                ('MODEL', 'MOE_USE_FIXED_WEIGHTS'),
                ('MODEL', 'MOE_USE_TOP_K_ROUTING')
            ]
            for section_name, param_name in bool_params:
                if hasattr(cfg, section_name):
                    section = getattr(cfg, section_name)
                    if hasattr(section, param_name):
                        val = getattr(section, param_name)
                        if isinstance(val, str):
                            setattr(section, param_name, val.lower() in ('true', '1', 'yes'))
        except Exception as e:
            print(f"❌ --opts 参数解析错误: {e}")
            raise
    
    cfg.TEST.FEAT = args.fea_cft # 设置特征选择
    
    # 命令行参数覆盖配置文件设置
    if args.use_multi_scale:
        cfg.MODEL.USE_CLIP_MULTI_SCALE = True
    elif args.no_multi_scale:
        cfg.MODEL.USE_CLIP_MULTI_SCALE = False
    
    if args.disable_moe:
        cfg.defrost()
        cfg.MODEL.USE_MULTI_SCALE_MOE = False
        cfg.freeze()
    elif args.use_moe:
        cfg.defrost()
        cfg.MODEL.USE_MULTI_SCALE_MOE = True
        cfg.MODEL.USE_CLIP_MULTI_SCALE = True
        cfg.freeze()
    elif args.no_moe:
        cfg.defrost()
        cfg.MODEL.USE_MULTI_SCALE_MOE = False
        cfg.freeze()
    
    if args.use_attention:
        cfg.MODEL.USE_GATE_FUSION = True
        cfg.MODEL.GATE_DROPOUT = args.attention_dropout
    elif args.disable_attention:
        cfg.MODEL.USE_GATE_FUSION = False
    
    cfg.freeze() # 冻结配置

    set_seed(cfg.SOLVER.SEED) # 设置随机种子
    
    if cfg.MODEL.DIST_TRAIN: # 如果使用分布式训练
        torch.cuda.set_device(args.local_rank) # 设置本地排名

    output_dir = cfg.OUTPUT_DIR # 设置输出目录
    if output_dir and not os.path.exists(output_dir): # 如果输出目录不存在
        os.makedirs(output_dir) # 创建输出目录

    logger = setup_logger("MambaPro", output_dir, if_train=True)
    
    # 打印最终配置（统一输出）
    print_final_config(cfg)

    if cfg.MODEL.DIST_TRAIN: # 如果使用分布式训练
        torch.distributed.init_process_group(backend='nccl', init_method='env://') # 初始化分布式训练

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    do_train(
        cfg, # 配置
        model, # 模型
        center_criterion, # 中心损失
        train_loader, # 训练数据
        val_loader, # 验证数据
        optimizer, # 优化器
        optimizer_center, # 中心优化器
        scheduler, # 调度器
        loss_func, # 损失函数
        num_query, args.local_rank # 查询数量和本地排名
    )
    
