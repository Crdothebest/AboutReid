# Category: train_utils (训练与实验控制)
# Description: 负责模型训练启动、自动化实验管理及消融实验运行

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
import json
import sys
import logging
from datetime import datetime
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


def log_run_parameters(logger, args, cfg):
    """
    将当前运行的命令行参数与最终配置写入 train_log，便于后续复现/调参。
    """
    separator = "=" * 80
    logger.info(separator)
    logger.info("🧾 实验参数快照（命令行 + 最终配置）")
    logger.info(separator)

    reproduce_cmd = f"{sys.executable} " + " ".join(sys.argv) if sys.argv else "python train_net.py"
    logger.info("【复现实验命令】")
    logger.info(reproduce_cmd)

    logger.info("【命令行参数（args）】")
    args_dict = vars(args)
    args_json = json.dumps(args_dict, ensure_ascii=False, indent=2, default=str)
    for line in args_json.splitlines():
        logger.info(line)

    logger.info(f"【工作目录】{os.getcwd()}")
    logger.info("【最终配置（YAML）】")
    cfg_yaml = cfg.dump()
    for line in cfg_yaml.splitlines():
        logger.info(line)
    logger.info(separator)


def rename_output_directory(output_dir, best_map):
    """
    将输出目录重命名为 "<mAP>_<组合方式>_<时间戳>"，返回新的目录路径。
    格式示例：85.23_4x4+8x8_20251216_220218
    """
    if not output_dir or not os.path.isdir(output_dir):
        return None

    normalized_dir = os.path.normpath(output_dir.rstrip(os.sep))
    parent_dir, base_name = os.path.split(normalized_dir)

    try:
        best_map_value = float(best_map) if best_map is not None else 0.0
    except (TypeError, ValueError):
        best_map_value = 0.0

    best_map_percent = max(best_map_value * 100.0, 0.0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 尝试从目录名中提取组合方式（例如：4x4+8x8_20251216_220218 -> 4x4+8x8）
    # 组合方式格式：可能包含 x 和 +，例如 4x4, 8x8, 4x4+8x8 等
    scale_label = None
    if '_' in base_name:
        # 如果目录名包含下划线，尝试提取第一部分作为组合方式
        parts = base_name.split('_')
        # 检查第一部分是否匹配组合方式格式（包含 x 或 +）
        if len(parts) > 0 and ('x' in parts[0] or '+' in parts[0]):
            scale_label = parts[0]
            # 如果提取到了组合方式，从时间戳中提取原始时间戳（如果有）
            if len(parts) > 1:
                # 尝试从后续部分提取时间戳
                for part in parts[1:]:
                    if len(part) == 17 and part.replace('_', '').isdigit():
                        timestamp = part
                        break
    
    # 如果未提取到组合方式，尝试从 run_ 前缀后的名称中提取
    if scale_label is None and base_name.startswith('run_'):
        # run_20251216_220218_s4s8 -> 尝试提取 s4s8 并转换为 4x4+8x8
        # 这里简化处理，如果无法提取，就使用原始目录名
        scale_label = base_name.replace('run_', '').split('_')[0] if '_' in base_name else None
    
    # 构建新的目录名：mAP_组合方式_时间戳
    if scale_label:
        base_candidate = f"{best_map_percent:.2f}_{scale_label}_{timestamp}"
    else:
        # 如果无法提取组合方式，使用原始格式（兼容性）
        base_candidate = f"{best_map_percent:.2f}_{base_name}_{timestamp}"
    
    candidate = os.path.join(parent_dir, base_candidate)

    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(parent_dir, f"{base_candidate}_{suffix}")
        suffix += 1

    os.rename(normalized_dir, candidate)
    return candidate

# 训练主函数
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MambaPro Training") # 创建命令行解析器
    parser.add_argument( # 添加配置文件路径参数
        "--config_file", default="", help="path to config file", type=str
    )# 默认配置文件路径（留空，必须通过命令行指定）
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int) # 添加特征选择参数
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
    # 🔥 新增：恢复训练控制参数
    parser.add_argument("--resume", type=str, default="",
                       help="恢复训练的检查点路径 (默认: 空，从头开始训练)")
    parser.add_argument("--no-resume", action="store_true",
                       help="强制禁用恢复训练功能，即使指定了resume路径也不恢复")
    parser.add_argument("--enable-resume", action="store_true",
                       help="强制启用恢复训练功能，忽略环境变量和配置文件的禁用设置")
    # 🔥 修复：将 opts 移到所有 -- 参数之后，避免 REMAINDER 捕获后续参数
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER) # 添加命令行参数（必须在所有 -- 参数之后）
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
                ('MODEL', 'MOE_USE_TOP_K_ROUTING'),
                ('MODEL', 'USE_CLIP_MULTI_SCALE'),  # 🔥 新增：多尺度滑动窗口布尔值
                ('MODEL', 'USE_MULTI_SCALE_MOE'),   # 🔥 新增：多尺度MoE布尔值
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
    
    # 🔥 自动调整 MOE_NUM_EXPERTS 使其与 MOE_SCALES 的长度一致
    if cfg.MODEL.USE_MULTI_SCALE_MOE:
        cfg.defrost()
        moe_scales = cfg.MODEL.MOE_SCALES
        if isinstance(moe_scales, str):
            # 如果 MOE_SCALES 是字符串，尝试解析为列表
            import ast
            try:
                moe_scales = ast.literal_eval(moe_scales)
            except:
                moe_scales = [4, 8, 16]  # 默认值
        
        num_scales = len(moe_scales) if isinstance(moe_scales, (list, tuple)) else 1
        cfg.MODEL.MOE_NUM_EXPERTS = num_scales
        
        # 如果使用固定权重，确保权重数量与专家数量一致
        if hasattr(cfg.MODEL, 'MOE_USE_FIXED_WEIGHTS') and cfg.MODEL.MOE_USE_FIXED_WEIGHTS:
            fixed_weights = getattr(cfg.MODEL, 'MOE_FIXED_WEIGHTS', None)
            if fixed_weights and len(fixed_weights) != num_scales:
                # 如果固定权重数量不匹配，使用均等权重
                import warnings
                warnings.warn(f"MOE_FIXED_WEIGHTS 数量 ({len(fixed_weights)}) 与专家数量 ({num_scales}) 不匹配，将使用均等权重")
                cfg.MODEL.MOE_FIXED_WEIGHTS = [1.0 / num_scales] * num_scales
        
        cfg.freeze()
    
    cfg.freeze() # 冻结配置

    set_seed(cfg.SOLVER.SEED) # 设置随机种子
    
    if cfg.MODEL.DIST_TRAIN: # 如果使用分布式训练
        torch.cuda.set_device(args.local_rank) # 设置本地排名

    base_output_dir = cfg.OUTPUT_DIR
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_output_dir:
        # 检查 OUTPUT_DIR 是否已经是一个完整的路径（包含组合方式和时间戳）
        # 如果目录名包含组合方式标识（x 或 +），则不再创建子目录
        normalized_base = os.path.normpath(base_output_dir.rstrip(os.sep))
        base_name = os.path.basename(normalized_base)
        
        # 如果目录名包含组合方式标识（x 或 +），说明是循环脚本设置的完整路径
        # 直接使用该路径，不再创建子目录
        if 'x' in base_name or '+' in base_name:
            run_dir = normalized_base
        else:
            # 否则，在基础目录下创建 run_{timestamp} 子目录（保持向后兼容）
            os.makedirs(base_output_dir, exist_ok=True)
            run_dir = os.path.join(base_output_dir, f"run_{run_timestamp}")
            suffix = 1
            while os.path.exists(run_dir):
                run_dir = os.path.join(base_output_dir, f"run_{run_timestamp}_{suffix}")
                suffix += 1
        
        cfg.defrost()
        cfg.OUTPUT_DIR = run_dir
        cfg.freeze()
    else:
        run_dir = None

    output_dir = run_dir
    logs_dir = output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

    log_filename = f"train_{run_timestamp}.log"
    logger = setup_logger("MambaPro", logs_dir, if_train=True, filename=log_filename)
    
    # 🔥 打印窗口尺度信息（在配置输出之前，更醒目）
    if cfg.MODEL.USE_MULTI_SCALE_MOE:
        moe_scales = cfg.MODEL.MOE_SCALES
        if isinstance(moe_scales, (list, tuple)):
            moe_scale_labels = [f"{s}x{s}" for s in moe_scales]
            window_display = "+".join(moe_scale_labels)
            num_experts = cfg.MODEL.MOE_NUM_EXPERTS
            print("\n" + "=" * 80)
            print(f"🔥 当前训练配置：使用 {window_display} 窗口，{num_experts} 个MoE专家")
            print("=" * 80 + "\n")
            logger.info(f"🔥 窗口配置：{window_display} ({moe_scales}), MoE专家数量：{num_experts}")
    elif cfg.MODEL.USE_CLIP_MULTI_SCALE:
        scales = cfg.MODEL.CLIP_MULTI_SCALE_SCALES
        if isinstance(scales, (list, tuple)):
            scale_labels = [f"{s}x{s}" for s in scales]
            window_display = "+".join(scale_labels)
            print("\n" + "=" * 80)
            print(f"🔥 当前训练配置：使用 {window_display} 窗口（传统MLP融合）")
            print("=" * 80 + "\n")
            logger.info(f"🔥 窗口配置：{window_display} ({scales})")
    
    # 打印最终配置（统一输出）
    print_final_config(cfg)
    log_run_parameters(logger, args, cfg)

    if cfg.MODEL.DIST_TRAIN: # 如果使用分布式训练
        torch.distributed.init_process_group(backend='nccl', init_method='env://') # 初始化分布式训练

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    # 🔥 获取恢复训练的检查点路径（优先使用命令行参数，其次使用配置文件）
    # 控制优先级：--enable-resume > --no-resume > 环境变量 > 命令行参数 > 配置文件
    import os

    # 检查环境变量 DISABLE_RESUME
    disable_resume_env = os.getenv('DISABLE_RESUME', '').lower() in ('1', 'true', 'yes')

    if args.enable_resume:
        # 强制启用：使用命令行或配置文件指定的路径
        resume_path = args.resume if args.resume else getattr(cfg.SOLVER, 'RESUME', "")
    elif args.no_resume or disable_resume_env:
        # 强制禁用：清空resume路径
        resume_path = ""
    else:
        # 默认行为：使用命令行或配置文件指定的路径
        resume_path = args.resume if args.resume else getattr(cfg.SOLVER, 'RESUME', "")
    best_index = do_train(
        cfg, # 配置
        model, # 模型
        center_criterion, # 中心损失
        train_loader, # 训练数据
        val_loader, # 验证数据
        optimizer, # 优化器
        optimizer_center, # 中心优化器
        scheduler, # 调度器
        loss_func, # 损失函数
        num_query, args.local_rank, # 查询数量和本地排名
        resume=resume_path # 🔥 新增：恢复训练的检查点路径
    )
    # 仅在主进程/单卡环境重命名大文件夹
    is_main_process = (not cfg.MODEL.DIST_TRAIN) or args.local_rank == 0
    if is_main_process and output_dir and os.path.isdir(output_dir):
        best_map_value = best_index.get('mAP', 0.0) if best_index else 0.0
        logging.shutdown()  # 释放文件句柄，避免Windows下重命名失败
        try:
            renamed_dir = rename_output_directory(output_dir, best_map_value)
            if renamed_dir:
                print(f"📁 输出目录重命名为: {renamed_dir}")
        except OSError as rename_err:
            print(f"❌ 输出目录重命名失败: {rename_err}")

