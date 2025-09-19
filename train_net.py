from utils.logger import setup_logger
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    args = parser.parse_args() # 解析参数

    if args.config_file != "":
        cfg.merge_from_file(args.config_file) # 从配置文件合并配置
    cfg.merge_from_list(args.opts) # 从命令行合并配置
    cfg.TEST.FEAT = args.fea_cft # 设置特征选择
    cfg.freeze() # 冻结配置

    set_seed(cfg.SOLVER.SEED) # 设置随机种子
    
    if cfg.MODEL.DIST_TRAIN: # 如果使用分布式训练
        torch.cuda.set_device(args.local_rank) # 设置本地排名

    output_dir = cfg.OUTPUT_DIR # 设置输出目录
    if output_dir and not os.path.exists(output_dir): # 如果输出目录不存在
        os.makedirs(output_dir) # 创建输出目录

    logger = setup_logger("MambaPro", output_dir, if_train=True) # 设置日志
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR)) # 打印输出目录
    logger.info(args) # 打印参数

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file)) # 打印加载的配置文件
        with open(args.config_file, 'r') as cf: # 打开配置文件
            config_str = "\n" + cf.read() # 读取配置文件
            logger.info(config_str) # 打印配置文件
    logger.info("Running with config:\n{}".format(cfg)) # 打印配置

    if cfg.MODEL.DIST_TRAIN: # 如果使用分布式训练
        torch.distributed.init_process_group(backend='nccl', init_method='env://') # 初始化分布式训练

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID # 设置可见设备
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg) # 加载数据
    print("data is ready") # 打印数据加载完成
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num) # 加载模型

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes) # 加载损失函数

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion) # 加载优化器

    scheduler = create_scheduler(cfg, optimizer) # 加载调度器
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
