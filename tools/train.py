"""
MambaPro 训练入口脚本（中文说明）

功能概述：
- 解析命令行参数与 YAML 配置，合并到全局配置 `cfg`
- 设置随机种子、可见 GPU、分布式（可选）与日志
- 构建数据加载器、模型、损失、优化器与学习率调度器
- 调用训练引擎 `engine.processor.do_train` 执行完整训练过程

常用启动方式示例：
    python tools/train.py --config_file configs/RGBNT201/MambaPro.yml

关键参数：
- --config_file: 指定 YAML 配置文件路径
- opts: 通过命令行动态覆盖配置，如 DATASETS.NAMES RGBNT201 等

输出：
- 日志与权重保存到 `cfg.OUTPUT_DIR`
"""

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
    torch.manual_seed(seed)  # 设置PyTorch CPU上的随机种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU上的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU上的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python原生随机模块的种子
    torch.backends.cudnn.deterministic = True  # 强制CuDNN使用确定性算法（可复现）
    torch.backends.cudnn.benchmark = True  # 允许CuDNN在静态输入形状时加速（非完全可复现）



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MambaPro Training")  # 构建命令行解析器
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str  # 指定配置文件路径
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)  # 命令行覆盖配置（键值对列表）
    parser.add_argument("--local_rank", default=0, type=int)  # 分布式训练时由 torchrun 传入
    args = parser.parse_args()  # 解析参数

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)  # 从 YAML 合并配置
    cfg.merge_from_list(args.opts)  # 从命令行 opts 合并覆盖
    cfg.freeze()  # 冻结配置，避免训练中被误改

    set_seed(cfg.SOLVER.SEED)  # 设定随机种子

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)  # 多卡分布式时设置当前进程所用 GPU

    output_dir = cfg.OUTPUT_DIR  # 输出目录（日志/权重）
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 若不存在则创建

    logger = setup_logger("MambaPro", output_dir, if_train=True)  # 初始化日志器
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))  # 打印保存路径
    logger.info(args)  # 打印启动参数

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))  # 记录已加载的配置文件
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()  # 读取配置内容
            logger.info(config_str)  # 打印配置详情
    logger.info("Running with config:\n{}".format(cfg))  # 打印最终 cfg

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')  # 初始化分布式进程组

    # 选择GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # 指定可见 GPU（如 "0,1"）
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)  # 构建数据加载器
    print("data is ready")  # 数据就绪提示

    # 构建模型 
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)  # 构建模型

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)  # 构建损失函数（含 center loss 可选）

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)  # 构建优化器（含 center 参数优化器）

    scheduler = create_scheduler(cfg, optimizer)  # 构建学习率调度器
    do_train(
        cfg,  # 全局配置
        model,  # 模型
        center_criterion,  # center loss 的中心参数（若启用）
        train_loader,  # 训练数据
        val_loader,  # 验证数据
        optimizer,  # 主优化器
        optimizer_center,  # center 优化器
        scheduler,  # 学习率调度
        loss_func,  # 复合损失
        num_query, args.local_rank  # 验证集 query 数量与本地 rank
    )  # 启动训练主循环
