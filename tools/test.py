"""
MambaPro 测试/推理入口脚本（中文说明）

功能概述：
- 解析命令行参数与 YAML 配置，合并到全局配置 `cfg`
- 设置日志与可见 GPU，构建数据加载器与模型
- 加载权重后，调用 `engine.processor.do_inference` 在验证/测试集上评估

常用启动方式示例：
    python tools/test.py --config_file configs/RGBNT201/MambaPro.yml \
        opts MODEL.DEVICE_ID 0 TEST.WEIGHT /path/to/your.pth

关键参数：
- --config_file: 指定 YAML 配置文件路径
- opts: 通过命令行动态覆盖配置，如 TEST.WEIGHT 测试权重路径等
"""

import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":  # 确保只在脚本直接运行时执行
    parser = argparse.ArgumentParser(description="MambaPro Testing")  # 创建命令行解析器
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str  # 指定配置文件路径
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)  # 命令行覆盖配置（键值对列表）

    args = parser.parse_args()  # 解析参数

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)  # 从 YAML 合并配置
    cfg.merge_from_list(args.opts)  # 从命令行 opts 合并覆盖
    cfg.freeze()  # 冻结配置，避免后续修改

    output_dir = cfg.OUTPUT_DIR  # 输出目录（保存日志/结果）
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 若不存在则创建

    logger = setup_logger("MambaPro", output_dir, if_train=False)  # 初始化日志器（测试模式）
    logger.info(args)  # 记录启动参数

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))  # 记录加载的配置文件
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()  # 读取配置文本
            logger.info(config_str)  # 打印配置详情
    logger.info("Running with config:\n{}".format(cfg))  # 打印最终 cfg

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # 指定可见 GPU（如 "0" 或 "0,1"）

    # 构建数据加载器（测试只用 val_loader 和 num_query 等信息）
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 构建模型并加载权重
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    # 建议通过配置传入 TEST.WEIGHT，或直接替换下行路径
    model.load_param("/path/to/your/.pth")  # 从指定路径加载模型权重

    do_inference(cfg, model, val_loader, num_query)  # 执行推理评估
