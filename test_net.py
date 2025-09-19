import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":# 测试主函数
    parser = argparse.ArgumentParser(description="MambaPro Testing") # 创建命令行解析器
    parser.add_argument( # 添加配置文件路径参数
        "--config_file", default="", help="path to config file", type=str # 默认配置文件路径
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, # 添加命令行参数
                        nargs=argparse.REMAINDER)

    args = parser.parse_args() # 解析参数

    if args.config_file != "": # 如果配置文件路径不为空
        cfg.merge_from_file(args.config_file) # 从配置文件合并配置
    cfg.merge_from_list(args.opts) # 从命令行合并配置
    cfg.freeze() # 冻结配置
    output_dir = cfg.OUTPUT_DIR # 设置输出目录
    if output_dir and not os.path.exists(output_dir): # 如果输出目录不存在
        os.makedirs(output_dir) # 创建输出目录

    logger = setup_logger("MambaPro", output_dir, if_train=False) # 设置日志
    logger.info(args) # 打印参数

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file)) # 打印加载的配置文件
        with open(args.config_file, 'r') as cf: # 打开配置文件
            config_str = "\n" + cf.read() # 读取配置文件
            logger.info(config_str) # 打印配置文件
    logger.info("Running with config:\n{}".format(cfg)) # 打印配置

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID # 设置可见设备

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg) # 加载数据

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num) # 加载模型
    model.eval() # 设置模型为评估模式
    model.load_param("your model path") # 加载模型权重
    do_inference(cfg,model,val_loader,num_query) # 进行推理
