import logging
import os
import sys
import os.path as osp


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)  # 获取或创建名为 'name' 的logger
    logger.setLevel(logging.DEBUG)  # 设置日志记录器的级别为 DEBUG，这样所有级别的日志都会被记录

    ch = logging.StreamHandler(stream=sys.stdout)  # 创建一个流处理器，将日志输出到控制台（标准输出）
    ch.setLevel(logging.DEBUG)  # 设置控制台日志输出的级别为 DEBUG
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")  # 定义日志格式，包含时间、logger名、日志级别和日志消息
    ch.setFormatter(formatter)  # 将格式化器设置给流处理器
    logger.addHandler(ch)  # 将流处理器添加到logger中

    if save_dir:  # 如果提供了保存日志文件的目录
        if not osp.exists(save_dir):  # 如果目录不存在
            os.makedirs(save_dir)  # 创建目录
        if if_train:  # 如果是训练阶段
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')  # 创建一个文件处理器，将日志写入 train_log.txt
        else:  # 如果是测试阶段
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')  # 创建一个文件处理器，将日志写入 test_log.txt
        fh.setLevel(logging.DEBUG)  # 设置文件日志输出的级别为 DEBUG
        fh.setFormatter(formatter)  # 将格式化器设置给文件处理器
        logger.addHandler(fh)  # 将文件处理器添加到logger中

    return logger  # 返回配置好的logger
