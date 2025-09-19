# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import errno
import json
import os

import os.path as osp

def mkdir_if_missing(directory):
    if not osp.exists(directory):  # 检查目录是否已存在
        try:
            os.makedirs(directory)  # 如果不存在，则创建目录
        except OSError as e:  # 如果创建目录时发生错误
            if e.errno != errno.EEXIST:  # 如果错误是目录已存在，则忽略
                raise  # 如果是其他错误，则抛出异常



def check_isfile(path):
    isfile = osp.isfile(path)  # 检查路径是否是一个文件
    if not isfile:  # 如果路径不是文件
        print("=> Warning: no file found at '{}' (ignored)".format(path))  # 打印警告信息
    return isfile  # 返回是否是文件



def read_json(fpath):
    with open(fpath, 'r') as f:  # 以只读模式打开文件
        obj = json.load(f)  # 从文件中读取JSON对象
    return obj  # 返回JSON对象


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))  # 确保目标路径的目录存在
    with open(fpath, 'w') as f:  # 以写模式打开文件
        json.dump(obj, f, indent=4, separators=(',', ': '))  # 将 Python 对象写入文件，并进行格式化输出

