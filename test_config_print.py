#!/usr/bin/env python3
"""
测试配置打印功能的脚本

运行方法：
cd /home/zhanghaoyang/Desktop/yzy/AboutReid
python test_config_print.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from utils.config_printer import print_final_config

def test_config_print():
    """测试配置打印功能"""

    # 加载默认配置
    cfg.merge_from_file('configs/RGBNT201/20251013_experiment_config.yml')

    # 模拟命令行参数覆盖
    cfg.merge_from_list([
        'DATALOADER.SAMPLER', 'softmax',
        'DATASETS.USE_TEXT_FEATURES', 'True',
        'DATASETS.NAMES', 'RGBNT201_IDEA',
        'MODEL.USE_TEXT_FUSION', 'True',
        'MODEL.USE_MODAL_GUIDANCE', 'True',
        'SOLVER.IMS_PER_BATCH', '64',
        'OUTPUT_DIR', 'outputs/test_config_print'
    ])

    print("🔍 测试配置打印功能...")
    print("配置文件：configs/RGBNT201/20251013_experiment_config.yml")
    print("命令行参数：模拟用户输入的参数")
    print()

    # 打印配置
    print_final_config(cfg)

    print("\n✅ 配置打印测试完成！")
    print("请检查上面的输出是否正确显示了：")
    print("- 文本融合的具体方法（注意力融合）")
    print("- 模态内引导的状态")
    print("- 数据采样器的类型")
    print("- 其他重要模块的详细信息")

if __name__ == "__main__":
    test_config_print()
