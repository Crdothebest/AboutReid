#!/usr/bin/env python3
"""
快速实验记录脚本
功能：简化版实验记录，适合快速记录
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

def log_experiment(config_file, experiment_name, results=None):
    """
    快速记录实验
    
    Args:
        config_file (str): 配置文件路径
        experiment_name (str): 实验名称
        results (dict): 实验结果
    """
    # 创建日志目录
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 记录实验信息
    log_entry = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "config_file": config_file,
        "results": results or {},
        "log_time": datetime.now().isoformat()
    }
    
    # 保存到JSON文件
    log_file = log_dir / f"{experiment_name}_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    # 追加到总日志
    total_log = log_dir / "all_experiments.jsonl"
    with open(total_log, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"📝 实验记录已保存: {log_file}")
    return log_file

def create_experiment_summary():
    """创建实验总结"""
    log_dir = Path("experiment_logs")
    total_log = log_dir / "all_experiments.jsonl"
    
    if not total_log.exists():
        print("❌ 没有找到实验记录")
        return
    
    # 读取所有实验记录
    experiments = []
    with open(total_log, 'r') as f:
        for line in f:
            experiments.append(json.loads(line))
    
    # 创建总结
    summary = {
        "total_experiments": len(experiments),
        "experiments": experiments,
        "summary_time": datetime.now().isoformat()
    }
    
    summary_file = log_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 实验总结已创建: {summary_file}")
    return summary_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法: python quick_experiment_log.py <config_file> <experiment_name>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    experiment_name = sys.argv[2]
    
    # 记录实验
    log_file = log_experiment(config_file, experiment_name)
    
    # 创建总结
    create_experiment_summary()
