#!/usr/bin/env python3
"""
实验记录系统
功能：自动记录实验配置、结果和分析
作者：实验管理系统
日期：2024
"""

import os
import json
import yaml
import time
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    """实验记录器"""
    
    def __init__(self, log_dir="experiment_logs"):
        """
        初始化实验记录器
        
        Args:
            log_dir (str): 日志保存目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.experiment_id = None
        self.start_time = None
        
    def start_experiment(self, config_file, experiment_name=None):
        """
        开始实验记录
        
        Args:
            config_file (str): 配置文件路径
            experiment_name (str): 实验名称
        """
        # 生成实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.experiment_id = f"{experiment_name}_{timestamp}"
        else:
            config_name = Path(config_file).stem
            self.experiment_id = f"{config_name}_{timestamp}"
        
        self.start_time = time.time()
        
        # 创建实验目录
        self.exp_dir = self.log_dir / self.experiment_id
        self.exp_dir.mkdir(exist_ok=True)
        
        # 记录实验开始信息
        self.log_experiment_start(config_file)
        
        print(f"🚀 实验开始: {self.experiment_id}")
        print(f"📁 实验目录: {self.exp_dir}")
        
    def log_experiment_start(self, config_file):
        """记录实验开始信息"""
        start_info = {
            "experiment_id": self.experiment_id,
            "start_time": datetime.now().isoformat(),
            "config_file": str(config_file),
            "status": "started"
        }
        
        # 保存配置文件
        config_path = self.exp_dir / "config.yml"
        with open(config_file, 'r') as f:
            config_content = f.read()
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # 保存开始信息
        start_path = self.exp_dir / "experiment_info.json"
        with open(start_path, 'w') as f:
            json.dump(start_info, f, indent=2)
    
    def log_training_progress(self, epoch, iteration, loss, accuracy, lr):
        """
        记录训练进度
        
        Args:
            epoch (int): 当前轮数
            iteration (int): 当前迭代
            loss (float): 损失值
            accuracy (float): 准确率
            lr (float): 学习率
        """
        progress_info = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "iteration": iteration,
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": lr
        }
        
        # 追加到进度文件
        progress_path = self.exp_dir / "training_progress.jsonl"
        with open(progress_path, 'a') as f:
            f.write(json.dumps(progress_info) + '\n')
    
    def log_experiment_results(self, results):
        """
        记录实验结果
        
        Args:
            results (dict): 实验结果字典
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        # 完整结果记录
        final_results = {
            "experiment_id": self.experiment_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
            "status": "completed",
            "results": results
        }
        
        # 保存结果
        results_path = self.exp_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # 保存到总览文件
        self._update_experiment_overview(final_results)
        
        print(f"✅ 实验完成: {self.experiment_id}")
        print(f"⏱️  训练时间: {duration/3600:.2f} 小时")
        print(f"📊 结果保存: {results_path}")
    
    def _update_experiment_overview(self, results):
        """更新实验总览"""
        overview_path = self.log_dir / "experiment_overview.csv"
        
        # 提取关键信息
        overview_data = {
            "experiment_id": results["experiment_id"],
            "start_time": results["start_time"],
            "duration_hours": results["duration_hours"],
            "status": results["status"]
        }
        
        # 添加结果指标
        if "results" in results:
            for key, value in results["results"].items():
                overview_data[key] = value
        
        # 更新CSV文件
        if overview_path.exists():
            df = pd.read_csv(overview_path)
            new_row = pd.DataFrame([overview_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([overview_data])
        
        df.to_csv(overview_path, index=False)
    
    def log_error(self, error_message, traceback_info=None):
        """记录实验错误"""
        error_info = {
            "experiment_id": self.experiment_id,
            "error_time": datetime.now().isoformat(),
            "error_message": error_message,
            "traceback": traceback_info
        }
        
        error_path = self.exp_dir / "error_log.json"
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"❌ 实验错误: {self.experiment_id}")
        print(f"📝 错误信息: {error_message}")

def run_experiment_with_logging(config_file, experiment_name=None):
    """
    运行实验并自动记录
    
    Args:
        config_file (str): 配置文件路径
        experiment_name (str): 实验名称
    """
    logger = ExperimentLogger()
    
    try:
        # 开始实验记录
        logger.start_experiment(config_file, experiment_name)
        
        # 运行训练命令
        cmd = f"python train_net.py --config_file {config_file}"
        print(f"🚀 执行命令: {cmd}")
        
        # 这里可以添加实时监控训练进度的逻辑
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 实验成功完成
            # 这里需要从输出中提取结果指标
            results = extract_results_from_output(result.stdout)
            logger.log_experiment_results(results)
        else:
            # 实验失败
            logger.log_error(result.stderr)
            
    except Exception as e:
        logger.log_error(str(e), str(e.__traceback__))

def extract_results_from_output(output):
    """
    从训练输出中提取结果指标
    
    Args:
        output (str): 训练输出文本
        
    Returns:
        dict: 提取的结果指标
    """
    # 这里需要根据您的训练输出格式来解析结果
    # 示例解析逻辑
    results = {}
    
    # 查找mAP指标
    if "mAP:" in output:
        # 解析mAP值
        pass
    
    # 查找Rank-1指标
    if "Rank-1:" in output:
        # 解析Rank-1值
        pass
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python experiment_logger.py <config_file> [experiment_name]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_experiment_with_logging(config_file, experiment_name)
