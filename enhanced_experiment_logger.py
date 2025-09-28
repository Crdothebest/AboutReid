#!/usr/bin/env python3
"""
增强版实验记录系统
功能：自动提取和记录实验参数，包括命令行参数和配置文件参数
"""

import os
import json
import yaml
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class EnhancedExperimentLogger:
    """增强版实验记录器"""
    
    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def extract_config_parameters(self, config_file):
        """提取配置文件参数"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 提取关键参数
        key_params = {
            # 模型参数
            "model": {
                "transformer_type": config.get("MODEL", {}).get("TRANSFORMER_TYPE"),
                "stride_size": config.get("MODEL", {}).get("STRIDE_SIZE"),
                "sie_camera": config.get("MODEL", {}).get("SIE_CAMERA"),
                "id_loss_weight": config.get("MODEL", {}).get("ID_LOSS_WEIGHT"),
                "triplet_loss_weight": config.get("MODEL", {}).get("TRIPLET_LOSS_WEIGHT"),
                "prompt": config.get("MODEL", {}).get("PROMPT"),
                "adapter": config.get("MODEL", {}).get("ADAPTER"),
                "mamba": config.get("MODEL", {}).get("MAMBA"),
                "frozen": config.get("MODEL", {}).get("FROZEN"),
            },
            
            # 多尺度参数
            "multi_scale": {
                "use_clip_multi_scale": config.get("MODEL", {}).get("USE_CLIP_MULTI_SCALE"),
                "clip_multi_scale_scales": config.get("MODEL", {}).get("CLIP_MULTI_SCALE_SCALES"),
            },
            
            # MoE参数
            "moe": {
                "use_multi_scale_moe": config.get("MODEL", {}).get("USE_MULTI_SCALE_MOE"),
                "moe_scales": config.get("MODEL", {}).get("MOE_SCALES"),
                "moe_expert_hidden_dim": config.get("MODEL", {}).get("MOE_EXPERT_HIDDEN_DIM"),
                "moe_temperature": config.get("MODEL", {}).get("MOE_TEMPERATURE"),
            },
            
            # 训练参数
            "training": {
                "optimizer": config.get("SOLVER", {}).get("OPTIMIZER_NAME"),
                "base_lr": config.get("SOLVER", {}).get("BASE_LR"),
                "weight_decay": config.get("SOLVER", {}).get("WEIGHT_DECAY"),
                "max_epochs": config.get("SOLVER", {}).get("MAX_EPOCHS"),
                "ims_per_batch": config.get("SOLVER", {}).get("IMS_PER_BATCH"),
                "warmup_iters": config.get("SOLVER", {}).get("WARMUP_ITERS"),
                "margin": config.get("SOLVER", {}).get("MARGIN"),
                "center_loss_weight": config.get("SOLVER", {}).get("CENTER_LOSS_WEIGHT"),
            },
            
            # 数据参数
            "data": {
                "dataset": config.get("DATASETS", {}).get("NAMES"),
                "size_train": config.get("INPUT", {}).get("SIZE_TRAIN"),
                "size_test": config.get("INPUT", {}).get("SIZE_TEST"),
                "num_instance": config.get("DATALOADER", {}).get("NUM_INSTANCE"),
                "num_workers": config.get("DATALOADER", {}).get("NUM_WORKERS"),
            }
        }
        
        return key_params
    
    def extract_command_line_args(self, args):
        """提取命令行参数"""
        cmd_params = {
            "config_file": args.config_file,
            "experiment_name": args.experiment_name,
            "use_moe": args.use_moe if hasattr(args, 'use_moe') else None,
            "disable_moe": args.disable_moe if hasattr(args, 'disable_moe') else None,
            "use_multi_scale": args.use_multi_scale if hasattr(args, 'use_multi_scale') else None,
            "output_dir": args.output_dir if hasattr(args, 'output_dir') else None,
            "resume": args.resume if hasattr(args, 'resume') else None,
            "test_only": args.test_only if hasattr(args, 'test_only') else None,
        }
        return cmd_params
    
    def run_experiment_with_logging(self, config_file, experiment_name, **kwargs):
        """运行实验并记录所有参数"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # 提取参数
        config_params = self.extract_config_parameters(config_file)
        cmd_params = kwargs
        
        # 构建实验记录
        experiment_record = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "start_time": datetime.now().isoformat(),
            "config_file": config_file,
            "parameters": {
                "config_parameters": config_params,
                "command_line_parameters": cmd_params,
                "effective_parameters": self._merge_parameters(config_params, cmd_params)
            },
            "status": "started"
        }
        
        # 保存实验记录
        self._save_experiment_record(experiment_record)
        
        print(f"🚀 开始实验: {experiment_id}")
        print(f"📁 配置文件: {config_file}")
        print(f"⚙️  关键参数:")
        self._print_key_parameters(experiment_record["parameters"]["effective_parameters"])
        
        return experiment_record
    
    def _merge_parameters(self, config_params, cmd_params):
        """合并配置参数和命令行参数，命令行参数优先级更高"""
        effective_params = config_params.copy()
        
        # 处理命令行参数覆盖
        if cmd_params.get("use_moe") is not None:
            effective_params["moe"]["use_multi_scale_moe"] = cmd_params["use_moe"]
        if cmd_params.get("disable_moe") is not None:
            effective_params["moe"]["use_multi_scale_moe"] = not cmd_params["disable_moe"]
        if cmd_params.get("use_multi_scale") is not None:
            effective_params["multi_scale"]["use_clip_multi_scale"] = cmd_params["use_multi_scale"]
        
        return effective_params
    
    def _print_key_parameters(self, params):
        """打印关键参数"""
        print(f"  - 多尺度滑动窗口: {params['multi_scale']['use_clip_multi_scale']}")
        print(f"  - 滑动窗口尺度: {params['multi_scale']['clip_multi_scale_scales']}")
        print(f"  - MoE融合: {params['moe']['use_multi_scale_moe']}")
        print(f"  - 专家隐藏层维度: {params['moe']['moe_expert_hidden_dim']}")
        print(f"  - 温度参数: {params['moe']['moe_temperature']}")
        print(f"  - 学习率: {params['training']['base_lr']}")
        print(f"  - 批次大小: {params['training']['ims_per_batch']}")
        print(f"  - 训练轮数: {params['training']['max_epochs']}")
    
    def _save_experiment_record(self, record):
        """保存实验记录"""
        record_file = self.log_dir / f"{record['experiment_id']}.json"
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)
        
        # 追加到总日志
        total_log = self.log_dir / "all_experiments.jsonl"
        with open(total_log, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def log_experiment_results(self, experiment_id, results):
        """记录实验结果"""
        # 查找实验记录
        record_file = self.log_dir / f"{experiment_id}.json"
        if not record_file.exists():
            print(f"❌ 找不到实验记录: {experiment_id}")
            return
        
        # 读取并更新记录
        with open(record_file, 'r') as f:
            record = json.load(f)
        
        record.update({
            "end_time": datetime.now().isoformat(),
            "status": "completed",
            "results": results
        })
        
        # 保存更新后的记录
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)
        
        print(f"✅ 实验完成: {experiment_id}")
        print(f"📊 结果: mAP={results.get('mAP', 'N/A')}%, Rank-1={results.get('Rank-1', 'N/A')}%")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强版实验记录系统")
    parser.add_argument("config_file", help="配置文件路径")
    parser.add_argument("experiment_name", help="实验名称")
    parser.add_argument("--use_moe", action="store_true", help="启用MoE")
    parser.add_argument("--disable_moe", action="store_true", help="禁用MoE")
    parser.add_argument("--use_multi_scale", action="store_true", help="启用多尺度")
    parser.add_argument("--output_dir", help="输出目录")
    parser.add_argument("--resume", help="恢复训练")
    parser.add_argument("--test_only", action="store_true", help="仅测试")
    
    args = parser.parse_args()
    
    # 创建记录器
    logger = EnhancedExperimentLogger()
    
    # 开始实验记录
    experiment_record = logger.run_experiment_with_logging(
        args.config_file, 
        args.experiment_name,
        **vars(args)
    )
    
    print(f"📝 实验记录已保存: {logger.log_dir / experiment_record['experiment_id']}.json")

if __name__ == "__main__":
    main()
