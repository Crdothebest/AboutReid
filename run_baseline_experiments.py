#!/usr/bin/env python3
"""
MambaPro Baseline Experiments Runner
用于运行毕业论文的基线实验和消融实验

使用方法:
python run_baseline_experiments.py --mode baseline
python run_baseline_experiments.py --mode ablation
"""

import os
import subprocess
import argparse
import yaml
from datetime import datetime

def run_experiment(config_path, gpu_id=0, experiment_name=""):
    """
    运行单个实验
    
    Args:
        config_path (str): 配置文件路径
        gpu_id (int): GPU ID
        experiment_name (str): 实验名称
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    
    cmd = [
        'python', 'train_net.py',
        '--config_file', config_path,
        'MODEL.DEVICE_ID', str(gpu_id)
    ]
    
    try:
        # 运行训练
        print(f"Training command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        if result.returncode == 0:
            print(f"✅ Training completed successfully for {experiment_name}")
            
            # 运行测试
            test_cmd = [
                'python', 'tools/test.py',
                '--config_file', config_path,
                'MODEL.DEVICE_ID', str(gpu_id)
            ]
            
            print(f"Testing command: {' '.join(test_cmd)}")
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            if test_result.returncode == 0:
                print(f"✅ Testing completed successfully for {experiment_name}")
            else:
                print(f"❌ Testing failed for {experiment_name}: {test_result.stderr}")
                
        else:
            print(f"❌ Training failed for {experiment_name}: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {experiment_name} timed out")
    except Exception as e:
        print(f"❌ Experiment {experiment_name} failed with exception: {e}")

def run_baseline_experiments(gpu_id=0):
    """
    运行基线实验
    """
    print("🚀 Starting Baseline Experiments")
    
    # 基线实验配置
    baseline_config = 'configs/RGBNT201/baseline.yml'
    
    # 运行基线实验
    run_experiment(baseline_config, gpu_id, "Baseline")

def run_ablation_experiments(gpu_id=0):
    """
    运行消融实验
    """
    print("🔬 Starting Ablation Studies")
    
    # 消融实验配置
    ablation_configs = {
        "No Prompt": {
            "config": "configs/RGBNT201/baseline.yml",
            "overrides": ["MODEL.PROMPT", "False"]
        },
        "No Adapter": {
            "config": "configs/RGBNT201/baseline.yml", 
            "overrides": ["MODEL.ADAPTER", "False"]
        },
        "No Mamba": {
            "config": "configs/RGBNT201/baseline.yml",
            "overrides": ["MODEL.MAMBA", "False"]
        },
        "No Frozen": {
            "config": "configs/RGBNT201/baseline.yml",
            "overrides": ["MODEL.FROZEN", "False"]
        },
        "RGB Only": {
            "config": "configs/RGBNT201/baseline.yml",
            "overrides": ["TEST.MISS", "nir_tir"]
        },
        "NI Only": {
            "config": "configs/RGBNT201/baseline.yml", 
            "overrides": ["TEST.MISS", "rgb_tir"]
        },
        "TI Only": {
            "config": "configs/RGBNT201/baseline.yml",
            "overrides": ["TEST.MISS", "rgb_nir"]
        }
    }
    
    for exp_name, exp_config in ablation_configs.items():
        # 构建命令
        cmd = [
            'python', 'train_net.py',
            '--config_file', exp_config['config'],
            'MODEL.DEVICE_ID', str(gpu_id)
        ]
        
        # 添加覆盖参数
        cmd.extend(exp_config['overrides'])
        
        print(f"\n{'='*60}")
        print(f"Running Ablation: {exp_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode == 0:
                print(f"✅ {exp_name} completed successfully")
            else:
                print(f"❌ {exp_name} failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {exp_name} timed out")
        except Exception as e:
            print(f"❌ {exp_name} failed with exception: {e}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="Run MambaPro experiments")
    parser.add_argument("--mode", choices=["baseline", "ablation", "all"], 
                       default="baseline", help="Experiment mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--config", type=str, default="configs/RGBNT201/baseline.yml",
                       help="Base configuration file")
    
    args = parser.parse_args()
    
    print(f"🎯 MambaPro Experiment Runner")
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    print(f"Config: {args.config}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.mode == "baseline":
        run_baseline_experiments(args.gpu)
    elif args.mode == "ablation":
        run_ablation_experiments(args.gpu)
    elif args.mode == "all":
        run_baseline_experiments(args.gpu)
        run_ablation_experiments(args.gpu)
    
    print(f"\n🎉 All experiments completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
