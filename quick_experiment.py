#!/usr/bin/env python3
"""
快速实验脚本
功能：一键运行您的训练命令并自动记录结果
作者：实验记录系统
日期：2024
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

def run_quick_experiment():
    """运行快速实验"""
    
    # 创建实验目录
    base_dir = Path("results/experiments")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成实验ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"experiment_{timestamp}"
    experiment_dir = base_dir / experiment_id
    experiment_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "configs").mkdir(exist_ok=True)
    
    print(f"🚀 开始快速实验: {experiment_id}")
    print(f"📁 实验目录: {experiment_dir}")
    
    # 读取并修改配置文件
    config_file = "configs/RGBNT201/MambaPro_moe.yml"
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 修改输出目录
    config_data["OUTPUT_DIR"] = str(experiment_dir / "logs")
    
    # 保存修改后的配置文件
    modified_config = experiment_dir / "configs" / "experiment_config.yml"
    with open(modified_config, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # 构建训练命令
    cmd = [
        "python", "train_net.py", "--config_file", str(modified_config),
        "MODEL.MOE_EXPERT_HIDDEN_DIM", "640",
        "MODEL.MOE_EXPERT_LAYERS", "1",
        "MODEL.MOE_TEMPERATURE", "0.7",
        "MODEL.MOE_EXPERT_DROPOUT", "0.18",
        "SOLVER.MOE_BALANCE_LOSS_WEIGHT", "0.0025"
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    # 记录实验信息
    experiment_info = {
        "experiment_id": experiment_id,
        "start_time": datetime.now().isoformat(),
        "experiment_dir": str(experiment_dir),
        "command": " ".join(cmd),
        "status": "running"
    }
    
    # 保存实验信息
    info_file = experiment_dir / "experiment_info.json"
    with open(info_file, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    try:
        # 运行训练
        print("🏃 开始训练...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 更新实验信息
        experiment_info["end_time"] = datetime.now().isoformat()
        experiment_info["return_code"] = result.returncode
        experiment_info["status"] = "completed" if result.returncode == 0 else "failed"
        
        # 保存训练输出
        with open(experiment_dir / "logs" / "training_output.txt", 'w') as f:
            f.write(result.stdout)
        
        if result.stderr:
            with open(experiment_dir / "logs" / "training_error.txt", 'w') as f:
                f.write(result.stderr)
        
        # 更新实验信息
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        # 复制模型权重
        if os.path.exists(experiment_dir / "logs" / "MambaProbest.pth"):
            shutil.copy2(
                experiment_dir / "logs" / "MambaProbest.pth",
                experiment_dir / "models" / "MambaProbest.pth"
            )
        
        print(f"✅ 实验完成: {experiment_id}")
        print(f"📁 结果目录: {experiment_dir}")
        print(f"📝 训练日志: {experiment_dir / 'logs' / 'train_log.txt'}")
        print(f"🏆 模型权重: {experiment_dir / 'models' / 'MambaProbest.pth'}")
        
        return experiment_dir, experiment_info
        
    except Exception as e:
        experiment_info["end_time"] = datetime.now().isoformat()
        experiment_info["status"] = "error"
        experiment_info["error"] = str(e)
        
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        print(f"❌ 实验失败: {e}")
        return experiment_dir, experiment_info

if __name__ == "__main__":
    run_quick_experiment()

