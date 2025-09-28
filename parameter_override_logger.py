#!/usr/bin/env python3
"""
参数覆盖记录脚本
功能：记录实验时实际使用的参数，包括命令行覆盖的参数
"""

import os
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path

def log_experiment_with_parameters(config_file, experiment_name, **override_params):
    """
    记录实验参数和结果
    
    Args:
        config_file (str): 配置文件路径
        experiment_name (str): 实验名称
        **override_params: 覆盖的参数
    """
    # 创建日志目录
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)
    
    # 读取配置文件
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"
    
    # 提取配置参数
    config_params = extract_config_parameters(config)
    
    # 应用参数覆盖
    effective_params = apply_parameter_overrides(config_params, override_params)
    
    # 构建实验记录
    experiment_record = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "start_time": datetime.now().isoformat(),
        "config_file": config_file,
        "parameter_overrides": override_params,
        "effective_parameters": effective_params,
        "status": "started"
    }
    
    # 保存实验记录
    record_file = log_dir / f"{experiment_id}.json"
    with open(record_file, 'w') as f:
        json.dump(experiment_record, f, indent=2)
    
    # 追加到总日志
    total_log = log_dir / "all_experiments.jsonl"
    with open(total_log, 'a') as f:
        f.write(json.dumps(experiment_record) + '\n')
    
    print(f"🚀 实验开始: {experiment_id}")
    print(f"📁 配置文件: {config_file}")
    print(f"⚙️  参数覆盖: {override_params}")
    print(f"📝 实验记录: {record_file}")
    
    return experiment_record

def extract_config_parameters(config):
    """提取配置文件参数"""
    return {
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
        "multi_scale": {
            "use_clip_multi_scale": config.get("MODEL", {}).get("USE_CLIP_MULTI_SCALE"),
            "clip_multi_scale_scales": config.get("MODEL", {}).get("CLIP_MULTI_SCALE_SCALES"),
        },
        "moe": {
            "use_multi_scale_moe": config.get("MODEL", {}).get("USE_MULTI_SCALE_MOE"),
            "moe_scales": config.get("MODEL", {}).get("MOE_SCALES"),
            "moe_expert_hidden_dim": config.get("MODEL", {}).get("MOE_EXPERT_HIDDEN_DIM"),
            "moe_temperature": config.get("MODEL", {}).get("MOE_TEMPERATURE"),
        },
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
        "data": {
            "dataset": config.get("DATASETS", {}).get("NAMES"),
            "size_train": config.get("INPUT", {}).get("SIZE_TRAIN"),
            "size_test": config.get("INPUT", {}).get("SIZE_TEST"),
            "num_instance": config.get("DATALOADER", {}).get("NUM_INSTANCE"),
            "num_workers": config.get("DATALOADER", {}).get("NUM_WORKERS"),
        }
    }

def apply_parameter_overrides(config_params, override_params):
    """应用参数覆盖"""
    effective_params = config_params.copy()
    
    # 处理常见的参数覆盖
    if "use_moe" in override_params:
        effective_params["moe"]["use_multi_scale_moe"] = override_params["use_moe"]
    if "disable_moe" in override_params:
        effective_params["moe"]["use_multi_scale_moe"] = not override_params["disable_moe"]
    if "use_multi_scale" in override_params:
        effective_params["multi_scale"]["use_clip_multi_scale"] = override_params["use_multi_scale"]
    if "base_lr" in override_params:
        effective_params["training"]["base_lr"] = override_params["base_lr"]
    if "max_epochs" in override_params:
        effective_params["training"]["max_epochs"] = override_params["max_epochs"]
    if "ims_per_batch" in override_params:
        effective_params["training"]["ims_per_batch"] = override_params["ims_per_batch"]
    if "moe_expert_hidden_dim" in override_params:
        effective_params["moe"]["moe_expert_hidden_dim"] = override_params["moe_expert_hidden_dim"]
    if "moe_temperature" in override_params:
        effective_params["moe"]["moe_temperature"] = override_params["moe_temperature"]
    
    return effective_params

def update_experiment_results(experiment_id, results):
    """更新实验结果"""
    log_dir = Path("experiment_logs")
    record_file = log_dir / f"{experiment_id}.json"
    
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
    parser = argparse.ArgumentParser(description="参数覆盖记录脚本")
    parser.add_argument("config_file", help="配置文件路径")
    parser.add_argument("experiment_name", help="实验名称")
    parser.add_argument("--use_moe", action="store_true", help="启用MoE")
    parser.add_argument("--disable_moe", action="store_true", help="禁用MoE")
    parser.add_argument("--use_multi_scale", action="store_true", help="启用多尺度")
    parser.add_argument("--base_lr", type=float, help="学习率")
    parser.add_argument("--max_epochs", type=int, help="训练轮数")
    parser.add_argument("--ims_per_batch", type=int, help="批次大小")
    parser.add_argument("--moe_expert_hidden_dim", type=int, help="MoE专家隐藏层维度")
    parser.add_argument("--moe_temperature", type=float, help="MoE温度参数")
    parser.add_argument("--output_dir", help="输出目录")
    
    args = parser.parse_args()
    
    # 提取参数覆盖
    override_params = {}
    if args.use_moe:
        override_params["use_moe"] = True
    if args.disable_moe:
        override_params["disable_moe"] = True
    if args.use_multi_scale:
        override_params["use_multi_scale"] = True
    if args.base_lr:
        override_params["base_lr"] = args.base_lr
    if args.max_epochs:
        override_params["max_epochs"] = args.max_epochs
    if args.ims_per_batch:
        override_params["ims_per_batch"] = args.ims_per_batch
    if args.moe_expert_hidden_dim:
        override_params["moe_expert_hidden_dim"] = args.moe_expert_hidden_dim
    if args.moe_temperature:
        override_params["moe_temperature"] = args.moe_temperature
    if args.output_dir:
        override_params["output_dir"] = args.output_dir
    
    # 记录实验
    experiment_record = log_experiment_with_parameters(
        args.config_file, 
        args.experiment_name,
        **override_params
    )
    
    print(f"📝 实验记录已保存: {experiment_record['experiment_id']}.json")

if __name__ == "__main__":
    main()
