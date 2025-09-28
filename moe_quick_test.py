#!/usr/bin/env python3
"""
MoE快速测试脚本
功能：快速测试单个MoE参数组合
作者：MoE测试系统
日期：2024
"""

import os
import json
import yaml
import time
import subprocess
from datetime import datetime
from pathlib import Path

def quick_moe_test(experiment_name, **moe_params):
    """
    快速MoE测试
    
    Args:
        experiment_name (str): 实验名称
        **moe_params: MoE参数
    """
    print(f"🚀 开始MoE快速测试: {experiment_name}")
    
    # 创建日志目录
    log_dir = Path("moe_quick_tests")
    log_dir.mkdir(exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"
    
    # 读取基础配置
    base_config = "configs/RGBNT201/MambaPro_moe.yml"
    with open(base_config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 应用MoE参数
    for param_name, param_value in moe_params.items():
        if param_name in config_data.get("MODEL", {}):
            config_data["MODEL"][param_name] = param_value
        elif param_name in config_data.get("SOLVER", {}):
            config_data["SOLVER"][param_name] = param_value
    
    # 设置输出目录
    config_data["OUTPUT_DIR"] = f"/home/zubuntu/workspace/yzy/MambaPro/outputs/moe_quick_{experiment_id}"
    
    # 保存配置文件
    config_file = log_dir / f"config_{experiment_id}.yml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # 记录实验开始
    start_time = time.time()
    experiment_record = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "start_time": datetime.now().isoformat(),
        "config_file": str(config_file),
        "moe_parameters": moe_params,
        "status": "running"
    }
    
    print(f"📊 MoE参数: {moe_params}")
    print(f"📁 配置文件: {config_file}")
    
    try:
        # 运行训练
        cmd = f"python train_net.py --config_file {config_file}"
        print(f"🔧 执行命令: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            # 解析结果
            results = parse_training_results(result.stdout)
            experiment_record.update({
                "end_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "status": "completed",
                "results": results
            })
            
            print(f"✅ 实验完成:")
            print(f"   mAP: {results.get('mAP', 'N/A')}%")
            print(f"   Rank-1: {results.get('Rank-1', 'N/A')}%")
            print(f"   Rank-5: {results.get('Rank-5', 'N/A')}%")
            print(f"   训练时间: {(time.time() - start_time)/3600:.2f} 小时")
            
        else:
            experiment_record.update({
                "end_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "status": "failed",
                "error": result.stderr
            })
            print(f"❌ 实验失败: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        experiment_record.update({
            "end_time": datetime.now().isoformat(),
            "duration_seconds": time.time() - start_time,
            "status": "timeout",
            "error": "Training timeout (2 hours)"
        })
        print(f"⏰ 实验超时")
        
    except Exception as e:
        experiment_record.update({
            "end_time": datetime.now().isoformat(),
            "duration_seconds": time.time() - start_time,
            "status": "error",
            "error": str(e)
        })
        print(f"❌ 实验错误: {str(e)}")
    
    # 保存实验记录
    record_file = log_dir / f"experiment_{experiment_id}.json"
    with open(record_file, 'w') as f:
        json.dump(experiment_record, f, indent=2)
    
    # 追加到总日志
    total_log = log_dir / "all_quick_tests.jsonl"
    with open(total_log, 'a') as f:
        f.write(json.dumps(experiment_record) + '\n')
    
    print(f"📝 实验记录已保存: {record_file}")
    return experiment_record

def parse_training_results(output):
    """解析训练结果"""
    results = {}
    
    # 查找mAP指标
    if "mAP:" in output:
        try:
            mAP_line = [line for line in output.split('\n') if 'mAP:' in line][-1]
            mAP_value = float(mAP_line.split('mAP:')[1].split('%')[0].strip())
            results['mAP'] = mAP_value
        except:
            results['mAP'] = None
    
    # 查找Rank-1指标
    if "Rank-1:" in output:
        try:
            rank1_line = [line for line in output.split('\n') if 'Rank-1:' in line][-1]
            rank1_value = float(rank1_line.split('Rank-1:')[1].split('%')[0].strip())
            results['Rank-1'] = rank1_value
        except:
            results['Rank-1'] = None
    
    # 查找Rank-5指标
    if "Rank-5:" in output:
        try:
            rank5_line = [line for line in output.split('\n') if 'Rank-5:' in line][-1]
            rank5_value = float(rank5_line.split('Rank-5:')[1].split('%')[0].strip())
            results['Rank-5'] = rank5_value
        except:
            results['Rank-5'] = None
    
    return results

def create_quick_test_summary():
    """创建快速测试总结"""
    log_dir = Path("moe_quick_tests")
    total_log = log_dir / "all_quick_tests.jsonl"
    
    if not total_log.exists():
        print("❌ 没有找到快速测试记录")
        return
    
    # 读取所有测试记录
    tests = []
    with open(total_log, 'r') as f:
        for line in f:
            tests.append(json.loads(line))
    
    # 创建总结
    summary = {
        "total_tests": len(tests),
        "completed_tests": len([t for t in tests if t.get("status") == "completed"]),
        "tests": tests,
        "summary_time": datetime.now().isoformat()
    }
    
    summary_file = log_dir / "quick_test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 快速测试总结已创建: {summary_file}")
    return summary_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python moe_quick_test.py <experiment_name> [参数名=参数值 ...]")
        print("")
        print("示例:")
        print("  python moe_quick_test.py test1 MOE_EXPERT_HIDDEN_DIM=1024 MOE_TEMPERATURE=0.7")
        print("  python moe_quick_test.py test2 MOE_EXPERT_HIDDEN_DIM=2048 MOE_TEMPERATURE=0.5")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    moe_params = {}
    
    # 解析参数
    for arg in sys.argv[2:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # 尝试转换数据类型
            try:
                if value.lower() in ['true', 'false']:
                    moe_params[key] = value.lower() == 'true'
                elif '.' in value:
                    moe_params[key] = float(value)
                else:
                    moe_params[key] = int(value)
            except ValueError:
                moe_params[key] = value
    
    # 运行快速测试
    result = quick_moe_test(experiment_name, **moe_params)
    
    # 创建总结
    create_quick_test_summary()
