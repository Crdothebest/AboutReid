#!/usr/bin/env python3
"""
实验对比脚本
用于验证多尺度MoE模块的实验效果

作者修改：创建自动化实验脚本，对比基线模型和多尺度MoE模型
功能：自动运行基线实验和MoE实验，并生成对比报告
撤销方法：删除此文件
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
import argparse

def run_experiment(config_file, experiment_name, output_dir):
    """
    运行单个实验
    
    作者修改：封装实验运行逻辑，便于批量执行
    功能：运行指定配置的实验并记录结果
    撤销方法：删除此函数
    """
    print(f"\n🚀 开始运行实验: {experiment_name}")
    print(f"📁 配置文件: {config_file}")
    print(f"📊 输出目录: {output_dir}")
    print("=" * 60)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行训练命令
        cmd = [
            "python", "train_net.py",
            "--config_file", config_file
        ]
        
        print(f"🔧 执行命令: {' '.join(cmd)}")
        
        # 执行训练
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 保存实验结果
        experiment_result = {
            "experiment_name": experiment_name,
            "config_file": config_file,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        # 保存结果到JSON文件
        result_file = os.path.join(output_dir, f"{experiment_name}_result.json")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, indent=2, ensure_ascii=False)
        
        if result.returncode == 0:
            print(f"✅ 实验 {experiment_name} 成功完成!")
            print(f"⏱️  耗时: {duration/3600:.2f} 小时")
        else:
            print(f"❌ 实验 {experiment_name} 失败!")
            print(f"🔍 错误信息: {result.stderr}")
            
        return experiment_result
        
    except Exception as e:
        print(f"💥 实验 {experiment_name} 出现异常: {str(e)}")
        return None

def generate_comparison_report(results, output_dir):
    """
    生成对比报告
    
    作者修改：创建实验结果对比分析功能
    功能：分析基线实验和MoE实验的结果差异
    撤销方法：删除此函数
    """
    print("\n📊 生成实验对比报告...")
    
    report = {
        "experiment_summary": {},
        "comparison": {},
        "recommendations": []
    }
    
    # 分析每个实验的结果
    for result in results:
        if result is None:
            continue
            
        exp_name = result["experiment_name"]
        report["experiment_summary"][exp_name] = {
            "duration_hours": result["duration_hours"],
            "success": result["return_code"] == 0,
            "config_file": result["config_file"]
        }
    
    # 生成对比分析
    if len(results) >= 2:
        baseline_result = results[0]  # 基线实验
        moe_result = results[1]       # MoE实验
        
        if baseline_result and moe_result:
            report["comparison"] = {
                "baseline_duration": baseline_result["duration_hours"],
                "moe_duration": moe_result["duration_hours"],
                "time_difference": moe_result["duration_hours"] - baseline_result["duration_hours"],
                "time_ratio": moe_result["duration_hours"] / baseline_result["duration_hours"] if baseline_result["duration_hours"] > 0 else 0
            }
            
            # 生成建议
            if report["comparison"]["time_ratio"] > 1.2:
                report["recommendations"].append("MoE模型训练时间显著增加，建议优化计算效率")
            elif report["comparison"]["time_ratio"] < 0.8:
                report["recommendations"].append("MoE模型训练时间减少，效率提升明显")
            else:
                report["recommendations"].append("MoE模型训练时间与基线相近，效率良好")
    
    # 保存报告
    report_file = os.path.join(output_dir, "experiment_comparison_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 对比报告已保存到: {report_file}")
    
    # 打印简要报告
    print("\n" + "="*60)
    print("📊 实验对比报告")
    print("="*60)
    
    for exp_name, summary in report["experiment_summary"].items():
        status = "✅ 成功" if summary["success"] else "❌ 失败"
        print(f"{exp_name}: {status} | 耗时: {summary['duration_hours']:.2f} 小时")
    
    if report["comparison"]:
        print(f"\n⏱️  时间对比:")
        print(f"   基线模型: {report['comparison']['baseline_duration']:.2f} 小时")
        print(f"   MoE模型:  {report['comparison']['moe_duration']:.2f} 小时")
        print(f"   时间差异: {report['comparison']['time_difference']:+.2f} 小时")
        print(f"   效率比:   {report['comparison']['time_ratio']:.2f}x")
    
    if report["recommendations"]:
        print(f"\n💡 建议:")
        for rec in report["recommendations"]:
            print(f"   - {rec}")

def main():
    """
    主函数：执行完整的实验对比流程
    
    作者修改：创建主实验流程控制逻辑
    功能：按顺序运行基线实验和MoE实验，并生成对比报告
    撤销方法：删除此函数
    """
    parser = argparse.ArgumentParser(description="运行MambaPro实验对比")
    parser.add_argument("--baseline_config", default="configs/RGBNT201/MambaPro_baseline.yml", 
                       help="基线实验配置文件")
    parser.add_argument("--moe_config", default="configs/RGBNT201/MambaPro_moe.yml", 
                       help="MoE实验配置文件")
    parser.add_argument("--output_dir", default="experiment_results", 
                       help="实验结果输出目录")
    parser.add_argument("--skip_baseline", action="store_true", 
                       help="跳过基线实验")
    parser.add_argument("--skip_moe", action="store_true", 
                       help="跳过MoE实验")
    
    args = parser.parse_args()
    
    print("🧪 MambaPro 多尺度MoE 实验对比")
    print("=" * 60)
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 输出目录: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    
    # 运行基线实验
    if not args.skip_baseline:
        baseline_result = run_experiment(
            args.baseline_config, 
            "baseline_experiment", 
            args.output_dir
        )
        results.append(baseline_result)
    else:
        print("⏭️  跳过基线实验")
    
    # 运行MoE实验
    if not args.skip_moe:
        moe_result = run_experiment(
            args.moe_config, 
            "moe_experiment", 
            args.output_dir
        )
        results.append(moe_result)
    else:
        print("⏭️  跳过MoE实验")
    
    # 生成对比报告
    if results:
        generate_comparison_report(results, args.output_dir)
    
    print(f"\n🎉 实验对比完成!")
    print(f"📁 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
