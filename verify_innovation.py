#!/usr/bin/env python3
"""
创新点验证脚本
用于验证多尺度MoE创新点的效果

作者修改：创建创新点验证脚本，对比基线模型和创新点模型
功能：运行基线实验和创新点实验，生成对比报告
撤销方法：删除此文件
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
import argparse

def run_experiment(config_file, experiment_name, max_epochs=5):
    """
    运行单个实验（快速验证版本）
    
    作者修改：创建快速验证实验函数，使用较少epoch进行快速测试
    功能：运行指定配置的实验并记录结果
    撤销方法：删除此函数
    """
    print(f"\n🚀 开始运行实验: {experiment_name}")
    print(f"📁 配置文件: {config_file}")
    print(f"⏱️  最大轮数: {max_epochs} (快速验证)")
    print("=" * 60)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行训练命令，添加epoch限制
        cmd = [
            "python", "train_net.py",
            "--config_file", config_file,
            "SOLVER.MAX_EPOCHS", str(max_epochs)  # 限制训练轮数进行快速验证
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
            "max_epochs": max_epochs,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        # 保存结果到JSON文件
        result_file = f"{experiment_name}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, indent=2, ensure_ascii=False)
        
        if result.returncode == 0:
            print(f"✅ 实验 {experiment_name} 成功完成!")
            print(f"⏱️  耗时: {duration/60:.2f} 分钟")
        else:
            print(f"❌ 实验 {experiment_name} 失败!")
            print(f"🔍 错误信息: {result.stderr}")
            
        return experiment_result
        
    except Exception as e:
        print(f"💥 实验 {experiment_name} 出现异常: {str(e)}")
        return None

def extract_performance_metrics(stdout_text):
    """
    从训练日志中提取性能指标
    
    作者修改：创建性能指标提取函数
    功能：从训练输出中提取mAP、Rank-1等关键指标
    撤销方法：删除此函数
    """
    metrics = {}
    
    # 提取mAP
    if "mAP:" in stdout_text:
        try:
            map_line = [line for line in stdout_text.split('\n') if 'mAP:' in line][-1]
            map_value = float(map_line.split('mAP:')[1].split('%')[0].strip())
            metrics['mAP'] = map_value
        except:
            pass
    
    # 提取Rank-1
    if "Rank-1:" in stdout_text:
        try:
            rank1_line = [line for line in stdout_text.split('\n') if 'Rank-1:' in line][-1]
            rank1_value = float(rank1_line.split('Rank-1:')[1].split('%')[0].strip())
            metrics['Rank-1'] = rank1_value
        except:
            pass
    
    # 提取Rank-5
    if "Rank-5:" in stdout_text:
        try:
            rank5_line = [line for line in stdout_text.split('\n') if 'Rank-5:' in line][-1]
            rank5_value = float(rank5_line.split('Rank-5:')[1].split('%')[0].strip())
            metrics['Rank-5'] = rank5_value
        except:
            pass
    
    return metrics

def generate_innovation_report(baseline_result, innovation_result):
    """
    生成创新点验证报告
    
    作者修改：创建创新点验证报告生成函数
    功能：对比基线实验和创新点实验的结果
    撤销方法：删除此函数
    """
    print("\n📊 生成创新点验证报告...")
    
    # 提取性能指标
    baseline_metrics = extract_performance_metrics(baseline_result['stdout']) if baseline_result else {}
    innovation_metrics = extract_performance_metrics(innovation_result['stdout']) if innovation_result else {}
    
    # 生成报告
    report = {
        "experiment_summary": {
            "baseline": {
                "success": baseline_result['return_code'] == 0 if baseline_result else False,
                "duration_minutes": baseline_result['duration_minutes'] if baseline_result else 0,
                "metrics": baseline_metrics
            },
            "innovation": {
                "success": innovation_result['return_code'] == 0 if innovation_result else False,
                "duration_minutes": innovation_result['duration_minutes'] if innovation_result else 0,
                "metrics": innovation_metrics
            }
        },
        "performance_comparison": {},
        "innovation_effectiveness": {}
    }
    
    # 性能对比
    if baseline_metrics and innovation_metrics:
        report["performance_comparison"] = {
            "mAP": {
                "baseline": baseline_metrics.get('mAP', 0),
                "innovation": innovation_metrics.get('mAP', 0),
                "improvement": innovation_metrics.get('mAP', 0) - baseline_metrics.get('mAP', 0)
            },
            "Rank-1": {
                "baseline": baseline_metrics.get('Rank-1', 0),
                "innovation": innovation_metrics.get('Rank-1', 0),
                "improvement": innovation_metrics.get('Rank-1', 0) - baseline_metrics.get('Rank-1', 0)
            },
            "Rank-5": {
                "baseline": baseline_metrics.get('Rank-5', 0),
                "innovation": innovation_metrics.get('Rank-5', 0),
                "improvement": innovation_metrics.get('Rank-5', 0) - baseline_metrics.get('Rank-5', 0)
            }
        }
        
        # 创新点有效性分析
        mAP_improvement = report["performance_comparison"]["mAP"]["improvement"]
        rank1_improvement = report["performance_comparison"]["Rank-1"]["improvement"]
        
        if mAP_improvement > 0 and rank1_improvement > 0:
            report["innovation_effectiveness"] = {
                "status": "✅ 创新点有效",
                "mAP_improvement": f"+{mAP_improvement:.2f}%",
                "rank1_improvement": f"+{rank1_improvement:.2f}%",
                "conclusion": "多尺度MoE模块显著提升了模型性能"
            }
        elif mAP_improvement > 0 or rank1_improvement > 0:
            report["innovation_effectiveness"] = {
                "status": "⚠️ 创新点部分有效",
                "mAP_improvement": f"{mAP_improvement:+.2f}%",
                "rank1_improvement": f"{rank1_improvement:+.2f}%",
                "conclusion": "多尺度MoE模块在部分指标上有所提升"
            }
        else:
            report["innovation_effectiveness"] = {
                "status": "❌ 创新点无效",
                "mAP_improvement": f"{mAP_improvement:+.2f}%",
                "rank1_improvement": f"{rank1_improvement:+.2f}%",
                "conclusion": "多尺度MoE模块未带来性能提升，需要调优"
            }
    
    # 保存报告
    with open("innovation_verification_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    print("\n" + "="*60)
    print("📊 创新点验证报告")
    print("="*60)
    
    print(f"\n🔬 实验状态:")
    print(f"   基线实验: {'✅ 成功' if report['experiment_summary']['baseline']['success'] else '❌ 失败'}")
    print(f"   创新点实验: {'✅ 成功' if report['experiment_summary']['innovation']['success'] else '❌ 失败'}")
    
    if report["performance_comparison"]:
        print(f"\n📈 性能对比:")
        for metric, data in report["performance_comparison"].items():
            print(f"   {metric}:")
            print(f"     基线: {data['baseline']:.2f}%")
            print(f"     创新点: {data['innovation']:.2f}%")
            print(f"     提升: {data['improvement']:+.2f}%")
    
    if report["innovation_effectiveness"]:
        print(f"\n🎯 创新点有效性:")
        print(f"   状态: {report['innovation_effectiveness']['status']}")
        print(f"   mAP提升: {report['innovation_effectiveness']['mAP_improvement']}")
        print(f"   Rank-1提升: {report['innovation_effectiveness']['rank1_improvement']}")
        print(f"   结论: {report['innovation_effectiveness']['conclusion']}")
    
    print(f"\n📁 详细报告已保存到: innovation_verification_report.json")

def main():
    """
    主函数：执行创新点验证流程
    
    作者修改：创建创新点验证主流程
    功能：按顺序运行基线实验和创新点实验，并生成验证报告
    撤销方法：删除此函数
    """
    parser = argparse.ArgumentParser(description="验证多尺度MoE创新点效果")
    parser.add_argument("--max_epochs", type=int, default=5, 
                       help="快速验证的最大训练轮数")
    parser.add_argument("--skip_baseline", action="store_true", 
                       help="跳过基线实验")
    parser.add_argument("--skip_innovation", action="store_true", 
                       help="跳过创新点实验")
    
    args = parser.parse_args()
    
    print("🧪 多尺度MoE创新点验证")
    print("=" * 60)
    print(f"📅 验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  快速验证轮数: {args.max_epochs}")
    
    results = []
    
    # 运行基线实验
    if not args.skip_baseline:
        baseline_result = run_experiment(
            "configs/RGBNT201/MambaPro_baseline.yml", 
            "baseline_experiment", 
            args.max_epochs
        )
        results.append(baseline_result)
    else:
        print("⏭️  跳过基线实验")
        results.append(None)
    
    # 运行创新点实验
    if not args.skip_innovation:
        innovation_result = run_experiment(
            "configs/RGBNT201/MambaPro.yml", 
            "innovation_experiment", 
            args.max_epochs
        )
        results.append(innovation_result)
    else:
        print("⏭️  跳过创新点实验")
        results.append(None)
    
    # 生成验证报告
    if results[0] or results[1]:
        generate_innovation_report(results[0], results[1])
    
    print(f"\n🎉 创新点验证完成!")

if __name__ == "__main__":
    main()
