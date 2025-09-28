#!/usr/bin/env python3
"""
MoE超参数调优系统
功能：系统性地测试MoE参数组合，找到最佳配置
作者：MoE调优系统
日期：2024
"""

import os
import json
import yaml
import time
import subprocess
import itertools
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

class MoEHyperparameterTuner:
    """MoE超参数调优器"""
    
    def __init__(self, base_config="configs/RGBNT201/MambaPro_moe.yml", log_dir="moe_tuning_logs"):
        """
        初始化MoE调优器
        
        Args:
            base_config (str): 基础配置文件路径
            log_dir (str): 调优日志目录
        """
        self.base_config = base_config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 读取基础配置
        with open(base_config, 'r') as f:
            self.base_config_data = yaml.safe_load(f)
        
        # 实验结果记录
        self.experiment_results = []
        self.best_result = None
        
    def define_parameter_space(self):
        """定义MoE参数搜索空间"""
        return {
            # ✨ MoE核心参数
            "MOE_EXPERT_HIDDEN_DIM": [512, 1024, 2048],  # 专家网络隐藏层维度
            "MOE_TEMPERATURE": [0.3, 0.5, 0.7, 1.0, 1.5],  # 门控网络温度参数
            "MOE_EXPERT_DROPOUT": [0.0, 0.1, 0.2],  # 专家网络Dropout比例
            "MOE_GATE_DROPOUT": [0.0, 0.1, 0.2],  # 门控网络Dropout比例
            
            # ✨ MoE网络结构参数
            "MOE_EXPERT_LAYERS": [1, 2, 3],  # 专家网络层数
            "MOE_GATE_LAYERS": [1, 2],  # 门控网络层数
            "MOE_EXPERT_THRESHOLD": [0.0, 0.1, 0.2],  # 专家激活阈值
            "MOE_RESIDUAL_WEIGHT": [0.5, 1.0],  # 残差连接权重
            
            # ✨ MoE损失权重参数
            "MOE_BALANCE_LOSS_WEIGHT": [0.0, 0.01, 0.1],  # 专家平衡损失权重
            "MOE_SPARSITY_LOSS_WEIGHT": [0.0, 0.001, 0.01],  # 稀疏性损失权重
            "MOE_DIVERSITY_LOSS_WEIGHT": [0.0, 0.01, 0.1],  # 多样性损失权重
        }
    
    def create_parameter_combinations(self, strategy="grid_search", max_combinations=50):
        """
        创建参数组合
        
        Args:
            strategy (str): 搜索策略 ("grid_search", "random_search", "focused_search")
            max_combinations (int): 最大组合数
        """
        param_space = self.define_parameter_space()
        
        if strategy == "grid_search":
            # 网格搜索：测试所有参数组合
            param_names = list(param_space.keys())
            param_values = list(param_space.values())
            combinations = list(itertools.product(*param_values))
            
            # 限制组合数量
            if len(combinations) > max_combinations:
                # 随机采样
                import random
                combinations = random.sample(combinations, max_combinations)
                
        elif strategy == "random_search":
            # 随机搜索：随机选择参数组合
            combinations = []
            for _ in range(max_combinations):
                combination = {}
                for param_name, param_values in param_space.items():
                    combination[param_name] = np.random.choice(param_values)
                combinations.append(combination)
                
        elif strategy == "focused_search":
            # 聚焦搜索：基于经验的重点参数组合
            combinations = self._create_focused_combinations()
            
        else:
            raise ValueError(f"未知的搜索策略: {strategy}")
        
        print(f"🔍 生成 {len(combinations)} 个参数组合 (策略: {strategy})")
        return combinations
    
    def _create_focused_combinations(self):
        """创建聚焦搜索的参数组合"""
        # 基于MoE调优经验的重点组合
        focused_combinations = [
            # 组合1：基础配置
            {
                "MOE_EXPERT_HIDDEN_DIM": 1024,
                "MOE_TEMPERATURE": 0.7,
                "MOE_EXPERT_DROPOUT": 0.1,
                "MOE_GATE_DROPOUT": 0.1,
                "MOE_EXPERT_LAYERS": 2,
                "MOE_GATE_LAYERS": 2,
                "MOE_EXPERT_THRESHOLD": 0.1,
                "MOE_RESIDUAL_WEIGHT": 1.0,
                "MOE_BALANCE_LOSS_WEIGHT": 0.01,
                "MOE_SPARSITY_LOSS_WEIGHT": 0.001,
                "MOE_DIVERSITY_LOSS_WEIGHT": 0.01,
            },
            # 组合2：高容量专家
            {
                "MOE_EXPERT_HIDDEN_DIM": 2048,
                "MOE_TEMPERATURE": 0.5,
                "MOE_EXPERT_DROPOUT": 0.1,
                "MOE_GATE_DROPOUT": 0.1,
                "MOE_EXPERT_LAYERS": 3,
                "MOE_GATE_LAYERS": 2,
                "MOE_EXPERT_THRESHOLD": 0.1,
                "MOE_RESIDUAL_WEIGHT": 1.0,
                "MOE_BALANCE_LOSS_WEIGHT": 0.01,
                "MOE_SPARSITY_LOSS_WEIGHT": 0.001,
                "MOE_DIVERSITY_LOSS_WEIGHT": 0.01,
            },
            # 组合3：低温度配置
            {
                "MOE_EXPERT_HIDDEN_DIM": 1024,
                "MOE_TEMPERATURE": 0.3,
                "MOE_EXPERT_DROPOUT": 0.0,
                "MOE_GATE_DROPOUT": 0.0,
                "MOE_EXPERT_LAYERS": 2,
                "MOE_GATE_LAYERS": 1,
                "MOE_EXPERT_THRESHOLD": 0.0,
                "MOE_RESIDUAL_WEIGHT": 0.5,
                "MOE_BALANCE_LOSS_WEIGHT": 0.1,
                "MOE_SPARSITY_LOSS_WEIGHT": 0.01,
                "MOE_DIVERSITY_LOSS_WEIGHT": 0.1,
            },
            # 组合4：高温度配置
            {
                "MOE_EXPERT_HIDDEN_DIM": 1024,
                "MOE_TEMPERATURE": 1.5,
                "MOE_EXPERT_DROPOUT": 0.2,
                "MOE_GATE_DROPOUT": 0.2,
                "MOE_EXPERT_LAYERS": 2,
                "MOE_GATE_LAYERS": 2,
                "MOE_EXPERT_THRESHOLD": 0.2,
                "MOE_RESIDUAL_WEIGHT": 1.0,
                "MOE_BALANCE_LOSS_WEIGHT": 0.0,
                "MOE_SPARSITY_LOSS_WEIGHT": 0.0,
                "MOE_DIVERSITY_LOSS_WEIGHT": 0.0,
            },
            # 组合5：平衡配置
            {
                "MOE_EXPERT_HIDDEN_DIM": 1024,
                "MOE_TEMPERATURE": 1.0,
                "MOE_EXPERT_DROPOUT": 0.1,
                "MOE_GATE_DROPOUT": 0.1,
                "MOE_EXPERT_LAYERS": 2,
                "MOE_GATE_LAYERS": 2,
                "MOE_EXPERT_THRESHOLD": 0.1,
                "MOE_RESIDUAL_WEIGHT": 1.0,
                "MOE_BALANCE_LOSS_WEIGHT": 0.01,
                "MOE_SPARSITY_LOSS_WEIGHT": 0.001,
                "MOE_DIVERSITY_LOSS_WEIGHT": 0.01,
            }
        ]
        
        return focused_combinations
    
    def create_experiment_config(self, param_combination, experiment_id):
        """
        为实验创建配置文件
        
        Args:
            param_combination (dict): 参数组合
            experiment_id (str): 实验ID
        """
        # 复制基础配置
        config_data = self.base_config_data.copy()
        
        # 应用参数组合
        for param_name, param_value in param_combination.items():
            if param_name in config_data.get("MODEL", {}):
                config_data["MODEL"][param_name] = param_value
            elif param_name in config_data.get("SOLVER", {}):
                config_data["SOLVER"][param_name] = param_value
        
        # 设置实验输出目录
        config_data["OUTPUT_DIR"] = f"/home/zubuntu/workspace/yzy/MambaPro/outputs/moe_tuning_{experiment_id}"
        
        # 保存配置文件
        config_file = self.log_dir / f"config_{experiment_id}.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        return str(config_file)
    
    def run_single_experiment(self, param_combination, experiment_id):
        """
        运行单个实验
        
        Args:
            param_combination (dict): 参数组合
            experiment_id (str): 实验ID
        """
        print(f"\n🚀 开始实验 {experiment_id}")
        print(f"📊 参数组合: {param_combination}")
        
        # 创建配置文件
        config_file = self.create_experiment_config(param_combination, experiment_id)
        
        # 记录实验开始
        start_time = time.time()
        experiment_record = {
            "experiment_id": experiment_id,
            "start_time": datetime.now().isoformat(),
            "config_file": config_file,
            "parameters": param_combination,
            "status": "running"
        }
        
        try:
            # 运行训练命令
            cmd = f"python train_net.py --config_file {config_file}"
            print(f"🔧 执行命令: {cmd}")
            
            # 执行训练
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)  # 2小时超时
            
            if result.returncode == 0:
                # 解析结果
                results = self.parse_training_results(result.stdout)
                experiment_record.update({
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": time.time() - start_time,
                    "status": "completed",
                    "results": results
                })
                
                print(f"✅ 实验完成: mAP={results.get('mAP', 'N/A')}%, Rank-1={results.get('Rank-1', 'N/A')}%")
                
            else:
                # 实验失败
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
        self.experiment_results.append(experiment_record)
        self.save_experiment_record(experiment_record)
        
        return experiment_record
    
    def parse_training_results(self, output):
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
        
        # 查找其他指标
        if "Rank-5:" in output:
            try:
                rank5_line = [line for line in output.split('\n') if 'Rank-5:' in line][-1]
                rank5_value = float(rank5_line.split('Rank-5:')[1].split('%')[0].strip())
                results['Rank-5'] = rank5_value
            except:
                results['Rank-5'] = None
        
        return results
    
    def save_experiment_record(self, experiment_record):
        """保存实验记录"""
        record_file = self.log_dir / f"experiment_{experiment_record['experiment_id']}.json"
        with open(record_file, 'w') as f:
            json.dump(experiment_record, f, indent=2)
    
    def run_hyperparameter_tuning(self, strategy="focused_search", max_experiments=10):
        """
        运行超参数调优
        
        Args:
            strategy (str): 搜索策略
            max_experiments (int): 最大实验数
        """
        print(f"🔍 开始MoE超参数调优 (策略: {strategy}, 最大实验数: {max_experiments})")
        
        # 创建参数组合
        param_combinations = self.create_parameter_combinations(strategy, max_experiments)
        
        # 运行实验
        for i, param_combination in enumerate(param_combinations):
            experiment_id = f"moe_tuning_{i+1:03d}"
            experiment_record = self.run_single_experiment(param_combination, experiment_id)
            
            # 更新最佳结果
            if experiment_record.get("status") == "completed" and experiment_record.get("results"):
                results = experiment_record["results"]
                if results.get("mAP") and (self.best_result is None or results["mAP"] > self.best_result.get("mAP", 0)):
                    self.best_result = experiment_record
                    print(f"🏆 新的最佳结果: mAP={results['mAP']}%")
        
        # 生成调优报告
        self.generate_tuning_report()
        
        return self.best_result
    
    def generate_tuning_report(self):
        """生成调优报告"""
        # 创建结果DataFrame
        df_data = []
        for record in self.experiment_results:
            if record.get("status") == "completed" and record.get("results"):
                row = {
                    "experiment_id": record["experiment_id"],
                    "mAP": record["results"].get("mAP"),
                    "Rank-1": record["results"].get("Rank-1"),
                    "Rank-5": record["results"].get("Rank-5"),
                    "duration_hours": record.get("duration_seconds", 0) / 3600
                }
                # 添加参数
                for param_name, param_value in record.get("parameters", {}).items():
                    row[param_name] = param_value
                df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # 保存结果CSV
            results_csv = self.log_dir / "tuning_results.csv"
            df.to_csv(results_csv, index=False)
            
            # 生成报告
            report = {
                "tuning_summary": {
                    "total_experiments": len(self.experiment_results),
                    "completed_experiments": len(df_data),
                    "best_mAP": df["mAP"].max() if "mAP" in df.columns else None,
                    "best_Rank-1": df["Rank-1"].max() if "Rank-1" in df.columns else None,
                    "best_experiment_id": df.loc[df["mAP"].idxmax(), "experiment_id"] if "mAP" in df.columns else None
                },
                "best_parameters": self.best_result.get("parameters", {}) if self.best_result else {},
                "experiment_results": df_data
            }
            
            # 保存报告
            report_file = self.log_dir / "tuning_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📊 调优报告已生成:")
            print(f"   - 总实验数: {len(self.experiment_results)}")
            print(f"   - 完成实验数: {len(df_data)}")
            print(f"   - 最佳mAP: {report['tuning_summary']['best_mAP']}%")
            print(f"   - 最佳Rank-1: {report['tuning_summary']['best_Rank-1']}%")
            print(f"   - 结果文件: {results_csv}")
            print(f"   - 报告文件: {report_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE超参数调优系统")
    parser.add_argument("--strategy", choices=["grid_search", "random_search", "focused_search"], 
                       default="focused_search", help="搜索策略")
    parser.add_argument("--max_experiments", type=int, default=10, help="最大实验数")
    parser.add_argument("--base_config", default="configs/RGBNT201/MambaPro_moe.yml", help="基础配置文件")
    
    args = parser.parse_args()
    
    # 创建调优器
    tuner = MoEHyperparameterTuner(args.base_config)
    
    # 运行调优
    best_result = tuner.run_hyperparameter_tuning(args.strategy, args.max_experiments)
    
    if best_result:
        print(f"\n🏆 最佳配置:")
        print(f"   实验ID: {best_result['experiment_id']}")
        print(f"   mAP: {best_result['results'].get('mAP')}%")
        print(f"   Rank-1: {best_result['results'].get('Rank-1')}%")
        print(f"   参数: {best_result['parameters']}")

if __name__ == "__main__":
    main()
