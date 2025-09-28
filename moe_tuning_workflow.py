#!/usr/bin/env python3
"""
MoE调参工作流程脚本
功能：指导用户进行系统性的MoE参数调优
作者：MoE调优指导系统
日期：2024
"""

import os
import json
import yaml
import time
import subprocess
from datetime import datetime
from pathlib import Path

class MoETuningWorkflow:
    """MoE调参工作流程指导器"""
    
    def __init__(self, base_config="configs/RGBNT201/MambaPro_moe.yml"):
        """初始化调参工作流程"""
        self.base_config = base_config
        self.tuning_log = Path("moe_tuning_workflow.json")
        self.current_stage = 1
        self.best_config = None
        self.best_performance = 0.0
        
        # 加载或创建调优日志
        if self.tuning_log.exists():
            with open(self.tuning_log, 'r') as f:
                self.workflow_data = json.load(f)
        else:
            self.workflow_data = {
                "tuning_start_time": datetime.now().isoformat(),
                "stages": {},
                "best_config": None,
                "best_performance": 0.0
            }
    
    def start_tuning_workflow(self):
        """开始调参工作流程"""
        print("🚀 开始MoE调参工作流程")
        print("=" * 50)
        
        # 显示调参计划
        self.show_tuning_plan()
        
        # 开始第1阶段
        self.stage1_core_parameters()
    
    def show_tuning_plan(self):
        """显示调参计划"""
        print("📋 MoE调参计划:")
        print("阶段1: 核心参数调优 (专家维度 + 温度参数)")
        print("阶段2: 网络结构调优 (专家层数 + 门控层数)")
        print("阶段3: 正则化调优 (Dropout参数)")
        print("阶段4: 损失权重调优 (平衡损失 + 稀疏性损失)")
        print("阶段5: 最佳配置验证")
        print("=" * 50)
    
    def stage1_core_parameters(self):
        """阶段1: 核心参数调优"""
        print("\n🔧 阶段1: 核心参数调优")
        print("目标: 找到最佳的专家维度和温度参数")
        print("=" * 30)
        
        # 专家维度测试
        print("📊 测试专家维度...")
        expert_dim_tests = [
            ("expert_512", {"MOE_EXPERT_HIDDEN_DIM": 512}),
            ("expert_1024", {"MOE_EXPERT_HIDDEN_DIM": 1024}),
            ("expert_2048", {"MOE_EXPERT_HIDDEN_DIM": 2048})
        ]
        
        expert_results = []
        for test_name, params in expert_dim_tests:
            print(f"  测试 {test_name}: 专家维度={params['MOE_EXPERT_HIDDEN_DIM']}")
            result = self.run_single_test(test_name, params)
            expert_results.append(result)
        
        # 温度参数测试
        print("\n📊 测试温度参数...")
        temperature_tests = [
            ("temp_03", {"MOE_TEMPERATURE": 0.3}),
            ("temp_05", {"MOE_TEMPERATURE": 0.5}),
            ("temp_07", {"MOE_TEMPERATURE": 0.7}),
            ("temp_10", {"MOE_TEMPERATURE": 1.0}),
            ("temp_15", {"MOE_TEMPERATURE": 1.5})
        ]
        
        temperature_results = []
        for test_name, params in temperature_tests:
            print(f"  测试 {test_name}: 温度={params['MOE_TEMPERATURE']}")
            result = self.run_single_test(test_name, params)
            temperature_results.append(result)
        
        # 分析结果
        self.analyze_stage1_results(expert_results, temperature_results)
        
        # 进入下一阶段
        self.stage2_network_structure()
    
    def stage2_network_structure(self):
        """阶段2: 网络结构调优"""
        print("\n🔧 阶段2: 网络结构调优")
        print("目标: 找到最佳的专家层数和门控层数")
        print("=" * 30)
        
        # 专家层数测试
        print("📊 测试专家层数...")
        expert_layer_tests = [
            ("expert_layers_1", {"MOE_EXPERT_LAYERS": 1}),
            ("expert_layers_2", {"MOE_EXPERT_LAYERS": 2}),
            ("expert_layers_3", {"MOE_EXPERT_LAYERS": 3})
        ]
        
        expert_layer_results = []
        for test_name, params in expert_layer_tests:
            print(f"  测试 {test_name}: 专家层数={params['MOE_EXPERT_LAYERS']}")
            result = self.run_single_test(test_name, params)
            expert_layer_results.append(result)
        
        # 门控层数测试
        print("\n📊 测试门控层数...")
        gate_layer_tests = [
            ("gate_layers_1", {"MOE_GATE_LAYERS": 1}),
            ("gate_layers_2", {"MOE_GATE_LAYERS": 2})
        ]
        
        gate_layer_results = []
        for test_name, params in gate_layer_tests:
            print(f"  测试 {test_name}: 门控层数={params['MOE_GATE_LAYERS']}")
            result = self.run_single_test(test_name, params)
            gate_layer_results.append(result)
        
        # 分析结果
        self.analyze_stage2_results(expert_layer_results, gate_layer_results)
        
        # 进入下一阶段
        self.stage3_regularization()
    
    def stage3_regularization(self):
        """阶段3: 正则化调优"""
        print("\n🔧 阶段3: 正则化调优")
        print("目标: 找到最佳的Dropout参数")
        print("=" * 30)
        
        # 专家Dropout测试
        print("📊 测试专家Dropout...")
        expert_dropout_tests = [
            ("expert_dropout_0", {"MOE_EXPERT_DROPOUT": 0.0}),
            ("expert_dropout_01", {"MOE_EXPERT_DROPOUT": 0.1}),
            ("expert_dropout_02", {"MOE_EXPERT_DROPOUT": 0.2})
        ]
        
        expert_dropout_results = []
        for test_name, params in expert_dropout_tests:
            print(f"  测试 {test_name}: 专家Dropout={params['MOE_EXPERT_DROPOUT']}")
            result = self.run_single_test(test_name, params)
            expert_dropout_results.append(result)
        
        # 门控Dropout测试
        print("\n📊 测试门控Dropout...")
        gate_dropout_tests = [
            ("gate_dropout_0", {"MOE_GATE_DROPOUT": 0.0}),
            ("gate_dropout_01", {"MOE_GATE_DROPOUT": 0.1}),
            ("gate_dropout_02", {"MOE_GATE_DROPOUT": 0.2})
        ]
        
        gate_dropout_results = []
        for test_name, params in gate_dropout_tests:
            print(f"  测试 {test_name}: 门控Dropout={params['MOE_GATE_DROPOUT']}")
            result = self.run_single_test(test_name, params)
            gate_dropout_results.append(result)
        
        # 分析结果
        self.analyze_stage3_results(expert_dropout_results, gate_dropout_results)
        
        # 进入下一阶段
        self.stage4_loss_weights()
    
    def stage4_loss_weights(self):
        """阶段4: 损失权重调优"""
        print("\n🔧 阶段4: 损失权重调优")
        print("目标: 找到最佳的损失权重参数")
        print("=" * 30)
        
        # 平衡损失权重测试
        print("📊 测试平衡损失权重...")
        balance_loss_tests = [
            ("balance_0", {"MOE_BALANCE_LOSS_WEIGHT": 0.0}),
            ("balance_001", {"MOE_BALANCE_LOSS_WEIGHT": 0.01}),
            ("balance_01", {"MOE_BALANCE_LOSS_WEIGHT": 0.1})
        ]
        
        balance_loss_results = []
        for test_name, params in balance_loss_tests:
            print(f"  测试 {test_name}: 平衡损失权重={params['MOE_BALANCE_LOSS_WEIGHT']}")
            result = self.run_single_test(test_name, params)
            balance_loss_results.append(result)
        
        # 稀疏性损失权重测试
        print("\n📊 测试稀疏性损失权重...")
        sparsity_loss_tests = [
            ("sparsity_0", {"MOE_SPARSITY_LOSS_WEIGHT": 0.0}),
            ("sparsity_001", {"MOE_SPARSITY_LOSS_WEIGHT": 0.001}),
            ("sparsity_01", {"MOE_SPARSITY_LOSS_WEIGHT": 0.01})
        ]
        
        sparsity_loss_results = []
        for test_name, params in sparsity_loss_tests:
            print(f"  测试 {test_name}: 稀疏性损失权重={params['MOE_SPARSITY_LOSS_WEIGHT']}")
            result = self.run_single_test(test_name, params)
            sparsity_loss_results.append(result)
        
        # 分析结果
        self.analyze_stage4_results(balance_loss_results, sparsity_loss_results)
        
        # 进入下一阶段
        self.stage5_final_validation()
    
    def stage5_final_validation(self):
        """阶段5: 最佳配置验证"""
        print("\n🔧 阶段5: 最佳配置验证")
        print("目标: 验证最佳配置的性能")
        print("=" * 30)
        
        if self.best_config:
            print(f"📊 验证最佳配置: {self.best_config}")
            result = self.run_single_test("final_validation", self.best_config)
            
            print(f"\n🏆 最终结果:")
            print(f"   最佳配置: {self.best_config}")
            print(f"   最佳性能: mAP={result.get('mAP', 'N/A')}%, Rank-1={result.get('Rank-1', 'N/A')}%")
            
            # 保存最终结果
            self.save_final_results()
        else:
            print("❌ 没有找到最佳配置")
    
    def run_single_test(self, test_name, params):
        """运行单个测试"""
        print(f"🚀 运行测试: {test_name}")
        
        # 创建配置文件
        config_file = self.create_test_config(test_name, params)
        
        # 运行训练
        cmd = f"python train_net.py --config_file {config_file}"
        print(f"   执行命令: {cmd}")
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                # 解析结果
                results = self.parse_training_results(result.stdout)
                print(f"   ✅ 测试完成: mAP={results.get('mAP', 'N/A')}%, Rank-1={results.get('Rank-1', 'N/A')}%")
                
                # 更新最佳配置
                if results.get('mAP', 0) > self.best_performance:
                    self.best_performance = results.get('mAP', 0)
                    self.best_config = params.copy()
                    print(f"   🏆 新的最佳配置!")
                
                return results
            else:
                print(f"   ❌ 测试失败: {result.stderr}")
                return {"error": result.stderr}
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ 测试超时")
            return {"error": "timeout"}
        except Exception as e:
            print(f"   ❌ 测试错误: {str(e)}")
            return {"error": str(e)}
    
    def create_test_config(self, test_name, params):
        """创建测试配置文件"""
        # 读取基础配置
        with open(self.base_config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # 应用参数
        for param_name, param_value in params.items():
            if param_name in config_data.get("MODEL", {}):
                config_data["MODEL"][param_name] = param_value
            elif param_name in config_data.get("SOLVER", {}):
                config_data["SOLVER"][param_name] = param_value
        
        # 设置输出目录
        config_data["OUTPUT_DIR"] = f"/home/zubuntu/workspace/yzy/MambaPro/outputs/moe_workflow_{test_name}"
        
        # 保存配置文件
        config_file = f"configs/RGBNT201/moe_workflow_{test_name}.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        return config_file
    
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
        
        return results
    
    def analyze_stage1_results(self, expert_results, temperature_results):
        """分析阶段1结果"""
        print("\n📊 阶段1结果分析:")
        
        # 分析专家维度结果
        print("   专家维度分析:")
        for result in expert_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 分析温度参数结果
        print("   温度参数分析:")
        for result in temperature_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 保存阶段结果
        self.workflow_data["stages"]["stage1"] = {
            "expert_results": expert_results,
            "temperature_results": temperature_results
        }
        self.save_workflow_data()
    
    def analyze_stage2_results(self, expert_layer_results, gate_layer_results):
        """分析阶段2结果"""
        print("\n📊 阶段2结果分析:")
        
        # 分析专家层数结果
        print("   专家层数分析:")
        for result in expert_layer_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 分析门控层数结果
        print("   门控层数分析:")
        for result in gate_layer_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 保存阶段结果
        self.workflow_data["stages"]["stage2"] = {
            "expert_layer_results": expert_layer_results,
            "gate_layer_results": gate_layer_results
        }
        self.save_workflow_data()
    
    def analyze_stage3_results(self, expert_dropout_results, gate_dropout_results):
        """分析阶段3结果"""
        print("\n📊 阶段3结果分析:")
        
        # 分析专家Dropout结果
        print("   专家Dropout分析:")
        for result in expert_dropout_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 分析门控Dropout结果
        print("   门控Dropout分析:")
        for result in gate_dropout_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 保存阶段结果
        self.workflow_data["stages"]["stage3"] = {
            "expert_dropout_results": expert_dropout_results,
            "gate_dropout_results": gate_dropout_results
        }
        self.save_workflow_data()
    
    def analyze_stage4_results(self, balance_loss_results, sparsity_loss_results):
        """分析阶段4结果"""
        print("\n📊 阶段4结果分析:")
        
        # 分析平衡损失权重结果
        print("   平衡损失权重分析:")
        for result in balance_loss_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 分析稀疏性损失权重结果
        print("   稀疏性损失权重分析:")
        for result in sparsity_loss_results:
            if result.get('mAP'):
                print(f"     {result.get('experiment_name', 'N/A')}: mAP={result.get('mAP')}%")
        
        # 保存阶段结果
        self.workflow_data["stages"]["stage4"] = {
            "balance_loss_results": balance_loss_results,
            "sparsity_loss_results": sparsity_loss_results
        }
        self.save_workflow_data()
    
    def save_workflow_data(self):
        """保存工作流程数据"""
        with open(self.tuning_log, 'w') as f:
            json.dump(self.workflow_data, f, indent=2)
    
    def save_final_results(self):
        """保存最终结果"""
        final_results = {
            "tuning_completed_time": datetime.now().isoformat(),
            "best_config": self.best_config,
            "best_performance": self.best_performance,
            "workflow_data": self.workflow_data
        }
        
        with open("moe_tuning_final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📝 最终结果已保存: moe_tuning_final_results.json")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE调参工作流程指导")
    parser.add_argument("--base_config", default="configs/RGBNT201/MambaPro_moe.yml", help="基础配置文件")
    
    args = parser.parse_args()
    
    # 创建调参工作流程
    workflow = MoETuningWorkflow(args.base_config)
    
    # 开始调参工作流程
    workflow.start_tuning_workflow()

if __name__ == "__main__":
    main()
