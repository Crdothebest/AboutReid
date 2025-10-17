#!/usr/bin/env python3
"""
ReID模型Top-K Ranked List可视化工具使用示例

该脚本展示了如何使用visualize_ranked_list.py生成ReID模型的Top-K检索可视化结果。

使用方法：
1. 确保数据集路径正确
2. 确保模型权重文件存在
3. 运行脚本生成可视化结果

作者：MambaPro团队
日期：2024
"""

import os
import subprocess
import sys

def run_visualization_example():
    """
    运行可视化示例
    """
    print("🚀 ReID模型Top-K Ranked List可视化工具使用示例")
    print("=" * 60)
    
    # 检查必要的文件是否存在
    script_path = "visualize_ranked_list.py"
    if not os.path.exists(script_path):
        print(f"❌ 脚本文件不存在: {script_path}")
        return
    
    # 示例1：RGB模态可视化
    print("\n📸 示例1: RGB模态Top-9检索可视化")
    print("-" * 40)
    
    cmd_rgb = [
        "python", script_path,
        "--dataset_root", "data/RGBNT201",
        "--config_path", "configs/RGBNT201/MambaPro_moe.yml", 
        "--model_path", "pths/MambaProbest.pth",
        "--modality", "RGB",
        "--top_k", "9",
        "--num_queries", "5",
        "--output_dir", "ranked_list_results_rgb"
    ]
    
    print(f"执行命令: {' '.join(cmd_rgb)}")
    try:
        result = subprocess.run(cmd_rgb, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ RGB模态可视化完成")
            print("输出:", result.stdout[-500:])  # 显示最后500个字符
        else:
            print("❌ RGB模态可视化失败")
            print("错误:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
    except Exception as e:
        print(f"❌ 执行错误: {e}")
    
    # 示例2：NI模态可视化
    print("\n📸 示例2: NI模态Top-5检索可视化")
    print("-" * 40)
    
    cmd_ni = [
        "python", script_path,
        "--dataset_root", "data/RGBNT201",
        "--config_path", "configs/RGBNT201/MambaPro_moe.yml",
        "--model_path", "pths/MambaProbest.pth", 
        "--modality", "NI",
        "--top_k", "5",
        "--num_queries", "3",
        "--output_dir", "ranked_list_results_ni"
    ]
    
    print(f"执行命令: {' '.join(cmd_ni)}")
    try:
        result = subprocess.run(cmd_ni, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ NI模态可视化完成")
            print("输出:", result.stdout[-500:])
        else:
            print("❌ NI模态可视化失败")
            print("错误:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
    except Exception as e:
        print(f"❌ 执行错误: {e}")
    
    # 示例3：TI模态可视化
    print("\n📸 示例3: TI模态Top-10检索可视化")
    print("-" * 40)
    
    cmd_ti = [
        "python", script_path,
        "--dataset_root", "data/RGBNT201",
        "--config_path", "configs/RGBNT201/MambaPro_moe.yml",
        "--model_path", "pths/MambaProbest.pth",
        "--modality", "TI", 
        "--top_k", "10",
        "--num_queries", "3",
        "--output_dir", "ranked_list_results_ti"
    ]
    
    print(f"执行命令: {' '.join(cmd_ti)}")
    try:
        result = subprocess.run(cmd_ti, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ TI模态可视化完成")
            print("输出:", result.stdout[-500:])
        else:
            print("❌ TI模态可视化失败")
            print("错误:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
    except Exception as e:
        print(f"❌ 执行错误: {e}")
    
    print("\n🎉 所有示例执行完成！")
    print("\n📁 生成的结果文件:")
    print("   - ranked_list_results_rgb/: RGB模态可视化结果")
    print("   - ranked_list_results_ni/: NI模态可视化结果") 
    print("   - ranked_list_results_ti/: TI模态可视化结果")
    print("\n📊 每个结果目录包含:")
    print("   - ranked_list_XXXXXX_模态_topK.png: 可视化图像")
    print("   - summary_report.txt: 汇总报告")

def show_usage_help():
    """
    显示使用帮助
    """
    print("\n📖 使用帮助")
    print("=" * 30)
    print("直接运行脚本:")
    print("python visualize_ranked_list.py --help")
    print("\n常用参数:")
    print("  --dataset_root: 数据集根目录 (默认: data/RGBNT201)")
    print("  --config_path: 配置文件路径 (默认: configs/RGBNT201/MambaPro_moe.yml)")
    print("  --model_path: 模型权重路径 (默认: pths/MambaProbest.pth)")
    print("  --modality: 模态类型 (RGB/NI/TI, 默认: RGB)")
    print("  --top_k: Top-K检索的K值 (默认: 9)")
    print("  --num_queries: 要可视化的Query数量 (默认: 10)")
    print("  --output_dir: 输出目录 (默认: ranked_list_results)")
    print("\n示例命令:")
    print("python visualize_ranked_list.py --modality RGB --top_k 9 --num_queries 5")
    print("python visualize_ranked_list.py --modality NI --top_k 5 --num_queries 3")
    print("python visualize_ranked_list.py --modality TI --top_k 10 --num_queries 3")

if __name__ == '__main__':
    print("🔧 ReID模型Top-K Ranked List可视化工具")
    print("选择操作:")
    print("1. 运行示例")
    print("2. 显示使用帮助")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "1":
        run_visualization_example()
    elif choice == "2":
        show_usage_help()
    else:
        print("❌ 无效选择")
        sys.exit(1)
