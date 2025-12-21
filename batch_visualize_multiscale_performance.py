#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理 multiscale 文件夹下的所有训练日志，生成性能与专家权重演化图表
"""
import os
import subprocess
import sys
from pathlib import Path


def find_multiscale_logs(multiscale_dir):
    """
    查找 multiscale 文件夹下所有的训练日志文件
    
    Args:
        multiscale_dir: multiscale 文件夹路径
        
    Returns:
        list: 日志文件信息列表，每个元素包含 'log_path' 和 'model_name'
    """
    log_files = []
    
    if not os.path.exists(multiscale_dir):
        print(f"⚠️  multiscale 目录不存在: {multiscale_dir}")
        return log_files
    
    # 查找所有 train_*.log 文件
    for root, dirs, files in os.walk(multiscale_dir):
        for file in files:
            if file.startswith('train_') and file.endswith('.log'):
                log_path = os.path.join(root, file)
                
                # 从路径中提取模型名称
                # 例如: .../multiscale/73.58_8x8+16x16_20251217_175523/logs/train_*.log
                # 模型名称: 73.58_8x8+16x16_20251217_175523
                path_parts = Path(log_path).parts
                multiscale_idx = None
                for i, part in enumerate(path_parts):
                    if part == 'multiscale':
                        multiscale_idx = i
                        break
                
                if multiscale_idx is not None and multiscale_idx + 1 < len(path_parts):
                    model_name = path_parts[multiscale_idx + 1]
                    
                    # 如果日志在子文件夹中（如 77.76_4x4+16x16_20251217_160700/78.3mAP_1212_1331_run_20251212_130937/logs/）
                    # 则使用子文件夹名称作为模型名称
                    if multiscale_idx + 2 < len(path_parts):
                        # 检查是否是子文件夹（包含 mAP 的文件夹名）
                        subfolder = path_parts[multiscale_idx + 2]
                        if 'mAP' in subfolder or 'run_' in subfolder:
                            model_name = subfolder
                    
                    log_files.append({
                        'log_path': log_path,
                        'model_name': model_name
                    })
    
    return sorted(log_files, key=lambda x: x['model_name'])


def generate_performance_plot(log_info, output_dir):
    """
    为单个日志文件生成性能与专家权重演化图表
    
    Args:
        log_info: 包含 'log_path' 和 'model_name' 的字典
        output_dir: 输出目录
        
    Returns:
        bool: 是否成功生成
    """
    log_path = log_info['log_path']
    model_name = log_info['model_name']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    output_filename = f"{model_name}_performance_expert_weights.png"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\n{'='*60}")
    print(f"处理模型: {model_name}")
    print(f"日志文件: {log_path}")
    print(f"输出文件: {output_path}")
    print(f"{'='*60}")
    
    # 构建命令
    script_path = os.path.join(os.path.dirname(__file__), 'visualize_performance_expert_weights.py')
    
    cmd = [
        sys.executable,
        script_path,
        '--log_path', log_path,
        '--title_suffix', f" - {model_name}",
        '--output_path', output_path
    ]
    
    try:
        # 运行可视化脚本
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✅ 成功生成: {output_filename}")
        if result.stdout:
            print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成失败: {model_name}")
        print(f"错误信息: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ 发生异常: {model_name}")
        print(f"异常信息: {str(e)}")
        return False


def main():
    """主函数"""
    # 配置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    multiscale_dir = os.path.join(base_dir, 'outputs', 'multiscale')
    output_dir = os.path.join(base_dir, 'outputs', 'performance_analysis')
    
    print(f"🔍 搜索 multiscale 日志文件...")
    print(f"搜索目录: {multiscale_dir}")
    
    # 查找所有日志文件
    log_files = find_multiscale_logs(multiscale_dir)
    
    if not log_files:
        print(f"⚠️  未找到任何日志文件")
        return
    
    print(f"\n📊 找到 {len(log_files)} 个日志文件:")
    for i, log_info in enumerate(log_files, 1):
        print(f"  {i}. {log_info['model_name']}")
        print(f"     {log_info['log_path']}")
    
    # 批量生成图表
    print(f"\n🚀 开始批量生成性能与专家权重演化图表...")
    print(f"输出目录: {output_dir}\n")
    
    success_count = 0
    fail_count = 0
    
    for log_info in log_files:
        if generate_performance_plot(log_info, output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    print(f"\n{'='*60}")
    print(f"🎉 批量处理完成！")
    print(f"✅ 成功: {success_count} 个")
    print(f"❌ 失败: {fail_count} 个")
    print(f"📁 输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    # 列出生成的文件
    if success_count > 0:
        print("📋 生成的文件列表:")
        for log_info in log_files:
            output_filename = f"{log_info['model_name']}_performance_expert_weights.png"
            output_path = os.path.join(output_dir, output_filename)
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024  # KB
                print(f"  ✅ {output_filename} ({file_size:.1f} KB)")


if __name__ == '__main__':
    main()


