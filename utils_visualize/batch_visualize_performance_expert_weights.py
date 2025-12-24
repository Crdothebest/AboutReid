# Category: vis_utils (可视化工具)
# Description: 提供热力图 (CAM)、t-SNE 降维、检索结果展示等模型可视化功能

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理训练日志，为每个日志文件生成性能与专家权重演化图
"""
import os
import glob
import subprocess
import sys
from pathlib import Path


def find_log_files(search_dirs):
    """
    查找所有训练日志文件
    
    Args:
        search_dirs: 搜索目录列表
        
    Returns:
        list: 日志文件路径列表
    """
    log_files = []
    for search_dir in search_dirs:
        # 查找所有 train_*.log 文件
        pattern = os.path.join(search_dir, '**', 'logs', 'train_*.log')
        found = glob.glob(pattern, recursive=True)
        log_files.extend(found)
    
    return sorted(log_files)


def main():
    """批量处理日志文件"""
    # 搜索目录
    search_dirs = [
        '/home/zhanghaoyang/Desktop/yzy',
        '/home/zhanghaoyang/Desktop/yzy/AboutReid/outputs'
    ]
    
    print("🔍 查找训练日志文件...")
    log_files = find_log_files(search_dirs)
    
    if not log_files:
        print("❌ 未找到任何日志文件")
        return
    
    print(f"✅ 找到 {len(log_files)} 个日志文件\n")
    
    # 脚本路径
    script_path = os.path.join(os.path.dirname(__file__), 'visualize_performance_expert_weights.py')
    
    success_count = 0
    fail_count = 0
    
    for idx, log_file in enumerate(log_files, 1):
        print(f"[{idx}/{len(log_files)}] 处理: {os.path.basename(os.path.dirname(log_file))}")
        
        # 从路径中提取模型名称
        log_dir = os.path.dirname(log_file)
        parent_dir = os.path.basename(os.path.dirname(log_dir))
        title_suffix = f" - {parent_dir}"
        
        try:
            # 运行可视化脚本
            cmd = [
                sys.executable,
                script_path,
                '--log_path', log_file,
                '--title_suffix', title_suffix
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"  ✅ 成功生成")
                success_count += 1
            else:
                print(f"  ❌ 生成失败: {result.stderr[:100]}")
                fail_count += 1
        
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"批量处理完成！")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{len(log_files)}")
    print(f"❌ 失败: {fail_count}/{len(log_files)}")


if __name__ == '__main__':
    main()



