#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成所有 query_id 的热力图可视化
"""
import os
import sys
import subprocess
from pathlib import Path

def get_all_query_ids(dataset_root):
    """获取数据集中所有的 query_id"""
    rgb_dir = os.path.join(dataset_root, 'test', 'RGB')
    if not os.path.exists(rgb_dir):
        print(f"❌ RGB 目录不存在: {rgb_dir}")
        return []
    
    query_ids = set()
    for filename in os.listdir(rgb_dir):
        if filename.endswith('.jpg') and '_cam' in filename:
            query_id = filename.split('_')[0]
            if query_id.isdigit():
                query_ids.add(query_id)
    
    return sorted(list(query_ids))

def main():
    # 配置参数
    weight_path = "/home/zhanghaoyang/Desktop/yzy/79.4mAP_1212_1144_run_20251212_112223/MambaProbest.pth"
    config_file = "configs/RGBNT201/yzy_best_Mambapro_moe.yml"
    dataset_root = "/home/zhanghaoyang/Desktop/yzy/MambaPro/data/datasets/RGBNT201"
    method = "eigencam"
    alpha = 0.5
    
    # 获取所有 query_id
    print("🔍 正在查找所有 query_id...")
    query_ids = get_all_query_ids(dataset_root)
    print(f"✅ 找到 {len(query_ids)} 个 query_id")
    
    if not query_ids:
        print("❌ 未找到任何 query_id")
        return
    
    # 切换到 AboutReid 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aboutreid_dir = os.path.dirname(script_dir)
    os.chdir(aboutreid_dir)
    
    # 批量生成
    success_count = 0
    fail_count = 0
    failed_queries = []
    
    print(f"\n{'='*60}")
    print(f"开始批量生成热力图 (共 {len(query_ids)} 个)")
    print(f"{'='*60}\n")
    
    for idx, query_id in enumerate(query_ids, 1):
        print(f"\n[{idx}/{len(query_ids)}] 正在生成 query_id: {query_id}")
        
        try:
            # 构建命令
            cmd = [
                sys.executable,
                "visualize_Cam/generate_heatmap_visualization.py",
                "--weight_path", weight_path,
                "--config_file", config_file,
                "--query_id", query_id,
                "--dataset_root", dataset_root,
                "--method", method,
                "--alpha", str(alpha)
            ]
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print(f"  ✅ 成功生成: {query_id}")
                success_count += 1
            else:
                print(f"  ❌ 生成失败: {query_id}")
                print(f"  错误信息: {result.stderr[:200]}")
                fail_count += 1
                failed_queries.append(query_id)
        
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  超时: {query_id}")
            fail_count += 1
            failed_queries.append(query_id)
        except Exception as e:
            print(f"  ❌ 异常: {query_id} - {e}")
            fail_count += 1
            failed_queries.append(query_id)
    
    # 输出总结
    print(f"\n{'='*60}")
    print(f"批量生成完成！")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{len(query_ids)}")
    print(f"❌ 失败: {fail_count}/{len(query_ids)}")
    
    if failed_queries:
        print(f"\n失败的 query_id: {', '.join(failed_queries)}")
    
    # 输出文件夹路径
    weight_dir = os.path.dirname(os.path.abspath(weight_path))
    weight_name = os.path.basename(weight_dir)
    output_dir = os.path.join(aboutreid_dir, 'outputs', 'EigenCAM', weight_name)
    print(f"\n📁 所有热力图保存在: {output_dir}")

if __name__ == '__main__':
    main()




