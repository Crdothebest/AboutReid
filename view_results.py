#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看实验结果Excel文件
"""

import pandas as pd
import os

def view_excel_results():
    """查看Excel实验结果"""
    excel_file = "experiment_results.xlsx"
    
    if not os.path.exists(excel_file):
        print("❌ 实验结果文件不存在，请先运行实验")
        return
    
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        
        print("📊 实验结果汇总:")
        print("=" * 80)
        
        # 显示所有结果
        for index, row in df.iterrows():
            print(f"\n实验 {index + 1}:")
            print(f"  时间: {row['实验时间']}")
            print(f"  数据集: {row['数据集']}")
            print(f"  滑动窗口尺度: {row['滑动窗口尺度']}")
            print(f"  拼接方式: {row['拼接方式']}")
            print(f"  专家权重占比: {row['专家权重占比']}")
            print(f"  目录: {row['实验目录']}")
            print(f"  结果: mAP={row['mAP']:.1f}%, Rank-1={row['Rank-1']:.1f}%, Rank-5={row['Rank-5']:.1f}%, Rank-10={row['Rank-10']:.1f}%")
            print(f"  最佳: mAP={row['Best_mAP']:.1f}%, Rank-1={row['Best_Rank-1']:.1f}%, Rank-5={row['Best_Rank-5']:.1f}%, Rank-10={row['Best_Rank-10']:.1f}%")
            print("-" * 80)
        
        # 按数据集分组显示最佳结果
        if len(df) > 0:
            print(f"\n🏆 各数据集最佳结果:")
            print("=" * 80)
            
            # 按数据集分组
            for dataset in df['数据集'].unique():
                dataset_df = df[df['数据集'] == dataset]
                if len(dataset_df) > 0:
                    best_mAP_idx = dataset_df['Best_mAP'].idxmax()
                    best_rank1_idx = dataset_df['Best_Rank-1'].idxmax()
                    
                    print(f"\n📊 {dataset} 数据集:")
                    print(f"  最高mAP: {dataset_df.loc[best_mAP_idx, 'Best_mAP']:.1f}% (实验 {best_mAP_idx + 1})")
                    print(f"  最高Rank-1: {dataset_df.loc[best_rank1_idx, 'Best_Rank-1']:.1f}% (实验 {best_rank1_idx + 1})")
                    print(f"  实验次数: {len(dataset_df)}")
            
            # 全局最佳结果
            print(f"\n🌍 全局最佳结果:")
            print("=" * 80)
            best_mAP_idx = df['Best_mAP'].idxmax()
            best_rank1_idx = df['Best_Rank-1'].idxmax()
            
            print(f"  最高mAP: {df.loc[best_mAP_idx, 'Best_mAP']:.1f}% (实验 {best_mAP_idx + 1}, {df.loc[best_mAP_idx, '数据集']})")
            print(f"  最高Rank-1: {df.loc[best_rank1_idx, 'Best_Rank-1']:.1f}% (实验 {best_rank1_idx + 1}, {df.loc[best_rank1_idx, '数据集']})")
            
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")

if __name__ == "__main__":
    view_excel_results()
