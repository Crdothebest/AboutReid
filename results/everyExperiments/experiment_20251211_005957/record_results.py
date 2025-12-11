#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动记录训练结果到Excel
"""

import pandas as pd
import re
import sys
import os
from datetime import datetime

def parse_training_log(log_file):
    """解析训练日志，提取最佳结果"""
    results = {
        # 只保留Best相关记录
        'Best_mAP': 0.0,
        'Best_Rank-1': 0.0,
        'Best_Rank-5': 0.0,
        'Best_Rank-10': 0.0,
        '滑动窗口尺度': '',
        '拼接方式': ''
    }
    
    try:
        # 🔥 修复编码问题：尝试多种编码方式
        content = None
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'gbk']
        
        for encoding in encodings:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"✅ 成功使用 {encoding} 编码读取日志文件")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"❌ 无法读取日志文件，尝试了所有编码方式")
            return results
            
        # 🔥 调试：显示日志文件内容片段
        print(f"🔍 调试：日志文件内容长度: {len(content)}")
        print(f"🔍 调试：日志文件最后500字符:")
        print(content[-500:])
        
        # 获取所有匹配，取最后一个（最终结果）
        all_mAP_matches = re.findall(r'Best mAP: ([\d.]+)%', content)
        print(f"🔍 调试：找到 {len(all_mAP_matches)} 个Best mAP匹配: {all_mAP_matches}")
        if all_mAP_matches:
            results['Best_mAP'] = float(all_mAP_matches[-1])  # 取最后一个匹配
            print(f"🔍 调试：设置Best_mAP为: {results['Best_mAP']}")
            
        all_rank1_matches = re.findall(r'Best Rank-1: ([\d.]+)%', content)
        print(f"🔍 调试：找到 {len(all_rank1_matches)} 个Best Rank-1匹配: {all_rank1_matches}")
        if all_rank1_matches:
            results['Best_Rank-1'] = float(all_rank1_matches[-1])  # 取最后一个匹配
            print(f"🔍 调试：设置Best_Rank-1为: {results['Best_Rank-1']}")
            
        all_rank5_matches = re.findall(r'Best Rank-5: ([\d.]+)%', content)
        print(f"🔍 调试：找到 {len(all_rank5_matches)} 个Best Rank-5匹配: {all_rank5_matches}")
        if all_rank5_matches:
            results['Best_Rank-5'] = float(all_rank5_matches[-1])  # 取最后一个匹配
            print(f"🔍 调试：设置Best_Rank-5为: {results['Best_Rank-5']}")
            
        all_rank10_matches = re.findall(r'Best Rank-10: ([\d.]+)%', content)
        print(f"🔍 调试：找到 {len(all_rank10_matches)} 个Best Rank-10匹配: {all_rank10_matches}")
        if all_rank10_matches:
            results['Best_Rank-10'] = float(all_rank10_matches[-1])  # 取最后一个匹配
            print(f"🔍 调试：设置Best_Rank-10为: {results['Best_Rank-10']}")
            
        # 提取滑动窗口尺度信息
        window_scale_match = re.search(r'滑动窗口尺度: \[([\d, ]+)\]', content)
        if window_scale_match:
            results['滑动窗口尺度'] = window_scale_match.group(1).strip()
        else:
            # 从命令行参数中提取
            if 'CLIP_MULTI_SCALE_SCALES' in content:
                scale_match = re.search(r'CLIP_MULTI_SCALE_SCALES \[([\d, ]+)\]', content)
                if scale_match:
                    results['滑动窗口尺度'] = scale_match.group(1).strip()
            
        # 🔥 修复：提取拼接方式信息（根据新的输出格式）
        # 检查日志中的拼接方式输出
        if '拼接融合：使用门控加权-预处理' in content:
            results['拼接方式'] = '门控加权-预处理'
        elif '拼接融合：使用注意力-预处理' in content:
            results['拼接方式'] = '注意力-预处理'
        elif '拼接融合：使用无预处理' in content:
            results['拼接方式'] = '无预处理'
        else:
            # 如果没有找到明确的拼接方式，尝试从其他输出推断
            if '门控加权-预处理机制：已启用' in content:
                results['拼接方式'] = '门控加权-预处理'
            elif '注意力-预处理机制：已启用' in content:
                results['拼接方式'] = '注意力-预处理'
            else:
                results['拼接方式'] = '无预处理'  # 默认
            
        # 移除专家权重信息提取
        
        # 移除专家权重处理逻辑
            
    except Exception as e:
        print(f"解析日志文件时出错: {e}")
        
    return results

def extract_dataset_info(command_line):
    """从命令行中提取数据集信息"""
    dataset = "Unknown"
    
    # print(f"🔍 调试：开始提取数据集信息")
    # print(f"🔍 调试：命令行: {command_line}")
    
    # 从命令行中提取数据集信息（优先从配置文件路径）
    if "configs/RGBNT100/" in command_line:
        dataset = "RGBNT100"
        # print(f"🔍 调试：从命令行路径中提取到 RGBNT100")
    elif "configs/RGBNT201/" in command_line:
        dataset = "RGBNT201"
        # print(f"🔍 调试：从命令行路径中提取到 RGBNT201")
    elif "configs/MSVR310/" in command_line:
        dataset = "MSVR310"
        # print(f"🔍 调试：从命令行路径中提取到 MSVR310")
    elif "RGBNT100" in command_line:
        dataset = "RGBNT100"
        # print(f"🔍 调试：从命令行中提取到 RGBNT100")
    elif "RGBNT201" in command_line:
        dataset = "RGBNT201"
        # print(f"🔍 调试：从命令行中提取到 RGBNT201")
    elif "MSVR310" in command_line:
        dataset = "MSVR310"
        # print(f"🔍 调试：从命令行中提取到 MSVR310")
    elif "Market1501" in command_line:
        dataset = "Market1501"
        # print(f"🔍 调试：从命令行中提取到 Market1501")
    elif "DukeMTMC" in command_line:
        dataset = "DukeMTMC"
        # print(f"🔍 调试：从命令行中提取到 DukeMTMC")
    elif "MSMT17" in command_line:
        dataset = "MSMT17"
        # print(f"🔍 调试：从命令行中提取到 MSMT17")
    
    # 如果从命令行中无法提取，尝试从配置文件路径中提取
    if dataset == "Unknown":
        # print(f"🔍 调试：从命令行中无法提取，尝试从配置文件路径中提取")
        # 查找配置文件路径
        import re
        config_match = re.search(r'--config_file\s+([^\s]+)', command_line)
        if config_match:
            config_path = config_match.group(1)
            # print(f"🔍 调试：找到配置文件路径: {config_path}")
            # 检查是否是实验目录下的配置文件
            if "experiment_" in config_path and "configs/experiment_config.yml" in config_path:
                # print(f"🔍 调试：这是实验目录下的配置文件")
                # 这是实验目录下的配置文件，需要从原始配置文件路径中提取
                # 从实验信息文件中读取原始配置文件路径
                experiment_dir = config_path.replace("/configs/experiment_config.yml", "")
                info_file = f"{experiment_dir}/experiment_info.txt"
                # print(f"🔍 调试：实验信息文件路径: {info_file}")
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info_content = f.read()
                    # print(f"🔍 调试：实验信息文件内容: {info_content}")
                    # 从实验信息中提取原始配置文件路径
                    original_config_match = re.search(r'原始配置文件: ([^\s]+)', info_content)
                    if original_config_match:
                        original_config_path = original_config_match.group(1)
                        # print(f"🔍 调试：提取到原始配置文件路径: {original_config_path}")
                        if "RGBNT100" in original_config_path:
                            dataset = "RGBNT100"
                            # print(f"🔍 调试：从原始配置文件路径中提取到 RGBNT100")
                        elif "RGBNT201" in original_config_path:
                            dataset = "RGBNT201"
                            # print(f"🔍 调试：从原始配置文件路径中提取到 RGBNT201")
                        elif "MSVR310" in original_config_path:
                            dataset = "MSVR310"
                            # print(f"🔍 调试：从原始配置文件路径中提取到 MSVR310")
                    # else:
                        # print(f"🔍 调试：无法从实验信息文件中提取原始配置文件路径")
                except Exception as e:
                    # print(f"🔍 调试：读取实验信息文件时出错: {e}")
                    pass
            else:
                # print(f"🔍 调试：这不是实验目录下的配置文件，直接检查路径")
                # 直接检查配置文件路径
                if "RGBNT100" in config_path:
                    dataset = "RGBNT100"
                    # print(f"🔍 调试：从配置文件路径中提取到 RGBNT100")
                elif "RGBNT201" in config_path:
                    dataset = "RGBNT201"
                    # print(f"🔍 调试：从配置文件路径中提取到 RGBNT201")
                elif "MSVR310" in config_path:
                    dataset = "MSVR310"
                    # print(f"🔍 调试：从配置文件路径中提取到 MSVR310")
        # else:
            # print(f"🔍 调试：无法从命令行中提取配置文件路径")
    
    # print(f"🔍 调试：最终提取到的数据集: {dataset}")
    return dataset

def update_excel_results(experiment_dir, command_line, results):
    """更新Excel结果文件"""
    excel_file = "experiment_results.xlsx"
    
    # 提取数据集信息
    dataset = extract_dataset_info(command_line)
    
    # 准备新记录
    new_record = {
        '实验时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '数据集': dataset,
        '实验目录': experiment_dir,
        '命令行': command_line,
        '滑动窗口尺度': results['滑动窗口尺度'],
        '拼接方式': results['拼接方式'],
        # 只保留Best相关记录
        'Best_mAP': results['Best_mAP'],
        'Best_Rank-1': results['Best_Rank-1'],
        'Best_Rank-5': results['Best_Rank-5'],
        'Best_Rank-10': results['Best_Rank-10']
    }
    
    try:
        # 如果Excel文件存在，读取现有数据
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
        else:
            # 创建新的DataFrame
            df = pd.DataFrame(columns=[
                '实验时间', '数据集', '实验目录', '命令行', '滑动窗口尺度', '拼接方式',
                'Best_mAP', 'Best_Rank-1', 'Best_Rank-5', 'Best_Rank-10'
            ])
        
        # 添加新记录
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
        # 保存到Excel
        df.to_excel(excel_file, index=False)
        print(f"✅ 结果已记录到 {excel_file}")
        
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python record_results.py <实验目录> <命令行>")
        sys.exit(1)
        
    experiment_dir = sys.argv[1]
    command_line = sys.argv[2]
    log_file = os.path.join(experiment_dir, "logs", "train_log.txt")
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        sys.exit(1)
        
    # 解析结果
    results = parse_training_log(log_file)
    
    # 更新Excel
    update_excel_results(experiment_dir, command_line, results)
    
    # 打印结果摘要
    print(f"📊 实验结果摘要:")
    print(f"   滑动窗口尺度: {results['滑动窗口尺度']}")
    print(f"   拼接方式: {results['拼接方式']}")
    # 只输出Best相关结果
    print(f"   Best mAP: {results['Best_mAP']:.1f}%")
    print(f"   Best Rank-1: {results['Best_Rank-1']:.1f}%")
    print(f"   Best Rank-5: {results['Best_Rank-5']:.1f}%")
    print(f"   Best Rank-10: {results['Best_Rank-10']:.1f}%")
