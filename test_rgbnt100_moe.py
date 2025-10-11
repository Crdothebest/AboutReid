#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RGBNT100数据集上的滑动窗口+MoE模块
"""

import torch
import torch.nn as nn
from config import get_cfg_defaults
from modeling import make_model
from data.datasets import init_dataset
from data.datasets import make_dataloader

def test_rgbnt100_moe():
    """测试RGBNT100数据集上的滑动窗口+MoE模块"""
    print("🔥 开始测试RGBNT100数据集上的滑动窗口+MoE模块")
    
    # 1. 加载配置
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/RGBNT100/MambaPro_moe.yml")
    print(f"✅ 配置文件加载成功: {cfg.DATASETS.NAMES}")
    
    # 2. 初始化数据集
    print("📊 初始化RGBNT100数据集...")
    dataset = init_dataset(cfg.DATASETS.NAMES[0], root=cfg.DATASETS.ROOT_DIR)
    print(f"✅ 数据集统计:")
    print(f"   - 训练集: {dataset.num_train_pids}个ID, {dataset.num_train_imgs}张图像")
    print(f"   - 查询集: {dataset.num_query_pids}个ID, {dataset.num_query_imgs}张图像")
    print(f"   - 画廊集: {dataset.num_gallery_pids}个ID, {dataset.num_gallery_imgs}张图像")
    
    # 3. 创建数据加载器
    print("🔄 创建数据加载器...")
    train_loader, val_loader, num_classes, num_cams, num_views = make_dataloader(cfg)
    print(f"✅ 数据加载器创建成功:")
    print(f"   - 类别数: {num_classes}")
    print(f"   - 摄像头数: {num_cams}")
    print(f"   - 视角数: {num_views}")
    
    # 4. 创建模型
    print("🏗️ 创建模型...")
    model = make_model(num_classes, cfg, num_cams, num_views)
    print(f"✅ 模型创建成功")
    
    # 5. 测试模型前向传播
    print("🧪 测试模型前向传播...")
    model.eval()
    
    # 获取一个批次的数据
    for batch_idx, (imgs, pids, camids, viewids, img_paths) in enumerate(train_loader):
        print(f"📦 批次 {batch_idx + 1}:")
        print(f"   - 图像形状: {imgs['RGB'].shape}")
        print(f"   - 标签形状: {pids.shape}")
        print(f"   - 摄像头ID形状: {camids.shape}")
        
        # 测试前向传播
        with torch.no_grad():
            try:
                outputs = model(imgs, pids, camids, viewids)
                print(f"✅ 前向传播成功!")
                print(f"   - 输出类型: {type(outputs)}")
                if isinstance(outputs, tuple):
                    print(f"   - 输出数量: {len(outputs)}")
                    for i, output in enumerate(outputs):
                        if hasattr(output, 'shape'):
                            print(f"   - 输出{i}形状: {output.shape}")
                elif hasattr(outputs, 'shape'):
                    print(f"   - 输出形状: {outputs.shape}")
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # 只测试第一个批次
        break
    
    print("🎉 RGBNT100数据集上的滑动窗口+MoE模块测试完成!")
    return True

if __name__ == "__main__":
    test_rgbnt100_moe()
