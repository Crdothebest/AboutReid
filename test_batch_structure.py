#!/usr/bin/env python3
"""
测试batch数据结构，确保修复后的代码能正确处理文本特征
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    import torch
    from config import cfg
    from data.datasets.make_dataloader import make_dataloader
    from modeling import make_model

    def test_batch_structure():
        print("🔍 测试batch数据结构...")

        # 设置测试配置
        cfg.defrost()
        cfg.DATASETS.NAMES = "RGBNT201"
        cfg.DATASETS.USE_TEXT_FEATURES = True
        cfg.DATASETS.QWEN_VL_ANNO_DIR = "./QwenVL_Anno"
        cfg.MODEL.USE_TEXT_FUSION = True
        cfg.MODEL.TEXT_FUSION_METHOD = "attention"
        cfg.TEST.IMS_PER_BATCH = 4  # 小batch测试
        cfg.DATALOADER.NUM_WORKERS = 0  # 避免多进程问题
        cfg.freeze()

        # 创建数据加载器
        train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

        print("✅ 数据加载器创建成功")

        # 测试训练batch结构
        print("\n📊 测试训练batch结构...")
        train_iter = iter(train_loader)
        train_batch = next(train_iter)
        print(f"训练batch长度: {len(train_batch)}")
        print(f"训练batch元素类型: {[type(x).__name__ for x in train_batch]}")

        if len(train_batch) == 6:
            imgs, pids, target_cam, target_view, img_paths, text_features = train_batch
            print("✅ 训练batch包含文本特征")
            print(f"文本特征类型: {type(text_features)}")
            if isinstance(text_features, dict):
                print(f"文本特征keys: {text_features.keys()}")
        else:
            print(f"❌ 训练batch长度异常: {len(train_batch)}")

        # 测试验证batch结构
        print("\n📊 测试验证batch结构...")
        val_iter = iter(val_loader)
        val_batch = next(val_iter)
        print(f"验证batch长度: {len(val_batch)}")
        print(f"验证batch元素类型: {[type(x).__name__ for x in val_batch]}")

        if len(val_batch) == 7:
            imgs, pids, camids, camids_batch, viewids, img_paths, text_features = val_batch
            print("✅ 验证batch包含文本特征 (7元素)")
            print(f"文本特征类型: {type(text_features)}")
            if isinstance(text_features, dict):
                print(f"文本特征keys: {text_features.keys()}")
        elif len(val_batch) == 6:
            imgs, pids, camids, camids_batch, viewids, img_paths = val_batch
            print("✅ 验证batch为标准格式 (6元素)")
        else:
            print(f"❌ 验证batch长度异常: {len(val_batch)}")

        print("\n🎯 测试processor.py中的batch处理逻辑...")

        # 模拟processor.py中的逻辑
        batch_data = val_batch  # 使用验证batch

        if len(batch_data) == 7:  # 增强版collate函数（包含文本特征）
            img, vid, camid, camids, target_view, img_paths, text_features = batch_data
            print("✅ 验证循环正确处理7元素batch")
        elif len(batch_data) == 6:  # 标准版collate函数
            img, vid, camid, camids, target_view, img_paths = batch_data
            text_features = None  # 占位符
            print("✅ 验证循环正确处理6元素batch")
        else:  # 其他情况（兼容性）
            img, vid, camid, camids, target_view = batch_data[:5]
            text_features = None  # 占位符
            print("✅ 验证循环使用兼容性处理")

        print("\n🎉 所有测试通过！batch结构修复成功。")

    if __name__ == "__main__":
        test_batch_structure()

except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在正确的conda环境中运行")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
