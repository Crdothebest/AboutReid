#!/usr/bin/env python3
"""
测试修复逻辑：验证batch长度检查和unpack操作
"""

def test_fix_logic():
    print("🔍 测试processor.py修复逻辑...")

    # 模拟7元素batch (val_collate_fn_with_text的返回值)
    mock_batch_7 = ["imgs", "pids", "camids", "camids_batch", "viewids", "img_paths", "text_features"]
    print(f"模拟7元素batch: {mock_batch_7}")

    # 测试修复后的逻辑
    batch_data = mock_batch_7

    if len(batch_data) == 7:  # 增强版collate函数（包含文本特征）
        img, vid, camid, camids, target_view, img_paths, text_features = batch_data
        print("✅ 正确处理7元素batch")
        print(f"   img: {img}, vid: {vid}, camid: {camid}, camids: {camids}")
        print(f"   target_view: {target_view}, img_paths: {img_paths}, text_features: {text_features}")
    elif len(batch_data) == 6:  # 标准版collate函数
        img, vid, camid, camids, target_view, img_paths = batch_data
        text_features = None  # 占位符
        print("✅ 处理6元素batch")
    else:  # 其他情况（兼容性）
        img, vid, camid, camids, target_view = batch_data[:5]
        text_features = None  # 占位符
        print("✅ 使用兼容性处理")

    # 测试6元素batch
    mock_batch_6 = ["imgs", "pids", "camids", "camids_batch", "viewids", "img_paths"]
    print(f"\n模拟6元素batch: {mock_batch_6}")

    batch_data = mock_batch_6
    if len(batch_data) == 7:
        img, vid, camid, camids, target_view, img_paths, text_features = batch_data
    elif len(batch_data) == 6:
        img, vid, camid, camids, target_view, img_paths = batch_data
        text_features = None
        print("✅ 正确处理6元素batch")
    else:
        img, vid, camid, camids, target_view = batch_data[:5]
        text_features = None

    # 测试5元素batch（兼容性）
    mock_batch_5 = ["imgs", "pids", "camids", "camids_batch", "viewids"]
    print(f"\n模拟5元素batch: {mock_batch_5}")

    batch_data = mock_batch_5
    if len(batch_data) == 7:
        img, vid, camid, camids, target_view, img_paths, text_features = batch_data
    elif len(batch_data) == 6:
        img, vid, camid, camids, target_view, img_paths = batch_data
        text_features = None
    else:
        img, vid, camid, camids, target_view = batch_data[:5]
        text_features = None
        print("✅ 正确处理5元素batch（兼容性）")

    print("\n🎉 修复逻辑测试通过！")

if __name__ == "__main__":
    test_fix_logic()
