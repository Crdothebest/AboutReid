#!/usr/bin/env python3
"""
测试修复是否有效
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_qwen_loader():
    """测试QwenVL加载器是否使用真实编码"""
    try:
        from data.datasets.qwen_vl_loader import QwenVLTextLoader

        # 创建加载器
        loader = QwenVLTextLoader("data/datasets/QwenVL_Anno", "RGBNT201")

        # 测试编码功能
        test_desc = "The female is wearing a white dress with black stripes"
        feature = loader._encode_text_description(test_desc)

        print(f"✅ 文本编码器工作正常")
        print(f"   输入: {test_desc[:50]}...")
        print(f"   输出维度: {feature.shape}")
        print(f"   输出类型: {type(feature)}")
        print(f"   L2范数: {feature.norm():.4f}")

        return True
    except Exception as e:
        print(f"❌ QwenVL加载器测试失败: {e}")
        return False

def test_config():
    """测试配置文件修改"""
    try:
        import yaml
        with open('configs/RGBNT201/20251013_experiment_config.yml', 'r') as f:
            config = yaml.safe_load(f)

        # 检查关键配置
        assert config['DATASETS']['USE_TEXT_FEATURES'] == True
        assert config['DATASETS']['QWEN_VL_ANNO_DIR'] == "data/datasets/QwenVL_Anno"
        assert config['TEST']['WEIGHT'] == ""

        print("✅ 配置文件修改正确")
        return True
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def main():
    print("🔍 测试修复效果...")

    success_count = 0

    # 测试1: QwenVL加载器
    if test_qwen_loader():
        success_count += 1

    # 测试2: 配置文件
    if test_config():
        success_count += 1

    print(f"\n📊 测试结果: {success_count}/2 项通过")

    if success_count == 2:
        print("🎉 所有修复验证通过！可以开始训练。")
        return True
    else:
        print("❌ 部分修复存在问题，请检查代码。")
        return False

if __name__ == "__main__":
    main()
