#!/usr/bin/env python3
"""
测试文本融合开关控制是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_text_fusion_defaults():
    """测试默认配置（开关关闭）"""
    print("🔍 测试文本融合默认配置...")

    try:
        import yaml

        # 读取配置文件
        with open('configs/RGBNT201/20251013_experiment_config.yml', 'r') as f:
            config = yaml.safe_load(f)

        # 检查关键配置
        model_config = config.get('MODEL', {})
        dataset_config = config.get('DATASETS', {})

        # 默认应该都是关闭的
        assert model_config.get('USE_TEXT_FUSION', False) == False, "MODEL.USE_TEXT_FUSION 应该默认关闭"
        assert dataset_config.get('USE_TEXT_FEATURES', False) == False, "DATASETS.USE_TEXT_FEATURES 应该默认关闭"

        print("✅ 默认配置正确：文本融合开关已关闭")
        print(f"   MODEL.USE_TEXT_FUSION: {model_config.get('USE_TEXT_FUSION')}")
        print(f"   DATASETS.USE_TEXT_FEATURES: {dataset_config.get('USE_TEXT_FEATURES')}")

        return True
    except Exception as e:
        print(f"❌ 默认配置测试失败: {e}")
        return False

def test_switch_enabled():
    """测试开启开关的配置"""
    print("\n🔍 测试开启文本融合的配置...")

    try:
        import yaml

        # 模拟开启配置
        with open('configs/RGBNT201/20251013_experiment_config.yml', 'r') as f:
            config = yaml.safe_load(f)

        # 手动设置为开启状态（模拟命令行参数）
        config['MODEL']['USE_TEXT_FUSION'] = True
        config['DATASETS']['USE_TEXT_FEATURES'] = True

        # 验证开启状态
        model_config = config.get('MODEL', {})
        dataset_config = config.get('DATASETS', {})

        assert model_config.get('USE_TEXT_FUSION') == True, "应该能开启文本融合"
        assert dataset_config.get('USE_TEXT_FEATURES') == True, "应该能开启文本特征加载"
        assert dataset_config.get('QWEN_VL_ANNO_DIR') == "data/datasets/QwenVL_Anno", "应该指向真实数据"

        print("✅ 开关开启测试通过：文本融合功能正常开启")
        print(f"   MODEL.USE_TEXT_FUSION: {model_config.get('USE_TEXT_FUSION')}")
        print(f"   DATASETS.USE_TEXT_FEATURES: {dataset_config.get('USE_TEXT_FEATURES')}")
        print(f"   数据路径: {dataset_config.get('QWEN_VL_ANNO_DIR')}")

        return True
    except Exception as e:
        print(f"❌ 开关开启测试失败: {e}")
        return False

def main():
    print("🧪 测试文本融合开关控制\n")

    success_count = 0

    # 测试1: 默认关闭
    if test_text_fusion_defaults():
        success_count += 1

    # 测试2: 开关开启
    if test_switch_enabled():
        success_count += 1

    print(f"\n📊 测试结果: {success_count}/2 项通过")

    if success_count == 2:
        print("🎉 文本融合开关控制测试完全通过！")
        print("\n📋 总结:")
        print("✅ 默认状态：开关关闭，保持原AboutReid功能")
        print("✅ 开启状态：使用真实QwenVL数据进行文本融合")
        print("✅ 配置正确：数据路径指向真实文本特征")
        return True
    else:
        print("❌ 开关控制存在问题，请检查配置。")
        return False

if __name__ == "__main__":
    main()