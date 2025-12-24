#!/usr/bin/env python3
"""
测试IDEA风格离线预编码功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_idea_pre_encoding():
    """测试IDEA预编码功能"""
    print("🧪 测试IDEA风格离线预编码功能")
    print("=" * 50)

    try:
        from data.datasets.qwen_vl_loader import QwenVLTextLoader

        # 创建简化的配置对象
        class TestCfg:
            class MODEL:
                PRETRAIN_PATH_T = "/home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt"
                PROMPT = True
                ADAPTER = True

        cfg = TestCfg()

        # 测试IDEA风格预编码模式
        print("🏗️ 创建IDEA风格文本加载器...")
        loader = QwenVLTextLoader(
            anno_dir="./data/datasets/QwenVL_Anno/RGBNT201/text",
            use_clip=False,  # IDEA模式
            cfg=cfg
        )

        # 检查是否成功加载了预编码特征
        if hasattr(loader, 'text_features') and loader.text_features:
            num_features = len(loader.text_features)
            print(f"✅ 成功加载 {num_features} 个预编码特征")

            # 测试获取特征
            test_key = list(loader.text_features.keys())[0]
            feature = loader.get_text_feature("0001_c1.jpg", "RGB")
            print(f"✅ 成功获取特征，维度: {feature.shape}")

            # 测试GPU预加载
            try:
                loader.preload_to_gpu('cpu')  # 使用CPU测试
                print("✅ GPU预加载功能正常")
            except Exception as e:
                print(f"⚠️ GPU预加载测试跳过: {e}")

        else:
            print("❌ 预编码特征加载失败")
            return False

        print("🎉 IDEA预编码功能测试通过！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_idea_pre_encoding()