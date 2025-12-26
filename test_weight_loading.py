#!/usr/bin/env python3
"""
测试权重加载打印功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 创建一个简化的配置对象
class SimpleConfig:
    def __init__(self):
        self.MODEL = type('Model', (), {})()
        self.MODEL.TRANSFORMER_TYPE = 'ViT-B-16'
        self.MODEL.PRETRAIN_PATH_T = '/home/zhanghaoyang/Desktop/yzy/AboutReid/pths/ViT-B-16.pt'
        self.INPUT = type('Input', (), {})()
        self.INPUT.SIZE_TRAIN = [224, 224]
        self.MODEL.STRIDE_SIZE = [16, 16]
        self.MODEL.SIE_CAMERA = True
        self.MODEL.SIE_COE = 1.0
        self.MODEL.DROP_PATH = 0.1
        self.MODEL.DROP_OUT = 0.0
        self.MODEL.ATT_DROP_RATE = 0.0
        self.MODEL.USE_CLIP_MULTI_SCALE = False
        self.MODEL.USE_MULTI_SCALE_MOE = False
        self.MODEL.USE_GATE_FUSION = False
        self.MODEL.USE_ATTENTION_FUSION = False
        self.DATASETS = type('Datasets', (), {})()
        self.DATASETS.NAMES = 'test'
        self.TEST = type('Test', (), {})()
        self.TEST.FEAT_NORM = True
        self.TEST.NECK_FEAT = 'after'
        self.MODEL.NECK = 'bnneck'
        self.MODEL.ID_LOSS_TYPE = 'softmax'
        self.MODEL.DIRECT = True
        self.MODEL.MAMBA = True
        self.MODEL.USE_TEXT_FUSION = False
        self.MODEL.USE_MODAL_GUIDANCE = False
        self.MODEL.FLOPS_TEST = False
        self.SOLVER = type('Solver', (), {})()
        self.SOLVER.SEED = 42

def test_weight_loading():
    from modeling.make_model import make_model

    print("🔍 测试预训练权重加载打印功能...")
    print("=" * 60)

    cfg = SimpleConfig()

    try:
        # 创建模型（这会触发预训练权重加载）
        model = make_model(cfg, num_class=100, camera_num=5, view_num=0)
        print("✅ 模型创建成功！")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

    return True

if __name__ == "__main__":
    test_weight_loading()