# Category: vis_utils (可视化工具)
# Description: 提供热力图 (CAM)、t-SNE 降维、检索结果展示等模型可视化功能

"""
人员重识别模型性能对比可视化工具

该脚本用于对比分析两个ReID模型在不同模态下的检索性能，
特别关注旧模型表现优于新模型的案例，用于模型改进分析。

主要功能：
1. 加载两个训练好的ReID模型（旧模型 vs 新模型）
2. 在RGB、NI、TI三种模态下进行特征提取和相似度计算
3. 识别旧模型表现更好的图像样本
4. 输出详细的性能对比统计信息

作者：MambaPro团队
日期：2024
"""

import os
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from modeling.make_model import make_model
from config import cfg

def build_transforms(is_train=False):
    """
    构建图像预处理变换管道
    
    Args:
        is_train (bool): 是否为训练模式，影响数据增强策略
        
    Returns:
        transforms.Compose: 图像变换管道
    """
    # ImageNet预训练模型的标准化参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if is_train:
        # 训练模式：包含数据增强
        transform = transforms.Compose([
            transforms.Resize((256, 128)),        # 调整图像尺寸为ReID标准尺寸
            transforms.RandomHorizontalFlip(),    # 随机水平翻转增强
            transforms.ToTensor(),               # 转换为张量
            normalize,                          # 标准化
        ])
    else:
        # 测试模式：仅基础预处理
        transform = transforms.Compose([
            transforms.Resize((256, 128)),        # 调整图像尺寸
            transforms.ToTensor(),               # 转换为张量
            normalize,                          # 标准化
        ])
    return transform

def detect_camera_num_from_weights(weight_path):
    """
    从模型权重文件中自动检测相机数量
    
    Args:
        weight_path (str): 模型权重文件路径
        
    Returns:
        int: 检测到的相机数量，默认为4
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    for key in checkpoint:
        if 'BACKBONE.cv_embed' in key:
            # 从cv_embed层的形状推断相机数量
            return checkpoint[key].shape[0]
    return 4  # 默认相机数量

def process_gallery_query(root_dir, modality):
    """
    处理数据集，分离Gallery和Query图像
    
    Args:
        root_dir (str): 数据集根目录
        modality (str): 模态类型（RGB/NI/TI）
        
    Returns:
        tuple: (gallery_paths, query_paths) - Gallery和Query图像路径列表
    """
    gallery_paths, query_paths = [], []
    dir_path = os.path.join(root_dir, 'test', modality)
    
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith('.jpg'):
            full_path = os.path.join(dir_path, fname)
            # 根据文件名倒数第5位数字的奇偶性分离Gallery和Query
            # 偶数索引 -> Gallery，奇数索引 -> Query
            if int(fname[-5]) % 2 == 0:
                gallery_paths.append(full_path)
            else:
                query_paths.append(full_path)
    return gallery_paths, query_paths

def get_pid_from_path(path):
    """
    从图像路径中提取人员ID
    
    Args:
        path (str): 图像文件路径
        
    Returns:
        int: 人员ID（从文件名前6位提取）
    """
    return int(os.path.basename(path)[:6])

def extract_feature(model, paths, transform, device, modality):
    """
    使用指定模型提取图像特征
    
    Args:
        model: 训练好的ReID模型
        paths (list): 图像路径列表
        transform: 图像预处理变换
        device: 计算设备（CPU/GPU）
        modality (str): 模态类型（RGB/NI/TI）
        
    Returns:
        np.ndarray: 提取的特征矩阵，形状为(n_samples, feature_dim)
    """
    features = []
    for p in tqdm(paths, desc=f"提取 {modality} 特征"):
        # 加载并预处理图像
        img = Image.open(p).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 构建多模态输入字典（只激活指定模态）
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),  # RGB模态占位符
            'NI': torch.zeros_like(img_tensor),   # NI模态占位符  
            'TI': torch.zeros_like(img_tensor)    # TI模态占位符
        }
        input_dict[modality] = img_tensor  # 激活当前模态
        
        # 特征提取（禁用梯度计算以节省内存）
        with torch.no_grad():
            feat = model(input_dict, 
                        cam_label=torch.tensor([0]).to(device), 
                        view_label=torch.tensor([0]).to(device))
        features.append(feat.cpu().numpy())
    
    return np.vstack(features)  # 堆叠所有特征为矩阵

def compute_topk_correct(query_feats, gallery_feats, query_paths, gallery_paths, k=9):
    """
    计算Top-K检索的正确匹配数量
    
    功能说明：
    - 计算每个 Query 图像与所有 Gallery 图像的相似度
    - 找出 Top-K 最相似的 Gallery 图像
    - 统计这 K 个结果中有多少个是正确的匹配（相同人员ID）
    - 用于评估模型在特定 Query 上的检索性能
    
    算法流程：
    1. 计算相似度矩阵：sim_mat = query_feats @ gallery_feats.T
       - 形状：(n_query, n_gallery)
       - 每个元素 sim_mat[i, j] 表示 Query i 与 Gallery j 的相似度
    2. 对每个 Query，找出 Top-K 最相似的 Gallery 图像
    3. 检查这 K 个结果中，有多少个与 Query 是相同的人员ID
    
    Args:
        query_feats (np.ndarray): Query特征矩阵，形状为 (n_query, feature_dim)
        gallery_feats (np.ndarray): Gallery特征矩阵，形状为 (n_gallery, feature_dim)
        query_paths (list): Query图像路径列表，长度为 n_query
        gallery_paths (list): Gallery图像路径列表，长度为 n_gallery
        k (int): Top-K检索的K值，默认为9。表示返回前K个最相似的结果
        
    Returns:
        dict: {query_path: correct_count} - 每个Query的Top-K正确匹配数
              - key: Query图像路径
              - value: Top-K中正确匹配的数量（0 到 k 之间的整数）
              
    示例:
        >>> query_feats = np.random.randn(10, 512)  # 10个Query，512维特征
        >>> gallery_feats = np.random.randn(100, 512)  # 100个Gallery，512维特征
        >>> correct_counts = compute_topk_correct(query_feats, gallery_feats, query_paths, gallery_paths, k=10)
        >>> # 结果: {'query1.jpg': 8, 'query2.jpg': 10, ...}  # 每个Query的Top-10正确数
    """
    # ========== 步骤 1: 计算相似度矩阵 ==========
    # 使用矩阵乘法计算所有 Query-Gallery 对之间的相似度
    # 假设特征已经 L2 归一化，则点积等于余弦相似度
    # sim_mat[i, j] = query_feats[i] · gallery_feats[j]^T
    sim_mat = np.matmul(query_feats, gallery_feats.T)  # 形状: (n_query, n_gallery)
    
    correct_counts = {}  # 存储每个Query的正确匹配数
    
    # ========== 步骤 2: 对每个Query计算Top-K正确匹配数 ==========
    for i, q_path in enumerate(query_paths):
        q_pid = get_pid_from_path(q_path)  # 从路径提取Query的人员ID（用于判断匹配正确性）
        
        # 获取Top-K最相似的Gallery图像索引
        # np.argsort(sim_mat[i]) 返回相似度从小到大的索引
        # [::-1] 反转，得到从大到小的索引（最相似的在前面）
        # [:k] 取前k个，得到Top-K索引
        indices = np.argsort(sim_mat[i])[::-1][:k]
        
        # 统计Top-K中正确匹配的数量
        # 正确匹配的定义：Gallery图像的人员ID与Query的人员ID相同
        correct = sum(get_pid_from_path(gallery_paths[j]) == q_pid for j in indices)
        correct_counts[q_path] = correct
    
    return correct_counts

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='ReID模型性能对比分析工具')
    parser.add_argument('--dataset_root', type=str, default='data/RGBNT201',
                        help='数据集根目录路径')
    parser.add_argument('--config_path', type=str, default='configs/RGBNT201/MambaPro_moe.yml',
                        help='配置文件路径')
    parser.add_argument('--old_model_path', type=str, default='pths/baseline_MambaProbest.pth',
                        help='旧模型（基线模型）权重路径')
    parser.add_argument('--new_model_path', type=str, default='pths/moe_innovation_MambaProbest.pth',
                        help='新模型（创新模型）权重路径')
    parser.add_argument('--top_k', type=int, default=9,
                        help='Top-K检索的K值，默认为9')
    return parser.parse_args()

def main():
    """
    主函数：执行ReID模型性能对比分析
    
    功能概述：
    本工具用于对比分析两个ReID模型（旧模型 vs 新模型）在不同模态下的检索性能，
    特别关注旧模型表现优于新模型的案例，用于模型改进分析。
    
    分析流程：
    1. 【模型加载】加载两个训练好的ReID模型（旧模型 vs 新模型）
       - 从配置文件读取模型架构参数
       - 从权重文件加载训练好的模型参数
       
    2. 【初步筛选】在RGB模态下筛选旧模型表现良好的Query图像
       - 使用旧模型提取RGB模态特征
       - 计算每个Query的Top-K检索结果
       - 筛选出Top-K正确数≥K的Query（旧模型表现优秀）
       - 这些Query构成候选集合P
       
    3. 【多模态对比】对筛选出的图像，在RGB/NI/TI三种模态下对比新旧模型性能
       - 对集合P中的每个Query图像：
         a. 在RGB/NI/TI三种模态下分别提取特征（使用旧模型和新模型）
         b. 计算Top-K检索结果
         c. 统计正确匹配数
         d. 对比新旧模型的性能
       
    4. 【结果筛选】识别旧模型在所有模态下都优于新模型的图像样本
       - 筛选条件：旧模型在RGB/NI/TI三种模态下的正确匹配数都 > 新模型
       - 这些样本构成最终结果集合S
       
    5. 【结果输出】输出详细的性能对比统计信息
       - 打印每个样本的详细对比结果
       - 显示各模态下的性能差异
       
    应用场景：
    - 模型改进分析：找出新模型表现不如旧模型的案例，分析原因
    - 困难样本识别：识别对模型具有挑战性的样本
    - 模态性能分析：分析不同模态下的模型表现差异
    
    输出示例：
        [1] 图像 ID: 000123
            图像路径: data/RGBNT201/test/RGB/000123_cam1_0_01.jpg
            匹配统计（Top-9正确匹配数）:
               ▸ RGB | 旧模型:  9/9   新模型:  7/9   优势: +2
               ▸ NI  | 旧模型:  8/9   新模型:  6/9   优势: +2
               ▸ TI  | 旧模型:  9/9   新模型:  8/9   优势: +1
    """
    # ========== 解析命令行参数 ==========
    args = parse_args()
    
    # ========== 配置参数 ==========
    dataset_root = args.dataset_root
    config_path = args.config_path
    old_model_path = args.old_model_path
    new_model_path = args.new_model_path
    top_k = args.top_k
    
    # 如果上述路径不存在，可以尝试以下备选路径：
    # old_model_path = "outputs/baseline_experiment/models/MambaProbest.pth"
    # new_model_path = "outputs/moe_innovation_experiment/models/MambaProbest.pth"
    
    # ========== 路径检查和自动检测 ==========
    print("🔍 检查模型权重文件...")
    
    # 检查旧模型路径
    if not os.path.exists(old_model_path):
        print(f"⚠️  旧模型路径不存在: {old_model_path}")
        # 尝试在outputs目录中查找
        potential_old_paths = [
            "outputs/baseline_experiment/models/MambaProbest.pth",
            "outputs/baseline_thesis/models/MambaProbest.pth",
            "pths/MambaProbest.pth"
        ]
        for path in potential_old_paths:
            if os.path.exists(path):
                old_model_path = path
                print(f"✅ 找到旧模型: {old_model_path}")
                break
        else:
            print("❌ 未找到旧模型权重文件，请检查路径")
            return
    
    # 检查新模型路径
    if not os.path.exists(new_model_path):
        print(f"⚠️  新模型路径不存在: {new_model_path}")
        # 尝试在outputs目录中查找
        potential_new_paths = [
            "outputs/moe_innovation_experiment/models/MambaProbest.pth",
            "outputs/moe_innovation_experiment/MambaProbest.pth",
            "pths/moe_MambaProbest.pth"
        ]
        for path in potential_new_paths:
            if os.path.exists(path):
                new_model_path = path
                print(f"✅ 找到新模型: {new_model_path}")
                break
        else:
            print("❌ 未找到新模型权重文件，请检查路径")
            return
    
    print(f"📁 旧模型路径: {old_model_path}")
    print(f"📁 新模型路径: {new_model_path}")
    
    # 检查数据集路径
    if not os.path.exists(dataset_root):
        print(f"❌ 数据集路径不存在: {dataset_root}")
        print("请确保RGBNT201数据集已正确放置在data/目录下")
        return
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return

    # 初始化设备和数据预处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms(is_train=False)
    print(f"🔧 使用设备: {device}")

    # ========== 模型加载 ==========
    print("📦 加载模型配置和权重...")
    cfg.merge_from_file(config_path)
    cfg.freeze()
    camera_num = detect_camera_num_from_weights(old_model_path)
    num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    print(f"📊 检测到相机数量: {camera_num}, 类别数量: {num_class}")

    # 加载旧模型
    print("🔄 加载旧模型...")
    model_old = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model_old.load_param(old_model_path)
    model_old.eval()

    # 加载新模型  
    print("🔄 加载新模型...")
    model_new = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model_new.load_param(new_model_path)
    model_new.eval()
    print("✅ 模型加载完成")

    # ========== 步骤1：构造P集合（RGB模态下旧模型表现良好的Query图像） ==========
    print("\n🔍 步骤1: 筛选RGB模态下旧模型表现良好的Query图像...")
    g_rgb, q_rgb = process_gallery_query(dataset_root, 'RGB')
    print(f"📊 RGB模态 - Gallery: {len(g_rgb)}张, Query: {len(q_rgb)}张")
    
    # 使用旧模型提取RGB模态特征
    f_q_old_rgb = extract_feature(model_old, q_rgb, transform, device, 'RGB')
    f_g_old_rgb = extract_feature(model_old, g_rgb, transform, device, 'RGB')
    
    # 计算旧模型在RGB模态下的Top-K正确匹配数
    correct_topk_old_rgb = compute_topk_correct(f_q_old_rgb, f_g_old_rgb, q_rgb, g_rgb, k=top_k)
    
    # 筛选Top-K正确数≥K的Query图像（旧模型表现优秀）
    P = [p for p, c in correct_topk_old_rgb.items() if c >= top_k]
    print(f"✅ RGB模态下旧模型匹配正确 ≥{top_k} 的图像数: {len(P)}")

    # ========== 步骤2&3：多模态性能对比分析 ==========
    print("\n🔍 步骤2&3: 对比新旧模型在三模态下的性能...")
    modalities = ['RGB', 'NI', 'TI']
    S = []  # 存储旧模型在所有模态下都优于新模型的图像

    for p in tqdm(P, desc="对比新旧模型中三模态的匹配结果"):
        pid = get_pid_from_path(p)
        passed = True  # 是否通过所有模态的测试
        stats = {'old': {}, 'new': {}}  # 记录各模态的性能统计

        # 对每个模态进行性能对比
        for modality in modalities:
            # 获取当前模态的Gallery图像
            g_paths, _ = process_gallery_query(dataset_root, modality)
            
            # 使用旧模型提取特征
            f_q_old = extract_feature(model_old, [p], transform, device, modality)
            f_g_old = extract_feature(model_old, g_paths, transform, device, modality)
            
            # 使用新模型提取特征
            f_q_new = extract_feature(model_new, [p], transform, device, modality)
            f_g_new = extract_feature(model_new, g_paths, transform, device, modality)

            # 计算相似度并获取Top-K索引
            sim_old = np.dot(f_q_old[0], f_g_old.T)
            sim_new = np.dot(f_q_new[0], f_g_new.T)

            idx_old = np.argsort(sim_old)[::-1][:top_k]  # 旧模型Top-K
            idx_new = np.argsort(sim_new)[::-1][:top_k]  # 新模型Top-K

            # 统计Top-K中的正确匹配数
            correct_old = sum(get_pid_from_path(g_paths[i]) == pid for i in idx_old)
            correct_new = sum(get_pid_from_path(g_paths[i]) == pid for i in idx_new)

            # 记录性能统计
            stats['old'][modality] = correct_old
            stats['new'][modality] = correct_new

            # 检查旧模型是否在该模态下优于新模型
            if correct_old <= correct_new:
                passed = False  # 如果新模型不差于旧模型，则不通过测试

        # 如果旧模型在所有模态下都优于新模型，则加入结果集
        if passed:
            S.append((p, stats))

    # 步骤4：输出统计信息
        # 步骤4：输出统计信息
    print(f"\n�� 满足三模态旧模型准确数均高于新模型的图像数量：{len(S)}")

    for i, (path, stat) in enumerate(S):
        image_id = os.path.basename(path).split('_')[0]  # e.g., 000123_cam1_0_01.jpg -> 000123
        print(f"\n[{i+1}] 图像 ID: {image_id}")
        print(f"     图像路径: {path}")
        print(f"     匹配统计（Top-{top_k}正确匹配数）:")
        for m in modalities:
            old_score = stat['old'][m]
            new_score = stat['new'][m]
            improvement = old_score - new_score
            print(f"       ▸ {m:<3} | 旧模型: {old_score:2d}/{top_k}   新模型: {new_score:2d}/{top_k}   优势: +{improvement}")



if __name__ == '__main__':
    main()

