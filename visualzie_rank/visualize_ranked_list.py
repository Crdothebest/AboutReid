"""
ReID模型Top-K Ranked List可视化工具

该脚本用于生成ReID模型的Top-K检索结果可视化，展示Query图像与Gallery图像的相似度排序，
并通过颜色框标注Ground Truth的正确性。

主要功能：
1. 加载训练好的ReID模型（.pth文件）
2. 对指定数据集进行Query-Gallery检索
3. 生成Top-K Ranked List可视化结果
4. 用颜色框标注匹配的正确性（绿色=正确，红色=错误）
5. 保存可视化结果到文件

作者：MambaPro团队
日期：2024
"""

import os
import sys
import torch
import numpy as np
import argparse
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# 添加项目根目录到Python路径
# 获取脚本所在目录的父目录（即MambaPro项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 调试信息
print(f"🔍 脚本目录: {script_dir}")
print(f"🔍 项目根目录: {project_root}")
print(f"🔍 Python路径: {sys.path[:3]}")

from modeling.make_model import make_model
from config import cfg

def build_transforms():
    """
    构建图像预处理变换管道
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform

def detect_camera_num_from_weights(weight_path):
    """
    从模型权重文件中自动检测相机数量
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    for key in checkpoint:
        if 'BACKBONE.cv_embed' in key:
            return checkpoint[key].shape[0]
    return 4

def process_gallery_query(root_dir, modality):
    """
    处理数据集，分离Gallery和Query图像
    """
    gallery_paths, query_paths = [], []
    dir_path = os.path.join(root_dir, 'test', modality)
    
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith('.jpg'):
            full_path = os.path.join(dir_path, fname)
            if int(fname[-5]) % 2 == 0:
                gallery_paths.append(full_path)
            else:
                query_paths.append(full_path)
    return gallery_paths, query_paths

def get_pid_from_path(path):
    """
    从图像路径中提取人员ID
    """
    return int(os.path.basename(path)[:6])

def extract_feature(model, paths, transform, device, modality):
    """
    使用指定模型提取图像特征
    
    功能说明：
    - 批量处理图像，提取 ReID 特征向量
    - 支持多模态输入（RGB、NI、TI），但每次只激活一种模态
    - 使用 torch.no_grad() 禁用梯度计算，节省内存和加速
    
    处理流程：
    1. 加载图像（PIL Image 格式）
    2. 预处理：调整尺寸、归一化、转换为张量
    3. 构建多模态输入字典（只激活指定模态）
    4. 模型前向传播，提取特征
    5. 转换为 numpy 数组并堆叠
    
    Args:
        model (nn.Module): 训练好的 ReID 模型
        paths (list): 图像路径列表，每个元素是一个图像文件路径
        transform (transforms.Compose): 图像预处理变换管道
        device (torch.device): 计算设备（CPU 或 GPU）
        modality (str): 模态类型，可选值：'RGB'、'NI'、'TI'
            - 'RGB': 可见光图像
            - 'NI': 近红外图像
            - 'TI': 热红外图像
        
    Returns:
        np.ndarray: 特征矩阵，形状为 (n_samples, feature_dim)
            - n_samples: 图像数量（等于 len(paths)）
            - feature_dim: 特征维度（通常为 512 或 768）
    
    示例:
        >>> model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
        >>> paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
        >>> features = extract_feature(model, paths, transform, device, 'RGB')
        >>> # features 形状: (3, 512)
    """
    features = []
    for p in tqdm(paths, desc=f"提取 {modality} 特征"):
        # 加载图像：使用 PIL 加载并转换为 RGB 格式（确保3通道）
        img = Image.open(p).convert('RGB')
        
        # 预处理：调整尺寸、归一化、转换为张量
        # transform 输出形状: [C, H, W]，值域已归一化
        img_tensor = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
        
        # 构建多模态输入字典
        # ReID 模型期望接收字典格式的输入，包含 RGB、NI、TI 三种模态
        # 对于单模态特征提取，只激活指定模态，其他模态用零张量填充
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),  # RGB 模态占位符（如果当前不是 RGB）
            'NI': torch.zeros_like(img_tensor),   # NI 模态占位符（如果当前不是 NI）
            'TI': torch.zeros_like(img_tensor)     # TI 模态占位符（如果当前不是 TI）
        }
        input_dict[modality] = img_tensor  # 激活当前模态
        
        # 特征提取：禁用梯度计算以节省内存和加速
        # cam_label 和 view_label 用于相机/视角嵌入（SIE），这里使用默认值
        with torch.no_grad():
            feat = model(
                input_dict, 
                cam_label=torch.tensor([0]).to(device),   # 相机ID（默认0）
                view_label=torch.tensor([0]).to(device)  # 视角ID（默认0）
            )
        # 转换为 numpy 数组并添加到列表
        features.append(feat.cpu().numpy())  # 移到 CPU 并转换为 numpy
    
    # 堆叠所有特征为矩阵：从列表转换为数组
    return np.vstack(features)  # 形状: (n_samples, feature_dim)

def compute_similarity_matrix(query_feats, gallery_feats):
    """
    计算 Query 和 Gallery 之间的相似度矩阵
    
    功能说明：
    - 使用矩阵乘法计算所有 Query-Gallery 对之间的相似度
    - 假设特征已经 L2 归一化，则点积等于余弦相似度
    - 相似度值范围：[-1, 1]（如果特征已归一化）
    
    算法：
        sim_mat[i, j] = query_feats[i] · gallery_feats[j]^T
        其中 · 表示点积（内积）
    
    Args:
        query_feats (np.ndarray): Query 特征矩阵，形状为 (n_query, feature_dim)
        gallery_feats (np.ndarray): Gallery 特征矩阵，形状为 (n_gallery, feature_dim)
        
    Returns:
        np.ndarray: 相似度矩阵，形状为 (n_query, n_gallery)
            - sim_mat[i, j] 表示第 i 个 Query 与第 j 个 Gallery 的相似度
            - 值越大表示越相似
    
    示例:
        >>> query_feats = np.random.randn(10, 512)  # 10个Query
        >>> gallery_feats = np.random.randn(100, 512)  # 100个Gallery
        >>> sim_mat = compute_similarity_matrix(query_feats, gallery_feats)
        >>> # sim_mat 形状: (10, 100)
        >>> # sim_mat[0, 5] 表示 Query 0 与 Gallery 5 的相似度
    """
    # 矩阵乘法：query_feats @ gallery_feats.T
    # 结果形状: (n_query, n_gallery)
    # 每个元素 sim_mat[i, j] = sum(query_feats[i, k] * gallery_feats[j, k] for k in range(feature_dim))
    return np.matmul(query_feats, gallery_feats.T)

def get_topk_ranked_results(query_feat, gallery_feats, gallery_paths, query_pid, k=9):
    """
    获取 Top-K 检索结果，包含相似度分数和 Ground Truth 标注
    
    功能说明：
    - 计算单个 Query 与所有 Gallery 的相似度
    - 找出 Top-K 最相似的 Gallery 图像
    - 为每个结果标注是否为正确匹配（相同人员ID）
    - 返回排序后的结果列表，包含排名、路径、相似度等信息
    
    算法流程：
    1. 计算相似度：sim = query_feat @ gallery_feats.T
    2. 排序：找出 Top-K 最相似的索引
    3. 标注：检查每个结果是否为正确匹配（gallery_pid == query_pid）
    4. 构建结果列表：包含排名、路径、相似度、正确性等信息
    
    Args:
        query_feat (np.ndarray): 单个 Query 的特征向量，形状为 (feature_dim,)
        gallery_feats (np.ndarray): Gallery 特征矩阵，形状为 (n_gallery, feature_dim)
        gallery_paths (list): Gallery 图像路径列表，长度为 n_gallery
        query_pid (int): Query 的人员ID（用于判断匹配正确性）
        k (int): Top-K 检索的 K 值，默认 9。表示返回前 K 个最相似的结果
        
    Returns:
        list: Top-K 检索结果列表，每个元素是一个字典，包含：
            - 'rank' (int): 排名（1 到 k）
            - 'gallery_path' (str): Gallery 图像路径
            - 'gallery_pid' (int): Gallery 图像的人员ID
            - 'similarity_score' (float): 相似度分数（值越大越相似）
            - 'is_correct' (bool): 是否为正确匹配（True 表示相同人员ID，False 表示不同）
    
    示例:
        >>> query_feat = np.random.randn(512)  # 单个Query特征
        >>> gallery_feats = np.random.randn(100, 512)  # 100个Gallery特征
        >>> results = get_topk_ranked_results(query_feat, gallery_feats, gallery_paths, query_pid=123, k=10)
        >>> # results[0] = {'rank': 1, 'gallery_path': '...', 'gallery_pid': 123, 'similarity_score': 0.95, 'is_correct': True}
    """
    # 计算相似度：单个 Query 与所有 Gallery 的点积
    # similarities 形状: (n_gallery,)
    similarities = np.dot(query_feat, gallery_feats.T)
    
    # 获取 Top-K 最相似的 Gallery 索引
    # np.argsort(similarities) 返回相似度从小到大的索引
    # [::-1] 反转，得到从大到小的索引（最相似的在前面）
    # [:k] 取前 k 个，得到 Top-K 索引
    topk_indices = np.argsort(similarities)[::-1][:k]
    
    # 构建结果列表
    ranked_results = []
    for i, idx in enumerate(topk_indices):
        gallery_path = gallery_paths[idx]  # 获取 Gallery 图像路径
        gallery_pid = get_pid_from_path(gallery_path)  # 从路径提取人员ID
        similarity_score = similarities[idx]  # 获取相似度分数
        is_correct = (gallery_pid == query_pid)  # 判断是否为正确匹配
        
        # 添加到结果列表
        ranked_results.append({
            'rank': i + 1,                    # 排名（从1开始）
            'gallery_path': gallery_path,     # Gallery 图像路径
            'gallery_pid': gallery_pid,       # Gallery 人员ID
            'similarity_score': similarity_score,  # 相似度分数
            'is_correct': is_correct          # 是否为正确匹配（True/False）
        })
    
    return ranked_results

def draw_ground_truth_box(img, is_correct, thickness=3):
    """
    在图像上绘制 Ground Truth 标注框
    
    功能说明：
    - 在图像周围绘制彩色边框，用于标注检索结果的正确性
    - 绿色框：表示正确匹配（Gallery 图像与 Query 是相同人员）
    - 红色框：表示错误匹配（Gallery 图像与 Query 是不同人员）
    - 用于可视化中快速识别检索结果的正确性
    
    颜色编码：
    - 绿色 (0, 255, 0)：正确匹配，表示模型成功找到了相同的人员
    - 红色 (255, 0, 0)：错误匹配，表示模型误匹配了不同的人员
    
    Args:
        img (np.ndarray): 输入图像，形状为 [H, W, 3]（RGB 格式，值域 [0, 255]）
        is_correct (bool): 是否为正确匹配
            - True: 绘制绿色框（正确匹配）
            - False: 绘制红色框（错误匹配）
        thickness (int): 边框厚度（像素），默认 3
    
    Returns:
        np.ndarray: 绘制了边框的图像，形状与输入相同（RGB 格式）
    
    示例:
        >>> img = cv2.imread('image.jpg')
        >>> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        >>> img_with_box = draw_ground_truth_box(img_rgb, is_correct=True, thickness=3)
        >>> # 图像周围会绘制绿色边框
    """
    img_copy = img.copy()  # 复制图像，避免修改原始图像
    
    # 根据匹配正确性选择颜色
    if is_correct:
        color = (0, 255, 0)  # 绿色框表示正确匹配 (RGB格式)
    else:
        color = (255, 0, 0)  # 红色框表示错误匹配 (RGB格式)
    
    # 绘制矩形边框
    # cv2.rectangle(img, pt1, pt2, color, thickness)
    # pt1: 左上角坐标 (0, 0)
    # pt2: 右下角坐标 (width, height)
    # 这样会在整个图像周围绘制边框
    cv2.rectangle(
        img_copy, 
        (0, 0),                           # 左上角坐标
        (img_copy.shape[1], img_copy.shape[0]),  # 右下角坐标（宽度，高度）
        color,                            # 边框颜色（RGB格式）
        thickness                         # 边框厚度
    )
    return img_copy

def create_ranked_visualization(query_path, ranked_results, output_path, k=10):
    """
    创建简化的 Top-K Ranked List 可视化结果
    
    功能说明：
    - 生成 ReID 检索结果的可视化图像
    - 显示 Query 图像和 Top-K 最相似的 Gallery 图像
    - 用绿色框标注正确匹配，红色框标注错误匹配
    - 便于直观评估模型的检索性能
    
    可视化布局：
    - 1 行，K+1 列
    - 第 1 列：Query 图像（无边框）
    - 第 2 到 K+1 列：Top-K Gallery 图像（带颜色边框）
    
    颜色编码：
    - 绿色边框：正确匹配（Gallery 与 Query 是相同人员）
    - 红色边框：错误匹配（Gallery 与 Query 是不同人员）
    
    Args:
        query_path (str): Query 图像路径
        ranked_results (list): Top-K 检索结果列表，每个元素包含：
            - 'rank' (int): 排名
            - 'gallery_path' (str): Gallery 图像路径
            - 'gallery_pid' (int): Gallery 人员ID
            - 'similarity_score' (float): 相似度分数
            - 'is_correct' (bool): 是否为正确匹配
        output_path (str): 输出图像路径（PNG 格式）
        k (int): Top-K 的 K 值，默认 10。应与 ranked_results 的长度一致
    
    输出：
        - 保存可视化图像到 output_path
        - 图像分辨率：300 DPI，适合论文使用
    
    示例:
        >>> query_path = 'data/test/RGB/000123_cam1_0_01.jpg'
        >>> ranked_results = get_topk_ranked_results(...)
        >>> create_ranked_visualization(query_path, ranked_results, 'output.png', k=10)
        >>> # 生成包含 Query 和 Top-10 Gallery 的可视化图像
    """
    # 设置图像布局：1行，Query + Top-K Gallery
    fig, axes = plt.subplots(1, k+1, figsize=(25, 4))
    fig.suptitle(f'Query and Top-{k} Ranked Results', fontsize=16, fontweight='bold')
    
    # 确保axes是一维数组（当只有一行时）
    if axes.ndim > 1:
        axes = axes.flatten()
    
    # 加载Query图像
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"⚠️  无法加载Query图像: {query_path}")
        return
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    query_pid = get_pid_from_path(query_path)
    
    # 显示Query图像
    axes[0].imshow(query_img)
    axes[0].set_title(f'Query\nID: {query_pid:06d}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 显示Top-K Gallery图像
    for i, result in enumerate(ranked_results):
        gallery_img = cv2.imread(result['gallery_path'])
        if gallery_img is None:
            print(f"⚠️  无法加载Gallery图像: {result['gallery_path']}")
            continue
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        # 添加Ground Truth标注框
        gallery_img_with_box = draw_ground_truth_box(gallery_img, result['is_correct'])
        
        # 显示图像
        axes[i+1].imshow(gallery_img_with_box)
        # 在标题中只显示排名和ID
        axes[i+1].set_title(f'Rank {i+1}\nID: {result["gallery_pid"]:06d}', 
                           fontsize=10, fontweight='bold')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化结果已保存: {output_path}")

def create_multimodal_ranked_visualization(query_paths, ranked_results_dict, output_path, k=10):
    """
    创建多模态Top-K Ranked List可视化结果
    显示RGB、NIR、TIR三种模态的结果，每行一种模态
    """
    modalities = ['RGB', 'NIR', 'TIR']
    modality_paths = ['RGB', 'NI', 'TI']  # 对应的文件夹名称
    
    # 设置图像布局：3行（RGB、NIR、TIR），每行Query + Top-K Gallery
    fig, axes = plt.subplots(3, k+1, figsize=(25, 12))
    fig.suptitle(f'Multi-modal Query and Top-{k} Ranked Results', fontsize=16, fontweight='bold')
    
    # 获取Query的PID
    query_pid = get_pid_from_path(query_paths['RGB'])
    
    for modality_idx, (modality, modality_folder) in enumerate(zip(modalities, modality_paths)):
        # 加载Query图像
        query_img = cv2.imread(query_paths[modality_folder])
        if query_img is None:
            print(f"⚠️  无法加载{modality} Query图像: {query_paths[modality_folder]}")
            continue
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # 显示Query图像
        axes[modality_idx, 0].imshow(query_img)
        axes[modality_idx, 0].set_title(f'{modality} Query\nID: {query_pid:06d}', 
                                       fontsize=12, fontweight='bold')
        axes[modality_idx, 0].axis('off')
        
        # 显示Top-K Gallery图像
        ranked_results = ranked_results_dict[modality_folder]
        for i, result in enumerate(ranked_results):
            gallery_img = cv2.imread(result['gallery_path'])
            if gallery_img is None:
                print(f"⚠️  无法加载{modality} Gallery图像: {result['gallery_path']}")
                continue
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
            
            # 添加Ground Truth标注框
            gallery_img_with_box = draw_ground_truth_box(gallery_img, result['is_correct'])
            
            # 显示图像
            axes[modality_idx, i+1].imshow(gallery_img_with_box)
            # 在标题中只显示排名和ID
            axes[modality_idx, i+1].set_title(f'Rank {i+1}\nID: {result["gallery_pid"]:06d}', 
                                           fontsize=10, fontweight='bold')
            axes[modality_idx, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 多模态可视化结果已保存: {output_path}")

def visualize_single_query(model, query_path, gallery_paths, gallery_feats, transform, device, modality, output_dir, k=10, model_type=""):
    """
    为单个Query生成Top-K Ranked List可视化
    """
    # 提取Query特征
    query_feat = extract_feature(model, [query_path], transform, device, modality)
    query_pid = get_pid_from_path(query_path)
    
    # 获取Top-K检索结果
    ranked_results = get_topk_ranked_results(query_feat[0], gallery_feats, gallery_paths, query_pid, k)
    
    # 生成可视化结果
    query_id = os.path.basename(query_path).split('_')[0]
    if model_type:
        output_path = os.path.join(output_dir, f'ranked_list_{query_id}_{modality}_top{k}_{model_type}.png')
    else:
        output_path = os.path.join(output_dir, f'ranked_list_{query_id}_{modality}_top{k}.png')
    create_ranked_visualization(query_path, ranked_results, output_path, k)
    
    return ranked_results

def visualize_multimodal_query(model, query_paths, gallery_paths_dict, gallery_feats_dict, transform, device, output_dir, k=10, model_type=""):
    """
    为多模态Query生成Top-K Ranked List可视化
    """
    query_id = get_pid_from_path(query_paths['RGB'])
    ranked_results_dict = {}
    
    # 为每种模态生成排名结果
    for modality in ['RGB', 'NI', 'TI']:
        if modality in query_paths and modality in gallery_paths_dict:
            query_feat = extract_feature(model, [query_paths[modality]], transform, device, modality)
            ranked_results_dict[modality] = get_topk_ranked_results(
                query_feat[0], gallery_feats_dict[modality], gallery_paths_dict[modality], query_id, k
            )
    
    # 生成输出路径
    if model_type:
        output_path = os.path.join(output_dir, f'multimodal_ranked_list_{query_id:06d}_top{k}_{model_type}.png')
    else:
        output_path = os.path.join(output_dir, f'multimodal_ranked_list_{query_id:06d}_top{k}.png')
    
    create_multimodal_ranked_visualization(query_paths, ranked_results_dict, output_path, k)
    
    return ranked_results_dict

def get_numbered_output_dir(base_output_dir, auto_number=False):
    """
    获取编号的输出目录（使用时间戳）
    """
    if not auto_number:
        return base_output_dir
    
    # 使用时间戳创建编号目录，作为子目录
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    numbered_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    return numbered_dir

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='ReID模型Top-K Ranked List可视化工具')
    parser.add_argument('--dataset_root', type=str, default='data/RGBNT201',
                        help='数据集根目录路径')
    parser.add_argument('--config_path', type=str, default='configs/RGBNT201/MambaPro_moe.yml',
                        help='配置文件路径')
    parser.add_argument('--model_path', type=str, default='pths/MambaProbest.pth',
                        help='模型权重路径')
    parser.add_argument('--baseline_model_path', type=str, default='/home/zubuntu/workspace/yzy/MambaPro/pths/MambaProbest.pth',
                        help='Baseline模型权重路径（用于对比）')
    parser.add_argument('--compare_models', action='store_true',
                        help='是否对比两个模型（改进模型 vs Baseline模型）')
    parser.add_argument('--dual_model_mode', action='store_true',
                        help='双模型模式：同时运行两个模型，为每个Query生成两个不同的Rank-10图')
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'NI', 'TI'],
                        help='模态类型')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-K检索的K值')
    parser.add_argument('--num_queries', type=int, default=10,
                        help='要可视化的Query数量（-1表示处理所有Query）')
    parser.add_argument('--test_all_queries', action='store_true',
                        help='测试所有Query图像，为每个行人ID生成Rank-10图')
    parser.add_argument('--output_dir', type=str, default='RGB_rank-10_results',
                        help='输出目录')
    parser.add_argument('--auto_number', action='store_true',
                        help='自动为输出文件编号（使用时间戳），避免覆盖之前的运行结果')
    parser.add_argument('--multimodal', action='store_true',
                        help='启用多模态模式，同时处理RGB、NIR、TIR三种模态')
    return parser.parse_args()

def main():
    """
    主函数：生成ReID模型Top-K Ranked List可视化结果
    """
    # 解析命令行参数
    args = parse_args()
    
    # 获取编号的输出目录
    if args.auto_number:
        args.output_dir = get_numbered_output_dir(args.output_dir, args.auto_number)
        print(f"📁 使用时间戳编号输出目录: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查路径
    if not os.path.exists(args.model_path):
        print(f"❌ 模型权重文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.dataset_root):
        print(f"❌ 数据集路径不存在: {args.dataset_root}")
        return
    
    if not os.path.exists(args.config_path):
        print(f"❌ 配置文件不存在: {args.config_path}")
        return
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms()
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    print("📦 加载模型配置和权重...")
    cfg.merge_from_file(args.config_path)
    cfg.freeze()
    camera_num = detect_camera_num_from_weights(args.model_path)
    num_class = getattr(cfg.DATASETS, 'NUM_CLASSES', 171)
    
    model = make_model(cfg, num_class=num_class, camera_num=camera_num).to(device)
    model.load_param(args.model_path)
    model.eval()
    print("✅ 模型加载完成")
    
    # 如果启用模型对比或双模型模式，加载Baseline模型
    baseline_model = None
    if args.compare_models or args.dual_model_mode:
        if args.dual_model_mode:
            print("🔄 加载Baseline模型（双模型模式）...")
        else:
            print("🔄 加载Baseline模型进行对比...")
        if not os.path.exists(args.baseline_model_path):
            print(f"❌ Baseline模型权重文件不存在: {args.baseline_model_path}")
            return
            
        # 构建Baseline模型（使用相同的配置，但加载不同的权重）
        print(f"🔄 构建Baseline模型...")
        baseline_camera_num = detect_camera_num_from_weights(args.baseline_model_path)
        baseline_model = make_model(cfg, num_class=num_class, camera_num=baseline_camera_num).to(device)
        baseline_model.load_param(args.baseline_model_path)
        baseline_model.eval()
        if args.dual_model_mode:
            print("✅ Baseline模型加载完成（双模型模式）")
        else:
            print("✅ Baseline模型加载完成")
    
    # 处理数据集
    # 初始化变量，避免未定义错误
    gallery_paths_dict = {}
    query_paths_dict = {}
    gallery_feats_dict = {}
    gallery_paths = None
    query_paths = None
    gallery_feats = None
    
    if args.multimodal:
        print("🔍 处理多模态数据集（RGB、NIR、TIR）...")
        # 多模态处理
        for modality in ['RGB', 'NI', 'TI']:
            print(f"  📁 处理 {modality} 模态...")
            gallery_paths, query_paths = process_gallery_query(args.dataset_root, modality)
            gallery_paths_dict[modality] = gallery_paths
            query_paths_dict[modality] = query_paths
            print(f"    📊 {modality} Gallery: {len(gallery_paths)}张, Query: {len(query_paths)}张")
            
            # 提取Gallery特征
            print(f"    🔄 提取 {modality} Gallery特征...")
            gallery_feats_dict[modality] = extract_feature(model, gallery_paths, transform, device, modality)
        
        # 选择要处理的Query（使用RGB模态的Query路径）
        if args.test_all_queries or args.num_queries == -1:
            selected_queries = query_paths_dict['RGB']
            print(f"🎯 处理所有Query进行多模态可视化: {len(selected_queries)}个")
        else:
            selected_queries = query_paths_dict['RGB'][:args.num_queries]
            print(f"🎯 选择{len(selected_queries)}个Query进行多模态可视化")
    else:
        print(f"🔍 处理 {args.modality} 模态数据集...")
        gallery_paths, query_paths = process_gallery_query(args.dataset_root, args.modality)
        print(f"📊 Gallery: {len(gallery_paths)}张, Query: {len(query_paths)}张")
        
        # 提取Gallery特征
        print("🔄 提取Gallery特征...")
        gallery_feats = extract_feature(model, gallery_paths, transform, device, args.modality)
        
        # 选择要处理的Query
        if args.test_all_queries or args.num_queries == -1:
            selected_queries = query_paths
            print(f"🎯 处理所有Query进行可视化: {len(selected_queries)}个")
        else:
            selected_queries = query_paths[:args.num_queries]
            print(f"🎯 选择{len(selected_queries)}个Query进行可视化")
    
    if args.dual_model_mode:
        # 双模型模式：同时运行两个模型，为每个Query生成两个不同的Rank-10图
        print(f"\n🔄 双模型模式处理 {len(selected_queries)} 个Query...")
        
        if args.multimodal:
            print("📋 将为每个Query生成多模态Rank-10图：")
            print("   - 您的模型: multimodal_ranked_list_{query_id}_top{k}_your_model.png")
            print("   - Baseline模型: multimodal_ranked_list_{query_id}_top{k}_baseline.png")
        else:
            print("📋 将为每个Query生成两个Rank-10图：")
            print("   - 您的模型: ranked_list_{query_id}_{modality}_top{k}_your_model.png")
            print("   - Baseline模型: ranked_list_{query_id}_{modality}_top{k}_baseline.png")
        
        # 检查Baseline模型是否已加载
        if baseline_model is None:
            print("❌ Baseline模型未加载，无法运行双模型模式")
            return
        
        if args.multimodal:
            # 多模态双模型模式
            print("🔄 提取Baseline多模态Gallery特征...")
            baseline_gallery_feats_dict = {}
            for modality in ['RGB', 'NI', 'TI']:
                print(f"  🔄 提取Baseline {modality} Gallery特征...")
                baseline_gallery_feats_dict[modality] = extract_feature(
                    baseline_model, gallery_paths_dict[modality], transform, device, modality
                )
        else:
            # 单模态双模型模式
            print("🔄 提取Baseline Gallery特征...")
            baseline_gallery_feats = extract_feature(baseline_model, gallery_paths, transform, device, args.modality)
        
        all_results = []
        for i, query_path in enumerate(tqdm(selected_queries, desc="处理Query")):
            query_id = os.path.basename(query_path).split('_')[0]
            print(f"\n🔄 处理Query {i+1}/{len(selected_queries)}: {query_id}")
            
            if args.multimodal:
                # 多模态双模型模式
                # 构建多模态Query路径
                query_paths_multimodal = {}
                for modality in ['RGB', 'NI', 'TI']:
                    # 从RGB路径构建其他模态的路径
                    rgb_path = query_path
                    if modality == 'RGB':
                        query_paths_multimodal[modality] = rgb_path
                    elif modality == 'NI':
                        ni_path = rgb_path.replace('/RGB/', '/NI/')
                        query_paths_multimodal[modality] = ni_path
                    elif modality == 'TI':
                        ti_path = rgb_path.replace('/RGB/', '/TI/')
                        query_paths_multimodal[modality] = ti_path
                
                # 生成您的模型的多模态可视化
                print(f"   🔄 生成您的模型多模态Rank-10图...")
                your_model_results_dict = visualize_multimodal_query(
                    model, query_paths_multimodal, gallery_paths_dict, gallery_feats_dict,
                    transform, device, args.output_dir, args.top_k, "your_model"
                )
                
                # 生成Baseline模型的多模态可视化
                print(f"   🔄 生成Baseline模型多模态Rank-10图...")
                baseline_results_dict = visualize_multimodal_query(
                    baseline_model, query_paths_multimodal, gallery_paths_dict, baseline_gallery_feats_dict,
                    transform, device, args.output_dir, args.top_k, "baseline"
                )
                
                # 统计结果（使用RGB模态的结果）
                your_model_correct = sum(1 for r in your_model_results_dict['RGB'] if r['is_correct'])
                baseline_correct = sum(1 for r in baseline_results_dict['RGB'] if r['is_correct'])
                
                print(f"   ✅ 您的模型 Top-{args.top_k}中正确匹配: {your_model_correct}/{args.top_k}")
                print(f"   ✅ Baseline模型 Top-{args.top_k}中正确匹配: {baseline_correct}/{args.top_k}")
                
                all_results.append({
                    'query_id': query_id,
                    'query_path': query_path,
                    'your_model_correct_count': your_model_correct,
                    'baseline_correct_count': baseline_correct,
                    'your_model_results': your_model_results_dict,
                    'baseline_results': baseline_results_dict
                })
            else:
                # 单模态双模型模式
                # 生成您的模型的可视化
                print(f"   🔄 生成您的模型Rank-10图...")
                your_model_results = visualize_single_query(
                    model, query_path, gallery_paths, gallery_feats, 
                    transform, device, args.modality, args.output_dir, args.top_k, "your_model"
                )
                
                # 生成Baseline模型的可视化
                print(f"   🔄 生成Baseline模型Rank-10图...")
                baseline_results = visualize_single_query(
                    baseline_model, query_path, gallery_paths, baseline_gallery_feats, 
                    transform, device, args.modality, args.output_dir, args.top_k, "baseline"
                )
                
                # 统计结果
                your_model_correct = sum(1 for r in your_model_results if r['is_correct'])
                baseline_correct = sum(1 for r in baseline_results if r['is_correct'])
                
                print(f"   ✅ 您的模型 Top-{args.top_k}中正确匹配: {your_model_correct}/{args.top_k}")
                print(f"   ✅ Baseline模型 Top-{args.top_k}中正确匹配: {baseline_correct}/{args.top_k}")
                
                all_results.append({
                    'query_id': query_id,
                    'query_path': query_path,
                    'your_model_correct_count': your_model_correct,
                    'baseline_correct_count': baseline_correct,
                    'your_model_results': your_model_results,
                    'baseline_results': baseline_results
                })
    elif args.compare_models:
        # 对比两个模型
        print(f"\n🔄 对比模式处理 {len(selected_queries)} 个Query...")
        
        # 提取Baseline Gallery特征
        print("🔄 提取Baseline Gallery特征...")
        baseline_gallery_feats = extract_feature(baseline_model, gallery_paths, transform, device, args.modality)
        
        all_results = []
        for i, query_path in enumerate(tqdm(selected_queries, desc="处理Query")):
            query_id = os.path.basename(query_path).split('_')[0]
            print(f"\n🔄 处理Query {i+1}/{len(selected_queries)}: {query_id}")
            
            # 生成改进模型的可视化
            improved_results = visualize_single_query(
                model, query_path, gallery_paths, gallery_feats, 
                transform, device, args.modality, args.output_dir, args.top_k, "improved"
            )
            
            # 生成Baseline模型的可视化
            baseline_results = visualize_single_query(
                baseline_model, query_path, gallery_paths, baseline_gallery_feats, 
                transform, device, args.modality, args.output_dir, args.top_k, "baseline"
            )
            
            # 统计结果
            improved_correct = sum(1 for r in improved_results if r['is_correct'])
            baseline_correct = sum(1 for r in baseline_results if r['is_correct'])
            
            print(f"   ✅ 改进模型 Top-{args.top_k}中正确匹配: {improved_correct}/{args.top_k}")
            print(f"   ✅ Baseline模型 Top-{args.top_k}中正确匹配: {baseline_correct}/{args.top_k}")
            
            all_results.append({
                'query_id': query_id,
                'query_path': query_path,
                'improved_correct_count': improved_correct,
                'baseline_correct_count': baseline_correct,
                'improved_results': improved_results,
                'baseline_results': baseline_results
            })
    else:
        # 单个模型
        print(f"\n🔄 处理 {len(selected_queries)} 个Query...")
        
        all_results = []
        for i, query_path in enumerate(tqdm(selected_queries, desc="处理Query")):
            query_id = os.path.basename(query_path).split('_')[0]
            print(f"\n🔄 处理Query {i+1}/{len(selected_queries)}: {query_id}")
            
            if args.multimodal:
                # 多模态单个模型模式
                # 构建多模态Query路径
                query_paths_multimodal = {}
                for modality in ['RGB', 'NI', 'TI']:
                    # 从RGB路径构建其他模态的路径
                    rgb_path = query_path
                    if modality == 'RGB':
                        query_paths_multimodal[modality] = rgb_path
                    elif modality == 'NI':
                        ni_path = rgb_path.replace('/RGB/', '/NI/')
                        query_paths_multimodal[modality] = ni_path
                    elif modality == 'TI':
                        ti_path = rgb_path.replace('/RGB/', '/TI/')
                        query_paths_multimodal[modality] = ti_path
                
                # 生成多模态可视化
                print(f"   🔄 生成多模态Rank-{args.top_k}图...")
                results_dict = visualize_multimodal_query(
                    model, query_paths_multimodal, gallery_paths_dict, gallery_feats_dict,
                    transform, device, args.output_dir, args.top_k
                )
                
                # 统计结果（使用RGB模态的结果）
                correct_count = sum(1 for r in results_dict['RGB'] if r['is_correct'])
                print(f"   ✅ Top-{args.top_k}中正确匹配: {correct_count}/{args.top_k}")
                
                all_results.append({
                    'query_id': query_id,
                    'query_path': query_path,
                    'correct_count': correct_count,
                    'results': results_dict
                })
            else:
                # 单模态单个模型模式
                # 检查必要的变量是否已定义
                if gallery_paths is None or gallery_feats is None:
                    raise ValueError(
                        f"❌ 错误：单模态模式下，gallery_paths 或 gallery_feats 未定义。"
                        f"这通常意味着代码逻辑错误。请检查是否在多模态模式下错误地进入了单模态分支。"
                    )
                ranked_results = visualize_single_query(
                    model, query_path, gallery_paths, gallery_feats, 
                    transform, device, args.modality, args.output_dir, args.top_k
                )
                
                # 统计结果
                correct_count = sum(1 for r in ranked_results if r['is_correct'])
                print(f"   ✅ Top-{args.top_k}中正确匹配: {correct_count}/{args.top_k}")
                
                all_results.append({
                    'query_id': query_id,
                    'query_path': query_path,
                    'correct_count': correct_count,
                    'ranked_results': ranked_results
                })
    
    # 生成汇总报告
    print(f"\n📊 生成汇总报告...")
    report_path = os.path.join(args.output_dir, 'summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        if args.dual_model_mode:
            f.write(f"双模型模式 - Top-{args.top_k} Ranked List可视化结果汇总\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"运行时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {args.output_dir}\n")
            if args.auto_number:
                f.write(f"时间戳编号: {args.output_dir.split('_')[-1]}\n")
            f.write(f"模态: {args.modality}\n")
            f.write(f"您的模型: {args.model_path}\n")
            f.write(f"Baseline模型: {args.baseline_model_path}\n")
            f.write(f"配置文件: {args.config_path}\n")
            f.write(f"数据集: {args.dataset_root}\n")
            f.write(f"Top-K: {args.top_k}\n")
            f.write(f"处理Query数量: {len(all_results)}\n\n")
            
            # 计算总体统计
            total_your_model_correct = sum(r['your_model_correct_count'] for r in all_results)
            total_baseline_correct = sum(r['baseline_correct_count'] for r in all_results)
            total_possible = len(all_results) * args.top_k
            
            f.write(f"总体统计:\n")
            f.write(f"  您的模型总体准确率: {total_your_model_correct}/{total_possible} ({total_your_model_correct/total_possible:.2%})\n")
            f.write(f"  Baseline模型总体准确率: {total_baseline_correct}/{total_possible} ({total_baseline_correct/total_possible:.2%})\n")
            f.write(f"  总体性能提升: {total_your_model_correct - total_baseline_correct} 个正确匹配\n\n")
            
            f.write(f"详细结果:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"注意：可视化图中绿色框表示正确匹配，红色框表示错误匹配\n\n")
            for result in all_results:
                f.write(f"Query {result['query_id']}:\n")
                f.write(f"  您的模型: {result['your_model_correct_count']}/{args.top_k} ({result['your_model_correct_count']/args.top_k:.2%})\n")
                f.write(f"  Baseline模型: {result['baseline_correct_count']}/{args.top_k} ({result['baseline_correct_count']/args.top_k:.2%})\n")
                f.write(f"  您的模型图: ranked_list_{result['query_id']}_{args.modality}_top{args.top_k}_your_model.png\n")
                f.write(f"  Baseline图: ranked_list_{result['query_id']}_{args.modality}_top{args.top_k}_baseline.png\n\n")
        elif args.compare_models:
            f.write(f"ReID模型对比 - Top-{args.top_k} Ranked List可视化结果汇总\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"运行时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {args.output_dir}\n")
            if args.auto_number:
                f.write(f"时间戳编号: {args.output_dir.split('_')[-1]}\n")
            f.write(f"模态: {args.modality}\n")
            f.write(f"改进模型: {args.model_path}\n")
            f.write(f"Baseline模型: {args.baseline_model_path}\n")
            f.write(f"配置文件: {args.config_path}\n")
            f.write(f"数据集: {args.dataset_root}\n")
            f.write(f"Top-K: {args.top_k}\n")
            f.write(f"处理Query数量: {len(all_results)}\n\n")
            
            # 计算总体统计
            total_improved_correct = sum(r['improved_correct_count'] for r in all_results)
            total_baseline_correct = sum(r['baseline_correct_count'] for r in all_results)
            total_possible = len(all_results) * args.top_k
            
            f.write(f"总体统计:\n")
            f.write(f"  改进模型总体准确率: {total_improved_correct}/{total_possible} ({total_improved_correct/total_possible:.2%})\n")
            f.write(f"  Baseline模型总体准确率: {total_baseline_correct}/{total_possible} ({total_baseline_correct/total_possible:.2%})\n")
            f.write(f"  总体性能提升: {total_improved_correct - total_baseline_correct} 个正确匹配\n\n")
            
            f.write(f"详细结果:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"注意：可视化图中绿色框表示正确匹配，红色框表示错误匹配\n\n")
            for result in all_results:
                f.write(f"Query {result['query_id']}:\n")
                f.write(f"  改进模型: {result['improved_correct_count']}/{args.top_k} ({result['improved_correct_count']/args.top_k:.2%})\n")
                f.write(f"  Baseline模型: {result['baseline_correct_count']}/{args.top_k} ({result['baseline_correct_count']/args.top_k:.2%})\n")
                f.write(f"  可视化文件: ranked_list_{result['query_id']}_{args.modality}_top{args.top_k}_improved.png\n")
                f.write(f"  可视化文件: ranked_list_{result['query_id']}_{args.modality}_top{args.top_k}_baseline.png\n\n")
        else:
            f.write(f"ReID模型Top-{args.top_k} Ranked List可视化结果汇总\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"运行时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {args.output_dir}\n")
            if args.auto_number:
                f.write(f"时间戳编号: {args.output_dir.split('_')[-1]}\n")
            f.write(f"模态: {args.modality}\n")
            f.write(f"模型: {args.model_path}\n")
            f.write(f"数据集: {args.dataset_root}\n")
            f.write(f"Top-K: {args.top_k}\n")
            f.write(f"处理Query数量: {len(all_results)}\n\n")
            
            # 计算总体统计
            total_correct = sum(r['correct_count'] for r in all_results)
            total_possible = len(all_results) * args.top_k
            
            f.write(f"总体统计:\n")
            f.write(f"  总体准确率: {total_correct}/{total_possible} ({total_correct/total_possible:.2%})\n\n")
            
            f.write(f"详细结果:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"注意：可视化图中绿色框表示正确匹配，红色框表示错误匹配\n\n")
            for result in all_results:
                f.write(f"Query {result['query_id']}:\n")
                f.write(f"  正确匹配数: {result['correct_count']}/{args.top_k} ({result['correct_count']/args.top_k:.2%})\n")
                f.write(f"  可视化文件: ranked_list_{result['query_id']}_{args.modality}_top{args.top_k}.png\n\n")
    
    print(f"\n🎉 可视化完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    if args.dual_model_mode:
        # 计算总体统计
        total_your_model_correct = sum(r['your_model_correct_count'] for r in all_results)
        total_baseline_correct = sum(r['baseline_correct_count'] for r in all_results)
        total_possible = len(all_results) * args.top_k
        
        print(f"📊 双模型模式结果:")
        print(f"   - 处理Query数量: {len(all_results)}")
        print(f"   - 您的模型总体准确率: {total_your_model_correct}/{total_possible} ({total_your_model_correct/total_possible:.2%})")
        print(f"   - Baseline模型总体准确率: {total_baseline_correct}/{total_possible} ({total_baseline_correct/total_possible:.2%})")
        improvement = total_your_model_correct - total_baseline_correct
        print(f"   - 总体性能提升: {improvement} 个正确匹配 ({improvement/total_possible:.2%})")
        print(f"   - 每个Query生成2个Rank-10图：您的模型图和Baseline模型图")
    elif args.compare_models:
        # 计算总体统计
        total_improved_correct = sum(r['improved_correct_count'] for r in all_results)
        total_baseline_correct = sum(r['baseline_correct_count'] for r in all_results)
        total_possible = len(all_results) * args.top_k
        
        print(f"📊 对比结果:")
        print(f"   - 处理Query数量: {len(all_results)}")
        print(f"   - 改进模型总体准确率: {total_improved_correct}/{total_possible} ({total_improved_correct/total_possible:.2%})")
        print(f"   - Baseline模型总体准确率: {total_baseline_correct}/{total_possible} ({total_baseline_correct/total_possible:.2%})")
        improvement = total_improved_correct - total_baseline_correct
        print(f"   - 总体性能提升: {improvement} 个正确匹配 ({improvement/total_possible:.2%})")
    else:
        # 计算总体统计
        total_correct = sum(r['correct_count'] for r in all_results)
        total_possible = len(all_results) * args.top_k
        
        print(f"📊 统计结果:")
        print(f"   - 处理Query数量: {len(all_results)}")
        print(f"   - 总体准确率: {total_correct}/{total_possible} ({total_correct/total_possible:.2%})")
    print(f"   - 汇总报告: {report_path}")

if __name__ == '__main__':
    main()
