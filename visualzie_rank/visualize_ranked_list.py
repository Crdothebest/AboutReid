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
    """
    features = []
    for p in tqdm(paths, desc=f"提取 {modality} 特征"):
        img = Image.open(p).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        input_dict = {
            'RGB': torch.zeros_like(img_tensor),
            'NI': torch.zeros_like(img_tensor),
            'TI': torch.zeros_like(img_tensor)
        }
        input_dict[modality] = img_tensor
        
        with torch.no_grad():
            feat = model(input_dict, 
                        cam_label=torch.tensor([0]).to(device), 
                        view_label=torch.tensor([0]).to(device))
        features.append(feat.cpu().numpy())
    
    return np.vstack(features)

def compute_similarity_matrix(query_feats, gallery_feats):
    """
    计算Query和Gallery之间的相似度矩阵
    """
    return np.matmul(query_feats, gallery_feats.T)

def get_topk_ranked_results(query_feat, gallery_feats, gallery_paths, query_pid, k=9):
    """
    获取Top-K检索结果，包含相似度分数和Ground Truth标注
    """
    similarities = np.dot(query_feat, gallery_feats.T)
    topk_indices = np.argsort(similarities)[::-1][:k]
    
    ranked_results = []
    for i, idx in enumerate(topk_indices):
        gallery_path = gallery_paths[idx]
        gallery_pid = get_pid_from_path(gallery_path)
        similarity_score = similarities[idx]
        is_correct = (gallery_pid == query_pid)
        
        ranked_results.append({
            'rank': i + 1,
            'gallery_path': gallery_path,
            'gallery_pid': gallery_pid,
            'similarity_score': similarity_score,
            'is_correct': is_correct
        })
    
    return ranked_results

def draw_ground_truth_box(img, is_correct, thickness=3):
    """
    在图像上绘制Ground Truth标注框
    """
    img_copy = img.copy()
    color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色=正确，红色=错误
    cv2.rectangle(img_copy, (0, 0), (img_copy.shape[1], img_copy.shape[0]), color, thickness)
    return img_copy

def create_ranked_visualization(query_path, ranked_results, output_path, k=9):
    """
    创建Top-K Ranked List可视化结果
    """
    # 设置图像布局
    fig, axes = plt.subplots(2, k+1, figsize=(20, 8))
    fig.suptitle(f'Top-{k} Ranked List Visualization', fontsize=16, fontweight='bold')
    
    # 加载Query图像
    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    query_pid = get_pid_from_path(query_path)
    
    # 显示Query图像
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f'Query\nID: {query_pid:06d}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 显示Top-K Gallery图像
    for i, result in enumerate(ranked_results):
        gallery_img = cv2.imread(result['gallery_path'])
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        # 添加Ground Truth标注框
        gallery_img_with_box = draw_ground_truth_box(gallery_img, result['is_correct'])
        
        # 显示图像
        axes[0, i+1].imshow(gallery_img_with_box)
        axes[0, i+1].set_title(f'Rank {result["rank"]}\nID: {result["gallery_pid"]:06d}\nScore: {result["similarity_score"]:.3f}', 
                              fontsize=10, fontweight='bold')
        axes[0, i+1].axis('off')
        
        # 显示相似度条形图
        color = 'green' if result['is_correct'] else 'red'
        axes[1, i+1].barh(0, result['similarity_score'], color=color, alpha=0.7)
        axes[1, i+1].set_xlim(0, 1)
        axes[1, i+1].set_title(f'Score: {result["similarity_score"]:.3f}', fontsize=10)
        axes[1, i+1].set_yticks([])
        axes[1, i+1].grid(True, alpha=0.3)
    
    # 隐藏第一行第二列的空白
    axes[1, 0].axis('off')
    
    # 添加图例
    legend_elements = [
        patches.Patch(color='green', label='Correct Match'),
        patches.Patch(color='red', label='Incorrect Match')
    ]
    axes[1, 0].legend(handles=legend_elements, loc='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_single_query(model, query_path, gallery_paths, gallery_feats, transform, device, modality, output_dir, k=9):
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
    output_path = os.path.join(output_dir, f'ranked_list_{query_id}_{modality}_top{k}.png')
    create_ranked_visualization(query_path, ranked_results, output_path, k)
    
    return ranked_results

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
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'NI', 'TI'],
                        help='模态类型')
    parser.add_argument('--top_k', type=int, default=9,
                        help='Top-K检索的K值')
    parser.add_argument('--num_queries', type=int, default=10,
                        help='要可视化的Query数量')
    parser.add_argument('--output_dir', type=str, default='ranked_list_results',
                        help='输出目录')
    return parser.parse_args()

def main():
    """
    主函数：生成ReID模型Top-K Ranked List可视化结果
    """
    # 解析命令行参数
    args = parse_args()
    
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
    
    # 处理数据集
    print(f"🔍 处理 {args.modality} 模态数据集...")
    gallery_paths, query_paths = process_gallery_query(args.dataset_root, args.modality)
    print(f"📊 Gallery: {len(gallery_paths)}张, Query: {len(query_paths)}张")
    
    # 提取Gallery特征
    print("🔄 提取Gallery特征...")
    gallery_feats = extract_feature(model, gallery_paths, transform, device, args.modality)
    
    # 选择要可视化的Query
    selected_queries = query_paths[:args.num_queries]
    print(f"🎯 选择 {len(selected_queries)} 个Query进行可视化...")
    
    # 为每个Query生成可视化
    all_results = []
    for i, query_path in enumerate(tqdm(selected_queries, desc="生成可视化结果")):
        query_id = os.path.basename(query_path).split('_')[0]
        print(f"\n🔄 处理Query {i+1}/{len(selected_queries)}: {query_id}")
        
        # 生成可视化
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
        f.write(f"ReID模型Top-{args.top_k} Ranked List可视化结果汇总\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"模态: {args.modality}\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"数据集: {args.dataset_root}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"Query数量: {len(all_results)}\n\n")
        
        for i, result in enumerate(all_results):
            f.write(f"Query {i+1}: {result['query_id']}\n")
            f.write(f"  正确匹配数: {result['correct_count']}/{args.top_k}\n")
            f.write(f"  可视化文件: ranked_list_{result['query_id']}_{args.modality}_top{args.top_k}.png\n")
            f.write(f"  详细结果:\n")
            for rank_result in result['ranked_results']:
                status = "✓" if rank_result['is_correct'] else "✗"
                f.write(f"    Rank {rank_result['rank']}: {rank_result['gallery_pid']:06d} "
                       f"(Score: {rank_result['similarity_score']:.3f}) {status}\n")
            f.write("\n")
    
    # 计算总体统计
    total_correct = sum(r['correct_count'] for r in all_results)
    total_possible = len(all_results) * args.top_k
    accuracy = total_correct / total_possible if total_possible > 0 else 0
    
    print(f"\n🎉 可视化完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    print(f"📊 总体统计:")
    print(f"   - 处理的Query数量: {len(all_results)}")
    print(f"   - 总正确匹配数: {total_correct}/{total_possible}")
    print(f"   - Top-{args.top_k}准确率: {accuracy:.2%}")
    print(f"   - 汇总报告: {report_path}")

if __name__ == '__main__':
    main()
