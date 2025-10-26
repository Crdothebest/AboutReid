#!/usr/bin/env python3
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class SimpleAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/get_models':
            # 返回模型列表
            models_data = {
                "models": [
                    {
                        "id": "Model_A",
                        "supports": {
                            "sliding_window": [4, 8, 16],
                            "fusion_method": ["concat", "mlp", "attention_fusion"],
                            "use_moe": True
                        }
                    },
                    {
                        "id": "Model_B", 
                        "supports": {
                            "sliding_window": [8],
                            "fusion_method": ["concat"],
                            "use_moe": False
                        }
                    }
                ]
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(models_data).encode())
            
        elif self.path == '/api/get_random_target_id':
            # 返回随机目标ID
            import random
            import os
            from pathlib import Path
            
            # 从实际存在的文件中随机选择
            data_root = Path(__file__).resolve().parent.parent / 'frontend' / '1-testData' / 'test'
            rgb_dir = data_root / 'RGB'
            
            if rgb_dir.exists():
                # 获取所有RGB文件
                rgb_files = list(rgb_dir.glob('*.jpg'))
                if rgb_files:
                    # 随机选择一个文件
                    selected_file = random.choice(rgb_files)
                    # 提取基础ID（去掉_cam1_0_00等后缀）
                    target_id = selected_file.stem.split('_')[0]
                    
                    images_data = {
                        "target_id": target_id,
                        "images": {
                            "RGB": f"/datasets/RGB/{selected_file.name}",
                            "NIR": f"/datasets/NI/{selected_file.name}", 
                            "TI": f"/datasets/TI/{selected_file.name}"
                        }
                    }
                else:
                    # 如果没有文件，使用默认值
                    target_id = "000258"
                    images_data = {
                        "target_id": target_id,
                        "images": {
                            "RGB": f"/datasets/RGB/{target_id}_cam1_0_00.jpg",
                            "NIR": f"/datasets/NI/{target_id}_cam1_0_00.jpg", 
                            "TI": f"/datasets/TI/{target_id}_cam1_0_00.jpg"
                        }
                    }
            else:
                # 如果目录不存在，使用默认值
                target_id = "000258"
                images_data = {
                    "target_id": target_id,
                    "images": {
                        "RGB": f"/datasets/RGB/{target_id}_cam1_0_00.jpg",
                        "NIR": f"/datasets/NI/{target_id}_cam1_0_00.jpg", 
                        "TI": f"/datasets/TI/{target_id}_cam1_0_00.jpg"
                    }
                }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(images_data).encode())
            
        elif self.path == '/api/get_rank_results':
            # 返回可用的检索结果文件列表
            from pathlib import Path
            import os
            
            rank1_dir = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-1_results' / 'run_20251024_220340'
            rank5_dir = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-5_results' / 'run_20251026_003313'
            rank10_dir = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-10_results' / 'run_20251017_175911'
            
            results_data = {
                "rank1_results": [],
                "rank5_results": [],
                "rank10_results": []
            }
            
            # 获取Rank1结果文件
            if rank1_dir.exists():
                for file_path in rank1_dir.glob('*.png'):
                    if 'baseline' in file_path.name:
                        results_data["rank1_results"].append({
                            "filename": file_path.name,
                            "target_id": file_path.stem.split('_')[3],  # 提取ID
                            "type": "baseline",
                            "url": f"/datasets/rank1_results/{file_path.name}"
                        })
                    elif 'your_model' in file_path.name:
                        results_data["rank1_results"].append({
                            "filename": file_path.name,
                            "target_id": file_path.stem.split('_')[3],  # 提取ID
                            "type": "your_model",
                            "url": f"/datasets/rank1_results/{file_path.name}"
                        })
            
            # 获取Rank5结果文件
            if rank5_dir.exists():
                for file_path in rank5_dir.glob('*.png'):
                    if 'baseline' in file_path.name:
                        results_data["rank5_results"].append({
                            "filename": file_path.name,
                            "target_id": file_path.stem.split('_')[3],  # 提取ID
                            "type": "baseline",
                            "url": f"/datasets/rank5_results/{file_path.name}"
                        })
                    elif 'your_model' in file_path.name:
                        results_data["rank5_results"].append({
                            "filename": file_path.name,
                            "target_id": file_path.stem.split('_')[3],  # 提取ID
                            "type": "your_model",
                            "url": f"/datasets/rank5_results/{file_path.name}"
                        })
            
            # 获取Rank10结果文件
            if rank10_dir.exists():
                for file_path in rank10_dir.glob('*.png'):
                    if 'baseline' in file_path.name:
                        results_data["rank10_results"].append({
                            "filename": file_path.name,
                            "target_id": file_path.stem.split('_')[3],  # 提取ID
                            "type": "baseline",
                            "url": f"/datasets/rank10_results/{file_path.name}"
                        })
                    elif 'your_model' in file_path.name:
                        results_data["rank10_results"].append({
                            "filename": file_path.name,
                            "target_id": file_path.stem.split('_')[3],  # 提取ID
                            "type": "your_model",
                            "url": f"/datasets/rank10_results/{file_path.name}"
                        })
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(results_data).encode())
            
        elif self.path.startswith('/api/rank_image/'):
            # 处理检索结果图片请求: /api/rank_image/{target_id}/{modality}/{rank}/{model_type}
            from pathlib import Path
            import os
            
            # 解析路径: /api/rank_image/000273/RGB/10/baseline
            path_parts = self.path.split('/')
            if len(path_parts) >= 7:  # /api/rank_image/target_id/modality/rank/model_type
                target_id = path_parts[3]
                modality = path_parts[4]
                rank = path_parts[5]
                model_type = path_parts[6]
                
                # 根据rank选择对应的结果目录
                if rank == "1":
                    results_dir = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-1_results' / 'run_20251024_220340'
                elif rank == "5":
                    results_dir = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-5_results' / 'run_20251026_003313'
                elif rank == "10":
                    results_dir = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-10_results' / 'run_20251017_175911'
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                
                # 构建文件名
                if model_type == "baseline":
                    filename = f"multimodal_ranked_list_{target_id}_top{rank}_baseline.png"
                elif model_type == "your_model":
                    filename = f"multimodal_ranked_list_{target_id}_top{rank}_your_model.png"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                
                file_path = results_dir / filename
                
                # 调试信息
                print(f"🔍 调试信息:")
                print(f"   目标ID: {target_id}")
                print(f"   模态: {modality}")
                print(f"   Rank: {rank}")
                print(f"   模型类型: {model_type}")
                print(f"   结果目录: {results_dir}")
                print(f"   文件名: {filename}")
                print(f"   完整路径: {file_path}")
                print(f"   文件存在: {file_path.exists()}")
                
                if file_path.exists() and file_path.is_file():
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    error_msg = {"error": f"File not found: {filename}"}
                    self.wfile.write(json.dumps(error_msg).encode())
            else:
                self.send_response(404)
                self.end_headers()
            
        elif self.path.startswith('/datasets/'):
            # 处理静态文件服务
            import os
            from pathlib import Path
            
            # 构建文件路径
            if self.path.startswith('/datasets/rank1_results/'):
                # 处理Rank1结果图片
                rank1_results_root = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-1_results' / 'run_20251024_220340'
                relative_path = self.path[len('/datasets/rank1_results/'):]
                file_path = rank1_results_root / relative_path
                file_path = file_path.resolve()
                
                # 安全检查：确保文件在允许的目录内
                if not str(file_path).startswith(str(rank1_results_root.resolve())):
                    self.send_response(403)
                    self.end_headers()
                    return
            elif self.path.startswith('/datasets/rank5_results/'):
                # 处理Rank5结果图片
                rank5_results_root = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-5_results' / 'run_20251026_003313'
                relative_path = self.path[len('/datasets/rank5_results/'):]
                file_path = rank5_results_root / relative_path
                file_path = file_path.resolve()
                
                # 安全检查：确保文件在允许的目录内
                if not str(file_path).startswith(str(rank5_results_root.resolve())):
                    self.send_response(403)
                    self.end_headers()
                    return
            elif self.path.startswith('/datasets/rank10_results/'):
                # 处理Rank10结果图片
                rank10_results_root = Path(__file__).resolve().parent.parent / 'Rank_results' / 'RGB_rank-10_results' / 'run_20251017_175911'
                relative_path = self.path[len('/datasets/rank10_results/'):]
                file_path = rank10_results_root / relative_path
                file_path = file_path.resolve()
                
                # 安全检查：确保文件在允许的目录内
                if not str(file_path).startswith(str(rank10_results_root.resolve())):
                    self.send_response(403)
                    self.end_headers()
                    return
            else:
                # 处理普通数据集图片
                data_root = Path(__file__).resolve().parent.parent / 'frontend' / '1-testData' / 'test'
                file_path = data_root / self.path[len('/datasets/'):]
                file_path = file_path.resolve()
                
                # 安全检查：确保文件在允许的目录内
                if not str(file_path).startswith(str(data_root.resolve())):
                    self.send_response(403)
                    self.end_headers()
                    return
                
            if file_path.exists() and file_path.is_file():
                self.send_response(200)
                if file_path.suffix.lower() == '.jpg':
                    self.send_header('Content-type', 'image/jpeg')
                else:
                    self.send_header('Content-type', 'application/octet-stream')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(f"File not found: {file_path}".encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    server = HTTPServer(('127.0.0.1', 8001), SimpleAPIHandler)
    print("🚀 简化后端服务启动在 http://127.0.0.1:8001")
    print("📡 可用的API端点:")
    print("   GET /api/get_models")
    print("   GET /api/get_random_target_id")
    print("   GET /api/get_rank_results")
    print("   GET /api/rank_image/{target_id}/{modality}/{rank}/{model_type}")
    print("   GET /datasets/rank1_results/*.png")
    print("   GET /datasets/rank5_results/*.png")
    print("   GET /datasets/rank10_results/*.png")
    print("   GET /datasets/RGB/*.jpg")
    print("   GET /datasets/NIR/*.jpg")
    print("   GET /datasets/TI/*.jpg")
    server.serve_forever()

if __name__ == '__main__':
    run_server()
