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
            
        elif self.path.startswith('/datasets/'):
            # 处理静态文件服务
            import os
            from pathlib import Path
            
            # 构建文件路径
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
    server.serve_forever()

if __name__ == '__main__':
    run_server()
