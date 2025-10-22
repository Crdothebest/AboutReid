#!/usr/bin/env python3
"""
测试后端API和文件访问
"""
import requests
import json

def test_backend():
    base_url = "http://localhost:8001"
    
    print("🔍 测试后端API...")
    print("=" * 50)
    
    try:
        # 测试根路径
        print("1. 测试根路径...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 后端运行正常")
            print(f"   RANK_RESULTS_ROOT: {data.get('RANK_RESULTS_ROOT')}")
            print(f"   RANK_RESULTS_EXISTS: {data.get('RANK_RESULTS_EXISTS')}")
        else:
            print(f"❌ 后端响应错误: {response.status_code}")
            return
            
        # 测试文件检查
        print("\n2. 测试文件检查...")
        response = requests.get(f"{base_url}/api/test_rank_file")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 文件检查完成")
            print(f"   文件路径: {data.get('file_path')}")
            print(f"   文件存在: {data.get('exists')}")
            print(f"   父目录存在: {data.get('parent_exists')}")
            print(f"   总文件数: {data.get('total_files_count', 0)}")
            print(f"   前5个文件: {data.get('files_in_parent_dir', [])[:5]}")
        else:
            print(f"❌ 文件检查失败: {response.status_code}")
            
        # 测试列出所有文件
        print("\n3. 测试列出所有文件...")
        response = requests.get(f"{base_url}/api/list_rank_files")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 文件列表获取完成")
            print(f"   模态: {data.get('modality')}")
            print(f"   Rank: {data.get('rank')}")
            print(f"   总文件数: {data.get('total_count', 0)}")
            files = data.get('files', [])
            if files:
                print(f"   前5个文件:")
                for i, file_info in enumerate(files[:5]):
                    print(f"     {i+1}. {file_info.get('filename')} ({file_info.get('size', 0)} bytes)")
        else:
            print(f"❌ 文件列表获取失败: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端，请确保后端服务正在运行")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_backend()
