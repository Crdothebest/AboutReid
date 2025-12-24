#!/usr/bin/env python3
"""
简单测试IDEA风格离线预编码功能（语法检查）
"""

def test_syntax_only():
    """仅测试语法和类定义"""
    print("🧪 测试IDEA风格预编码语法")
    print("=" * 50)

    try:
        # 只测试语法，不实际运行
        import ast

        with open('data/datasets/qwen_vl_loader.py', 'r', encoding='utf-8') as f:
            source_code = f.read()

        # 解析AST
        tree = ast.parse(source_code)
        print("✅ 语法检查通过")

        # 检查关键类是否存在
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if 'QwenVLTextLoader' in classes:
            print("✅ QwenVLTextLoader类定义存在")
        else:
            print("❌ QwenVLTextLoader类定义缺失")
            return False

        # 检查关键方法是否存在
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)

        required_methods = ['_load_text_features', 'get_text_feature', 'preload_to_gpu']
        for method in required_methods:
            if method in methods:
                print(f"✅ {method}方法存在")
            else:
                print(f"❌ {method}方法缺失")
                return False

        print("🎉 IDEA预编码语法测试通过！")
        return True

    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_syntax_only()