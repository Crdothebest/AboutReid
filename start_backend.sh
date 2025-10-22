#!/bin/bash

# 设置环境变量
export DATA_ROOT=/Users/a11/Desktop/AboutReid
export MODELS_MANIFEST=/Users/a11/Desktop/AboutReid/frontend/3-configs/models_manifest.json
export RESULTS_ROOT=/Users/a11/Desktop/AboutReid/frontend/4-results
export DATASETS_PUBLIC_ROOT=/Users/a11/Desktop/AboutReid/frontend/1-testData/test

# 启动后端服务
echo "🚀 启动后端服务..."
echo "📁 数据根目录: $DATA_ROOT"
echo "📋 模型清单: $MODELS_MANIFEST"
echo "📊 结果目录: $RESULTS_ROOT"
echo "🖼️  数据集目录: $DATASETS_PUBLIC_ROOT"
echo "🌐 服务地址: http://localhost:8001"
echo ""

uvicorn backend.main:app --reload --port 8001
