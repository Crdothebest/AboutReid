import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles


DATA_ROOT = Path(os.getenv('DATA_ROOT', Path(__file__).resolve().parent.parent))
MANIFEST_PATH = Path(os.getenv('MODELS_MANIFEST', DATA_ROOT / 'inference_configs' / 'models_manifest.json'))
RESULTS_ROOT = Path(os.getenv('RESULTS_ROOT', DATA_ROOT / 'results'))
DATASETS_PUBLIC_ROOT = Path(os.getenv('DATASETS_PUBLIC_ROOT', DATA_ROOT / 'frontend' / '1-testData' / 'test'))

app = FastAPI(default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATASETS_PUBLIC_ROOT.exists():
    app.mount('/datasets', StaticFiles(directory=str(DATASETS_PUBLIC_ROOT)), name='datasets')

# 挂载结果图片目录
if RESULTS_ROOT.exists():
    app.mount('/datasets/results', StaticFiles(directory=str(RESULTS_ROOT)), name='results')


@app.get('/api/get_models')
def get_models():
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=404, detail=f'models_manifest not found: {MANIFEST_PATH}')
    try:
        import orjson
        return ORJSONResponse(orjson.loads(MANIFEST_PATH.read_bytes()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/get_random_target_id')
def get_random_target_id():
    # 简化：从 DATASETS_PUBLIC_ROOT 下随机挑一个 ID（需按约定组织）
    import random
    modalities = ['RGB', 'NIR', 'TI']
    if not DATASETS_PUBLIC_ROOT.exists():
        raise HTTPException(status_code=404, detail=f'datasets not found: {DATASETS_PUBLIC_ROOT}')

    candidate_ids = set()
    for modality in modalities:
        # 处理 NI -> NIR 的映射
        dir_name = 'NI' if modality == 'NIR' else modality
        m_dir = DATASETS_PUBLIC_ROOT / dir_name
        if m_dir.exists():
            for p in m_dir.glob('*.jpg'):
                candidate_ids.add(p.stem)
    if not candidate_ids:
        raise HTTPException(status_code=404, detail='no candidate ids in datasets')

    target_id = random.choice(sorted(list(candidate_ids)))
    # 构建图片URL，注意 NI -> NIR 的映射
    images = {}
    for modality in modalities:
        dir_name = 'NI' if modality == 'NIR' else modality
        images[modality] = f'/datasets/{dir_name}/{target_id}.jpg'
    return {"target_id": target_id, "images": images}


@app.post('/api/reid_rank_query')
def reid_rank_query(payload: dict):
    # payload: { target_id, query_modality, config{ model_id, sliding_window, fusion_method, use_moe } }
    try:
        target_id = payload['target_id']
        query_modality = payload['query_modality']
        config = payload['config']
        model_id = config['model_id']
        sliding_window = config.get('sliding_window')
        fusion_method = config.get('fusion_method')
        use_moe = config.get('use_moe')
    except Exception:
        raise HTTPException(status_code=400, detail='invalid payload')

    # 组合路径：results/{model_id}/slw{?}_fusion-{?}_moe-{true|false}/{query_modality}/{target_id}.json
    parts = [f"slw{sliding_window}" if sliding_window is not None else None,
             f"fusion-{fusion_method}" if fusion_method else None,
             f"moe-{str(bool(use_moe)).lower()}" if use_moe is not None else None]
    sub = '_'.join([p for p in parts if p]) or 'default'
    
    # 处理特殊查询：ALL 表示需要合成所有模态的结果图片
    if query_modality == 'ALL':
        # 查找合成图片路径
        result_image_path = RESULTS_ROOT / model_id / sub / 'ALL' / f'{target_id}_result.jpg'
        
        if not result_image_path.exists():
            raise HTTPException(status_code=404, detail=f'result image not found: {result_image_path}')
        
        # 返回合成图片的URL（通过现有的 /datasets 路径）
        return ORJSONResponse({
            'resultImage': f'/datasets/{target_id}_result.jpg'
        })
    
    # 原有的单模态查询逻辑
    result_path = RESULTS_ROOT / model_id / sub / query_modality / f'{target_id}.json'

    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f'result not found: {result_path}')

    try:
        import orjson
        data = orjson.loads(result_path.read_bytes())
        # 若未包含 echo，补齐 echo 以便前端展示配置摘要
        if 'echo' not in data:
            data['echo'] = {
                'target_id': target_id,
                'query_modality': query_modality,
                'config': {
                    'model_id': model_id,
                    **({ 'sliding_window': sliding_window } if sliding_window is not None else {}),
                    **({ 'fusion_method': fusion_method } if fusion_method else {}),
                    **({ 'use_moe': use_moe } if use_moe is not None else {}),
                }
            }
        return ORJSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/')
def root():
    return { 'ok': True, 'DATA_ROOT': str(DATA_ROOT), 'RESULTS_ROOT': str(RESULTS_ROOT) }


