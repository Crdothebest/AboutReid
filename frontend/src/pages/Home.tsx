import { useState } from 'react';
import { Spin, message } from 'antd';
import { useConfigStore } from '../store/config';
import ResultPanel from '../components/ResultPanel';
import ConfigPanel from '../components/ConfigPanel';
import TargetImagePanel from '../components/TargetImagePanel';
import { TrainingPanel } from '../components/TrainingPanel';

export default function Home() {
  const { toQueryConfig, target } = useConfigStore();
  const [result, setResult] = useState<any>();
  const [isPending, setIsPending] = useState(false);
  const [progress, setProgress] = useState(0);
  const [searchResults, setSearchResults] = useState<any>(null);
  const [isSearching, setIsSearching] = useState(false);

  // 处理检索功能
  const handleSearch = async () => {
    if (!target.targetId) {
      message.warning('请先抽取目标ID');
      return;
    }

    const cfg = toQueryConfig();
    if (!cfg) {
      message.warning('请先选择模型');
      return;
    }

    setIsSearching(true);
    try {
      // 模拟检索过程
      await new Promise(resolve => setTimeout(resolve, 2000));

      // 模拟检索结果
      const mockResults = {
        metrics: {
          mAP: 0.85,
          rank1: 0.92,
          rank5: 0.96,
          rank10: 0.98
        },
        rank_list: [
          { id: '000258_001', image_url: '/datasets/RGB/000258_cam1_0_01.jpg', is_correct: true },
          { id: '000258_002', image_url: '/datasets/RGB/000258_cam1_0_02.jpg', is_correct: true },
          { id: '000258_003', image_url: '/datasets/RGB/000258_cam1_0_03.jpg', is_correct: true },
          { id: '000260_001', image_url: '/datasets/RGB/000260_cam1_0_01.jpg', is_correct: false },
          { id: '000260_002', image_url: '/datasets/RGB/000260_cam1_0_02.jpg', is_correct: false },
          { id: '000269_001', image_url: '/datasets/RGB/000269_cam1_0_01.jpg', is_correct: false },
          { id: '000270_001', image_url: '/datasets/RGB/000270_cam1_0_01.jpg', is_correct: false },
          { id: '000271_001', image_url: '/datasets/RGB/000271_cam1_0_01.jpg', is_correct: false },
          { id: '000272_001', image_url: '/datasets/RGB/000272_cam1_0_01.jpg', is_correct: false },
          { id: '000273_001', image_url: '/datasets/RGB/000273_cam1_0_01.jpg', is_correct: false }
        ],
        echo: {
          target_id: target.targetId,
          query_modality: 'RGB', // 默认使用RGB模态
          config: cfg
        }
      };

      setSearchResults(mockResults);
      message.success('检索完成！');
    } catch (error) {
      message.error('检索失败');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSubmit = async () => {
    const cfg = toQueryConfig();
    if (!cfg) return message.warning('请先选择模型');

    setIsPending(true);

    try {
      // 生成模型训练结果名称
      const modelName = generateModelName(cfg);

      // 模拟训练过程，带进度条
      setProgress(0);
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(progressInterval);
            return 100;
          }
          return prev + Math.random() * 15; // 随机增加进度
        });
      }, 100);

      // 模拟训练时间
      await new Promise(resolve => setTimeout(resolve, 3000));

      // 确保进度条到100%
      setProgress(100);
      clearInterval(progressInterval);

      // 获取完整的配置信息用于显示
      const { modelId, slidingWindow, fusionMethod, useMoe } = useConfigStore.getState();
      const fullConfig = {
        model_id: modelId,
        sliding_window: slidingWindow, // 保持数组格式
        fusion_method: fusionMethod,
        use_moe: useMoe
      };

      // 返回训练结果
      const result = {
        modelName: modelName,
        config: fullConfig, // 使用完整配置
        status: '训练完成',
        timestamp: new Date().toLocaleString(),
        modelType: getModelType(fullConfig) // 使用完整配置生成模型类型
      };

      setResult(result);
      message.success('模型训练完成');
    } catch (error) {
      message.error('训练失败');
    } finally {
      setIsPending(false);
      setProgress(0);
    }
  };

  // 生成模型名称的函数
  const generateModelName = (config: any) => {
    const parts = [];

    // 模型类型
    if (config.model_id === 'baseline') {
      parts.push('Baseline');
    } else if (config.model_id === 'optimized') {
      parts.push('Optimized');
    }

    // 滑动窗口
    if (config.sliding_window) {
      parts.push(`SW${config.sliding_window}`);
    }

    // 融合方式
    if (config.fusion_method) {
      const fusionMap: { [key: string]: string } = {
        'concat': 'Concat',
        'mlp': 'MLP',
        'attention_fusion': 'Attn'
      };
      parts.push(fusionMap[config.fusion_method] || config.fusion_method);
    }

    // MoE
    if (config.use_moe) {
      parts.push('MoE');
    }

    // 时间戳
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '');
    parts.push(timestamp);

    return parts.join('_');
  };

  // 获取模型类型的函数
  const getModelType = (config: any) => {
    if (config.model_id === 'baseline') {
      return {
        baseType: 'Baseline 基础模型',
        config: '默认配置'
      };
    } else if (config.model_id === 'optimized') {
      const parts = [];

      // 多尺度部分
      if (config.sliding_window && Array.isArray(config.sliding_window) && config.sliding_window.length > 0) {
        const scales = config.sliding_window.sort((a: number, b: number) => a - b);
        parts.push(`多尺度[${scales.join(',')}]`);
      } else if (config.sliding_window && !Array.isArray(config.sliding_window)) {
        parts.push(`多尺度[${config.sliding_window}]`);
      }

      // 融合方式部分
      if (config.fusion_method) {
        const fusionNames: { [key: string]: string } = {
          'concat': '拼接融合',
          'mlp': 'MLP融合',
          'attention_fusion': '注意力融合'
        };
        parts.push(fusionNames[config.fusion_method] || config.fusion_method);
      }

      // MoE部分
      if (config.use_moe) {
        parts.push('MoE');
      }

      // 如果没有选择任何特殊配置，显示基础优化模型
      if (parts.length === 0) {
        return {
          baseType: '优化模型',
          config: '基础配置'
        };
      }

      return {
        baseType: '优化模型',
        config: parts.join(' + ')
      };
    }
    return {
      baseType: '未知模型类型',
      config: ''
    };
  };

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      gap: '16px',
      padding: '16px',
      background: '#f5f5f5',
      boxSizing: 'border-box'
    }}>
      {/* 第一列：配置面板 */}
      <div style={{
        width: '300px',
        background: '#fff',
        borderRadius: '12px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <ConfigPanel onSubmit={handleSubmit} />
      </div>

      {/* 第二列：训练模型 */}
      <div style={{
        width: '350px',
        background: '#fff',
        borderRadius: '12px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <TrainingPanel result={result} progress={progress} isPending={isPending} onStartTraining={handleSubmit} />
      </div>

      {/* 第三列：随机抽取ID和目标图片展示 */}
      <div style={{
        width: '400px', // 稍微减少宽度
        background: '#fff',
        borderRadius: '12px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <TargetImagePanel onSearch={handleSearch} />
      </div>

      {/* 第四列：检索结果 */}
      <div style={{
        width: 'calc(100% - 300px - 350px - 400px)', // 计算剩余宽度，让右侧更宽
        minWidth: '500px', // 设置最小宽度
        background: '#fff',
        borderRadius: '12px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <Spin spinning={isPending || isSearching}>
          <ResultPanel searchResults={searchResults} />
        </Spin>
      </div>
    </div>
  );
}


