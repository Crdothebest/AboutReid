import { Card, Typography, Select, Row, Col, Button, Image, Space } from 'antd';
import { useState } from 'react';
import { useConfigStore } from '../store/config';

const { Title, Text } = Typography;

interface ResultPanelProps {
  searchResults?: any;
}

export function ResultPanel({ searchResults }: ResultPanelProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>('rank1');
  const [selectedModel, setSelectedModel] = useState<'baseline' | 'your_model'>('baseline');
  const { target, queryModality } = useConfigStore();

  return (
    <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <Card
        title={<Title level={4} style={{ margin: 0, color: '#000000' }}>📊 检索结果</Title>}
        styles={{
          header: {
            background: 'linear-gradient(135deg, #fa541c 0%, #ff7a45 100%)',
            borderBottom: 'none'
          },
          body: {
            height: 'calc(100vh - 80px)',
            padding: '20px',
            overflow: 'auto'
          }
        }}
        style={{
          height: '100%',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          border: 'none'
        }}
        >
        {/* 检索指标选择器 - 始终显示 */}
        <div style={{ marginBottom: '20px' }}>
          <Row gutter={16} align="middle">
            <Col>
              <Text strong style={{ fontSize: '16px', color: '#262626' }}>📈 选择检索指标：</Text>
              <div style={{ marginTop: '8px' }}>
                <Select
                  value={selectedMetric}
                  onChange={setSelectedMetric}
                  style={{ width: '200px' }}
                  options={[
                    { label: 'Rank-1', value: 'rank1' },
                    { label: 'Rank-5', value: 'rank5' },
                    { label: 'Rank-10', value: 'rank10' }
                  ]}
                />
              </div>
            </Col>
          </Row>
        </div>

        {/* 模型效果选择按钮 */}
        <div style={{ marginBottom: '16px' }}>
          <Text strong style={{ fontSize: '14px', color: '#262626', marginBottom: '8px', display: 'block' }}>
            🎯 选择模型效果：
          </Text>
          <Space>
            <Button
              type={selectedModel === 'baseline' ? 'primary' : 'default'}
              onClick={() => setSelectedModel('baseline')}
              style={{
                width: '120px',
                height: '40px',
                borderRadius: '8px',
                background: selectedModel === 'baseline' ? 'linear-gradient(135deg, #1890ff 0%, #40a9ff 100%)' : undefined,
                border: selectedModel === 'baseline' ? 'none' : '1px solid #d9d9d9',
                color: selectedModel === 'baseline' ? '#fff' : '#262626',
                fontWeight: '500'
              }}
            >
              📊 Baseline效果
            </Button>
            <Button
              type={selectedModel === 'your_model' ? 'primary' : 'default'}
              onClick={() => setSelectedModel('your_model')}
              style={{
                width: '120px',
                height: '40px',
                borderRadius: '8px',
                background: selectedModel === 'your_model' ? 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)' : undefined,
                border: selectedModel === 'your_model' ? 'none' : '1px solid #d9d9d9',
                color: selectedModel === 'your_model' ? '#fff' : '#262626',
                fontWeight: '500'
              }}
            >
              🚀 优化效果
            </Button>
          </Space>
        </div>

        {/* 当前检索设置显示 - 压缩为一行小字 */}
        {target.targetId && (
          <div style={{ marginBottom: '16px', textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: '12px', color: '#8c8c8c' }}>
              当前设置：ID {target.targetId} | 模态 {queryModality || 'RGB'} | 指标 {selectedMetric.toUpperCase()} | 效果 {selectedModel === 'baseline' ? 'Baseline模型' : '优化模型'}
            </Text>
          </div>
        )}

        {searchResults && target.targetId ? (
          <div>

            {/* 检索结果显示区域 */}
            <div>
              <Text strong style={{ fontSize: '16px', color: '#262626', marginBottom: '12px', display: 'block' }}>
                检索结果 (ID: {target.targetId}, 模态: {queryModality || 'RGB'}, 模型: {selectedModel === 'baseline' ? 'Baseline' : '优化模型'}, 指标: {selectedMetric.toUpperCase()})：
              </Text>
              
              {/* 显示对应的图片 */}
              <div style={{ textAlign: 'center' }}>
                <Image
                  src={`/datasets/Rank_results/${queryModality || 'RGB'}_rank-${selectedMetric.replace('rank', '')}_results/run_20251017_175911/multimodal_ranked_list_${target.targetId}_top${selectedMetric.replace('rank', '')}_${selectedModel}.png`}
                  alt={`检索结果 - ${target.targetId} - ${queryModality || 'RGB'} - ${selectedModel}`}
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                  }}
                  fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                />
              </div>
            </div>
          </div>
        ) : (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#666',
            fontSize: '18px',
            fontWeight: '500',
            textAlign: 'center'
          }}>
            <div style={{
              fontSize: '64px',
              marginBottom: '24px',
              opacity: 0.8,
              filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))'
            }}>
              📊
            </div>
            <div style={{
              fontSize: '20px',
              fontWeight: '600',
              color: '#262626',
              marginBottom: '8px'
            }}>
              检索结果展示
            </div>
            <div style={{
              fontSize: '14px',
              color: '#8c8c8c',
              lineHeight: '1.5'
            }}>
              点击"进行检索"按钮开始检索，结果将在此处显示
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

export default ResultPanel;