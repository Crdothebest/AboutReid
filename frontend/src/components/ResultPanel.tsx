import { Card, Typography, Divider, Select, Row, Col } from 'antd';
import { useState } from 'react';

const { Title, Text } = Typography;

interface ResultPanelProps {
  searchResults?: any;
}

export function ResultPanel({ searchResults }: ResultPanelProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>('rank1');

  const getAccuracyValue = () => {
    if (!searchResults?.metrics) return 0;
    switch (selectedMetric) {
      case 'rank1':
        return searchResults.metrics.rank1 || 0;
      case 'rank5':
        return searchResults.metrics.rank5 || 0;
      case 'rank10':
        return searchResults.metrics.rank10 || 0;
      default:
        return 0;
    }
  };


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
        {searchResults ? (
          <div>
            {/* 检索指标选择和准确度 */}
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
                <Col>
                  <div style={{
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    borderRadius: '12px',
                    padding: '12px 16px',
                    textAlign: 'center',
                    border: 'none',
                    minWidth: '180px',
                    boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)',
                    position: 'relative',
                    overflow: 'hidden'
                  }}>
                    {/* 背景装饰 */}
                    <div style={{
                      position: 'absolute',
                      top: '-10px',
                      right: '-10px',
                      width: '40px',
                      height: '40px',
                      background: 'rgba(255, 255, 255, 0.1)',
                      borderRadius: '50%',
                      opacity: 0.6
                    }} />
                    <div style={{
                      position: 'absolute',
                      bottom: '-5px',
                      left: '-5px',
                      width: '20px',
                      height: '20px',
                      background: 'rgba(255, 255, 255, 0.1)',
                      borderRadius: '50%',
                      opacity: 0.4
                    }} />

                    <div style={{
                      fontSize: '16px',
                      fontWeight: '700',
                      color: '#ffffff',
                      marginBottom: '2px',
                      textShadow: '0 1px 2px rgba(0,0,0,0.1)'
                    }}>
                      {(getAccuracyValue() * 100).toFixed(1)}%
                    </div>
                    <div style={{
                      fontSize: '10px',
                      color: 'rgba(255, 255, 255, 0.8)',
                      fontWeight: '500',
                      letterSpacing: '0.5px'
                    }}>
                      {selectedMetric.toUpperCase()} 准确率
                    </div>
                  </div>
                </Col>
              </Row>
            </div>

            <Divider />


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