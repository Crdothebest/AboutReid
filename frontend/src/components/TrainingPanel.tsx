import { Card, Typography, Tag, Space, Divider, Progress, Button, Select } from 'antd';
import { useState } from 'react';

const { Title, Text } = Typography;

interface TrainingPanelProps {
    result?: any;
    progress: number;
    isPending: boolean;
    onStartTraining?: () => void;
}

export function TrainingPanel({ result, progress, isPending, onStartTraining }: TrainingPanelProps) {
    const [selectedRank, setSelectedRank] = useState<'rank1' | 'rank5' | 'rank10'>('rank1');
    
    return (
        <Card
            title={<Title level={4} style={{ margin: 0, color: '#000000' }}>🚀 训练模型</Title>}
            styles={{
                body: {
                    padding: '20px',
                    height: 'calc(100vh - 80px)',
                    overflow: 'auto'
                },
                header: {
                    background: 'linear-gradient(135deg, #722ed1 0%, #9254de 100%)',
                    borderBottom: 'none'
                }
            }}
            style={{ height: '100%', border: 'none' }}
        >
            {isPending ? (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    padding: '20px',
                    textAlign: 'center'
                }}>
                    <div style={{
                        fontSize: '48px',
                        marginBottom: '24px',
                        opacity: 0.8,
                        filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))'
                    }}>
                        🔄
                    </div>
                    <div style={{
                        fontSize: '20px',
                        fontWeight: '600',
                        color: '#722ed1',
                        marginBottom: '16px'
                    }}>
                        模型训练中...
                    </div>
                    <Progress
                        percent={Math.round(progress)}
                        status={progress >= 100 ? 'success' : 'active'}
                        strokeColor={{
                            '0%': '#722ed1',
                            '100%': '#9254de',
                        }}
                        style={{ marginBottom: '16px', width: '100%' }}
                    />
                    <Text type="secondary" style={{ fontSize: '14px' }}>
                        {progress < 100 ? `训练进度: ${Math.round(progress)}%` : '训练完成！'}
                    </Text>
                </div>
            ) : result ? (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    padding: '20px',
                    textAlign: 'center'
                }}>
                    <div style={{
                        fontSize: '64px',
                        marginBottom: '24px',
                        opacity: 0.8,
                        filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))'
                    }}>
                        ✅
                    </div>
                    <div style={{
                        fontSize: '24px',
                        fontWeight: '600',
                        color: '#52c41a',
                        marginBottom: '16px'
                    }}>
                        模型训练完成！
                    </div>
                    <div style={{
                        fontSize: '16px',
                        fontWeight: '500',
                        color: '#1890ff',
                        marginBottom: '16px',
                        padding: '8px 16px',
                        background: '#e6f7ff',
                        border: '1px solid #91d5ff',
                        borderRadius: '6px'
                    }}>
                        <div style={{ marginBottom: '4px' }}>
                            📋 模型类型：{result.modelType.baseType}
                        </div>
                        <div style={{ fontSize: '14px', color: '#666' }}>
                            配置：{result.modelType.config}
                        </div>
                    </div>
                    <Divider />
                    <div style={{ textAlign: 'left', width: '100%' }}>
                        <Text strong style={{ fontSize: '16px', color: '#262626' }}>配置详情：</Text>
                        <div style={{ marginTop: '12px' }}>
                            <Space direction="vertical" size="small" style={{ width: '100%' }}>
                                <div>
                                    <Text>模型类型：</Text>
                                    <Tag color={result.config.model_id === 'baseline' ? 'blue' : 'green'}>
                                        {result.config.model_id === 'baseline' ? 'Baseline 模型' : '优化模型'}
                                    </Tag>
                                </div>
                                {result.config.sliding_window && Array.isArray(result.config.sliding_window) && result.config.sliding_window.length > 0 && (
                                    <div>
                                        <Text>滑动窗口：</Text>
                                        {result.config.sliding_window.map((size: number, index: number) => (
                                            <Tag key={index} color="orange" style={{ marginLeft: '4px' }}>
                                                {size}×{size}窗口
                                            </Tag>
                                        ))}
                                    </div>
                                )}
                                {result.config.fusion_method && (
                                    <div>
                                        <Text>融合方式：</Text>
                                        <Tag color="purple">{result.config.fusion_method}</Tag>
                                    </div>
                                )}
                                {result.config.use_moe && (
                                    <div>
                                        <Text>MoE：</Text>
                                        <Tag color="red">已启用</Tag>
                                    </div>
                                )}
                            </Space>
                        </div>
                    </div>
                    <div style={{
                        marginTop: '16px',
                        fontSize: '12px',
                        color: '#8c8c8c'
                    }}>
                        训练时间：{result.timestamp}
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
                        🚀
                    </div>
                    <div style={{
                        fontSize: '20px',
                        fontWeight: '600',
                        color: '#262626',
                        marginBottom: '16px'
                    }}>
                        准备训练模型！
                    </div>
                    <div style={{
                        fontSize: '14px',
                        color: '#8c8c8c',
                        lineHeight: '1.5',
                        marginBottom: '24px'
                    }}>
                        配置完成后点击"训练模型"开始训练
                    </div>
                    
                    <Button
                        type="primary"
                        onClick={onStartTraining}
                        size="large"
                        style={{
                            height: '48px',
                            fontSize: '16px',
                            fontWeight: '600',
                            background: 'linear-gradient(135deg, #722ed1 0%, #9254de 100%)',
                            border: 'none',
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(114, 46, 209, 0.4)'
                        }}
                    >
                        🚀 训练模型
                    </Button>
                </div>
            )}
        </Card>
    );
}
