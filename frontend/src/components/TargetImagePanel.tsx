import { Button, Card, Space, Typography, message, Image, Row, Col } from 'antd';
import { getRandomTargetId } from '../api/reid';
import { useConfigStore } from '../store/config';

const { Title, Text } = Typography;

interface TargetImagePanelProps {
    onSearch?: () => void;
}

export function TargetImagePanel({ onSearch }: TargetImagePanelProps) {
    const { target, setTarget } = useConfigStore();

    const handleRandom = async () => {
        try {
            const data = await getRandomTargetId();
            setTarget({ targetId: data.target_id, images: data.images });
        } catch (e) {
            message.error('获取随机目标失败');
        }
    };

    return (
        <Card
            title={<Title level={4} style={{ margin: 0, color: '#000000' }}>🎯 目标图片</Title>}
            styles={{
                body: {
                    padding: '20px',
                    height: 'calc(100vh - 80px)',
                    overflow: 'auto'
                },
                header: {
                    background: 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)',
                    borderBottom: 'none'
                }
            }}
            style={{ height: '100%', border: 'none' }}
        >
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
                <Space>
                    <Button
                        type="primary"
                        onClick={handleRandom}
                        size="large"
                        style={{
                            height: '40px',
                            fontSize: '14px',
                            fontWeight: '600',
                            background: 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)',
                            border: 'none',
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(82, 196, 26, 0.4)'
                        }}
                    >
                        🎲 随机抽取 ID
                    </Button>
                    <Text type="secondary" style={{ fontSize: '14px', fontWeight: '500' }}>
                        当前 ID：{target.targetId || '-'}
                    </Text>
                </Space>

                {/* 三模态图片展示 */}
                <div>
                    <Text strong style={{ fontSize: '16px', color: '#262626' }}>📸 目标图片：</Text>
                    <Row gutter={20} style={{ marginTop: 16 }}>
                        <Col span={8}>
                            <div style={{
                                textAlign: 'center',
                                padding: '16px',
                                background: '#fafafa',
                                borderRadius: '12px',
                                border: '2px solid #f0f0f0',
                                height: '280px',
                                display: 'flex',
                                flexDirection: 'column',
                                justifyContent: 'space-between'
                            }}>
                                <Text type="secondary" style={{ fontSize: '14px', fontWeight: '600', color: '#ff4d4f' }}>🔴 RGB</Text>
                                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    {target.targetId && target.images?.RGB ? (
                                        <Image
                                            src={target.images.RGB}
                                            alt="RGB"
                                            style={{
                                                width: '100%',
                                                height: '200px',
                                                objectFit: 'cover',
                                                borderRadius: '8px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                                            }}
                                            fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                                        />
                                    ) : (
                                        <div style={{
                                            width: '100%',
                                            height: '200px',
                                            background: '#f5f5f5',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            border: '2px dashed #d9d9d9',
                                            borderRadius: '8px'
                                        }}>
                                            <Text type="secondary">待加载</Text>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </Col>
                        <Col span={8}>
                            <div style={{
                                textAlign: 'center',
                                padding: '16px',
                                background: '#fafafa',
                                borderRadius: '12px',
                                border: '2px solid #f0f0f0',
                                height: '280px',
                                display: 'flex',
                                flexDirection: 'column',
                                justifyContent: 'space-between'
                            }}>
                                <Text type="secondary" style={{ fontSize: '14px', fontWeight: '600', color: '#faad14' }}>🟡 NIR</Text>
                                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    {target.targetId && target.images?.NIR ? (
                                        <Image
                                            src={target.images.NIR}
                                            alt="NIR"
                                            style={{
                                                width: '100%',
                                                height: '200px',
                                                objectFit: 'cover',
                                                borderRadius: '8px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                                            }}
                                            fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                                        />
                                    ) : (
                                        <div style={{
                                            width: '100%',
                                            height: '200px',
                                            background: '#f5f5f5',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            border: '2px dashed #d9d9d9',
                                            borderRadius: '8px'
                                        }}>
                                            <Text type="secondary">待加载</Text>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </Col>
                        <Col span={8}>
                            <div style={{
                                textAlign: 'center',
                                padding: '16px',
                                background: '#fafafa',
                                borderRadius: '12px',
                                border: '2px solid #f0f0f0',
                                height: '280px',
                                display: 'flex',
                                flexDirection: 'column',
                                justifyContent: 'space-between'
                            }}>
                                <Text type="secondary" style={{ fontSize: '14px', fontWeight: '600', color: '#52c41a' }}>🟢 TI</Text>
                                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    {target.targetId && target.images?.TI ? (
                                        <Image
                                            src={target.images.TI}
                                            alt="TI"
                                            style={{
                                                width: '100%',
                                                height: '200px',
                                                objectFit: 'cover',
                                                borderRadius: '8px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                                            }}
                                            fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                                        />
                                    ) : (
                                        <div style={{
                                            width: '100%',
                                            height: '200px',
                                            background: '#f5f5f5',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            border: '2px dashed #d9d9d9',
                                            borderRadius: '8px'
                                        }}>
                                            <Text type="secondary">待加载</Text>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </Col>
                    </Row>
                </div>


                {/* 进行检索按钮 */}
                <div style={{ marginTop: '20px', textAlign: 'center' }}>
                    <Button
                        type="primary"
                        size="large"
                        onClick={() => {
                            if (onSearch) {
                                onSearch();
                            }
                        }}
                        disabled={!target.targetId}
                        style={{
                            height: '48px',
                            fontSize: '16px',
                            fontWeight: '600',
                            background: 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)',
                            border: 'none',
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(82, 196, 26, 0.4)',
                            minWidth: '200px'
                        }}
                    >
                        🔍 进行检索
                    </Button>
                    {!target.targetId && (
                        <div style={{ marginTop: '8px' }}>
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                                请先抽取目标ID
                            </Text>
                        </div>
                    )}
                </div>
            </Space>
        </Card>
    );
}

export default TargetImagePanel;
