import { Card, Col, Descriptions, Empty, Row, Statistic, Typography } from 'antd';
import type { ReidQueryResponse } from '../api/types';

const { Title } = Typography;

export function ResultPanel({ data }: { data?: ReidQueryResponse }) {
  const metrics = data?.metrics;
  const rankList = data?.rank_list || [];

  return (
    <div>
      <Card style={{ marginBottom: 16 }}>
        <Title level={5}>配置摘要</Title>
        {data?.echo ? (
          <Descriptions size="small" column={1} bordered>
            <Descriptions.Item label="模型">{data.echo.config.model_id}</Descriptions.Item>
            {data.echo.config.sliding_window !== undefined && (
              <Descriptions.Item label="滑动窗口">{data.echo.config.sliding_window}</Descriptions.Item>
            )}
            {data.echo.config.fusion_method && (
              <Descriptions.Item label="融合方式">{data.echo.config.fusion_method}</Descriptions.Item>
            )}
            {data.echo.config.use_moe !== undefined && (
              <Descriptions.Item label="使用 MoE">{String(data.echo.config.use_moe)}</Descriptions.Item>
            )}
            <Descriptions.Item label="查询模态">{data.echo.query_modality}</Descriptions.Item>
            <Descriptions.Item label="目标 ID">{data.echo.target_id}</Descriptions.Item>
          </Descriptions>
        ) : (
          <Descriptions size="small" column={1} bordered>
            <Descriptions.Item label="提示">后端未提供 echo，将仅展示结果</Descriptions.Item>
          </Descriptions>
        )}
      </Card>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Card>
            <Statistic title="mAP" value={metrics?.mAP ?? '-'} precision={4} />
          </Card>
        </Col>
        <Col span={12}>
          <Card>
            <Statistic title="Rank-1" value={metrics?.rank1 ?? '-'} precision={4} />
          </Card>
        </Col>
      </Row>

      <Card title={<Title level={5}>Top-N 匹配结果</Title>}>
        {rankList.length === 0 ? (
          <Empty description="暂无数据" />
        ) : (
          <Row gutter={[16, 16]}>
            {rankList.map((item, idx) => (
              <Col xs={24} sm={12} md={8} lg={6} xl={4} key={item.id}>
                <Card
                  hoverable
                  cover={<img src={item.image_url} alt={item.id} style={{ objectFit: 'cover', height: 160 }} />}
                  style={idx === 0 ? { border: '2px solid #faad14' } : undefined}
                >
                  <Card.Meta
                    title={`#${idx + 1}  Score: ${item.score.toFixed(4)}`}
                    description={`${item.camera_id || ''} ${item.timestamp || ''}`}
                  />
                </Card>
              </Col>
            ))}
          </Row>
        )}
      </Card>
    </div>
  );
}

export default ResultPanel;


