import { Card, Typography } from 'antd';

const { Title } = Typography;

export function ResultPanel() {
  return (
    <div style={{ height: '100vh', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <Card 
        title={<Title level={4}>检索结果</Title>}
        style={{ 
          height: '100%',
          width: '100%',
          display: 'flex', 
          flexDirection: 'column'
        }}
        bodyStyle={{ 
          height: 'calc(100vh - 120px)', // 充满整个右侧区域
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          padding: '20px',
          background: '#fafafa',
          border: '2px dashed #d9d9d9',
          borderRadius: '8px'
        }}
      >
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#666',
          fontSize: '18px',
          fontWeight: '500'
        }}>
          <div style={{
            fontSize: '48px',
            marginBottom: '16px',
            opacity: 0.6
          }}>
            📊
          </div>
          <div>待显示：REID识别的rank排序</div>
        </div>
      </Card>
    </div>
  );
}

export default ResultPanel;


