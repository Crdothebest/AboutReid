import { useState } from 'react';
import { Layout, Spin, message } from 'antd';
import { reidRankQuery } from '../api/reid';
import { useConfigStore } from '../store/config';
import ResultPanel from '../components/ResultPanel';
import ConfigPanel from '../components/ConfigPanel';

const { Sider, Content } = Layout;

export default function Home() {
  const { target, toQueryConfig } = useConfigStore();
  const [, setResult] = useState<any>();
  const [isPending, setIsPending] = useState(false);

  const handleSubmit = async () => {
    const cfg = toQueryConfig();
    if (!cfg) return message.warning('请先选择模型');
    if (!target.targetId) return message.warning('请先抽取目标 ID');
    
    setIsPending(true);
    
    try {
      // 调用后端生成合成图片的接口
      const result = await reidRankQuery({ 
        target_id: target.targetId!, 
        query_modality: 'ALL' as any, // 特殊标识，表示需要所有模态的合成结果
        config: cfg 
      });
      
      // 期望后端返回格式：{ resultImage: "图片URL" }
      setResult(result);
    } catch (error) {
      message.error('查询失败');
    } finally {
      setIsPending(false);
    }
  };

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider width={380} style={{ background: '#fff', padding: 16, overflow: 'auto' }}>
        <ConfigPanel onSubmit={handleSubmit} />
      </Sider>
      <Layout>
        <Content style={{ 
          padding: 0, 
          overflow: 'hidden',
          height: '100vh',
          width: 760 // 固定右侧宽度为760px
        }}>
          <Spin spinning={isPending}>
            <ResultPanel />
          </Spin>
        </Content>
      </Layout>
    </Layout>
  );
}


