import { useState } from 'react';
import { Layout, Spin, message } from 'antd';
import { useMutation } from '@tanstack/react-query';
import { reidRankQuery } from '../api/reid';
import { useConfigStore } from '../store/config';
import ResultPanel from '../components/ResultPanel';
import ConfigPanel from '../components/ConfigPanel';

const { Sider, Content } = Layout;

export default function Home() {
  const { target, queryModality, toQueryConfig } = useConfigStore();
  const [result, setResult] = useState<any>();

  const { mutate, isPending } = useMutation({
    mutationFn: reidRankQuery,
    onSuccess: (data) => setResult(data),
    onError: () => message.error('查询失败'),
  });

  const handleSubmit = () => {
    const cfg = toQueryConfig();
    if (!cfg) return message.warning('请先选择模型');
    if (!target.targetId) return message.warning('请先抽取目标 ID');
    if (!queryModality) return message.warning('请选择查询模态');
    mutate({ target_id: target.targetId, query_modality: queryModality, config: cfg });
  };

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider width={380} style={{ background: '#fff', padding: 16, overflow: 'auto' }}>
        <ConfigPanel onSubmit={handleSubmit} />
      </Sider>
      <Layout>
        <Content style={{ padding: 16, overflow: 'auto' }}>
          <Spin spinning={isPending}>
            <ResultPanel data={result} />
          </Spin>
        </Content>
      </Layout>
    </Layout>
  );
}


