import { useEffect, useMemo } from 'react';
import { Button, Card, Divider, Form, Radio, Select, Space, Switch, Typography, message } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { getModels, getRandomTargetId } from '../api/reid';
import { useConfigStore } from '../store/config';

const { Title, Text } = Typography;

export function ConfigPanel({ onSubmit }: { onSubmit: () => void }) {
  const [form] = Form.useForm();
  const {
    modelId,
    slidingWindow,
    fusionMethod,
    useMoe,
    queryModality,
    target,
    setModelId,
    setSlidingWindow,
    setFusionMethod,
    setUseMoe,
    setQueryModality,
    setTarget,
  } = useConfigStore();

  const { data: modelsResp, isLoading } = useQuery({ queryKey: ['models'], queryFn: getModels });

  const selectedModel = useMemo(() => {
    return modelsResp?.models?.find((m) => m.id === modelId);
  }, [modelsResp, modelId]);

  useEffect(() => {
    form.setFieldsValue({
      model_id: modelId,
      sliding_window: slidingWindow,
      fusion_method: fusionMethod,
      use_moe: useMoe,
      query_modality: queryModality,
    });
  }, [form, modelId, slidingWindow, fusionMethod, useMoe, queryModality]);

  const handleRandom = async () => {
    try {
      const data = await getRandomTargetId();
      setTarget({ targetId: data.target_id, images: data.images });
    } catch (e) {
      message.error('获取随机目标失败');
    }
  };

  return (
    <Card loading={isLoading} title={<Title level={5}>配置与输入</Title>} styles={{ body: { paddingBottom: 16 } }}>
      <Form
        form={form}
        layout="vertical"
        onValuesChange={(changed, all) => {
          if ('model_id' in changed) setModelId(all.model_id);
          if ('sliding_window' in changed) setSlidingWindow(all.sliding_window);
          if ('fusion_method' in changed) setFusionMethod(all.fusion_method);
          if ('use_moe' in changed) setUseMoe(all.use_moe);
          if ('query_modality' in changed) setQueryModality(all.query_modality);
        }}
      >
        <Form.Item label="模型选择" name="model_id" rules={[{ required: true, message: '请选择模型' }]}>
          <Select placeholder="选择模型" options={(modelsResp?.models || []).map((m) => ({ label: m.id, value: m.id }))} />
        </Form.Item>

        {selectedModel?.supports?.sliding_window && (
          <Form.Item label="滑动窗口" name="sliding_window">
            <Radio.Group>
              {selectedModel.supports.sliding_window.map((s) => (
                <Radio key={s} value={s}>{s}</Radio>
              ))}
            </Radio.Group>
          </Form.Item>
        )}

        {selectedModel?.supports?.fusion_method && (
          <Form.Item label="融合方式" name="fusion_method">
            <Radio.Group>
              {selectedModel.supports.fusion_method.map((f) => (
                <Radio key={f} value={f}>{f}</Radio>
              ))}
            </Radio.Group>
          </Form.Item>
        )}

        {selectedModel?.supports?.use_moe !== undefined && (
          <Form.Item label="使用 MoE" name="use_moe" valuePropName="checked">
            <Switch />
          </Form.Item>
        )}

        <Divider />

        <Space direction="vertical" size={8} style={{ width: '100%' }}>
          <Space>
            <Button onClick={handleRandom}>随机抽取 ID</Button>
            <Text type="secondary">当前 ID：{target.targetId || '-'}</Text>
          </Space>

          <Form.Item label="查询模态" name="query_modality" rules={[{ required: true, message: '请选择查询模态' }]}>
            <Radio.Group>
              <Radio value="RGB">RGB</Radio>
              <Radio value="NIR">NIR</Radio>
              <Radio value="TI">TI</Radio>
            </Radio.Group>
          </Form.Item>
        </Space>

        <Divider />

        <Button type="primary" onClick={onSubmit} block>
          执行查询
        </Button>
      </Form>
    </Card>
  );
}

export default ConfigPanel;


