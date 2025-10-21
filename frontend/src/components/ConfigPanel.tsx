import { useEffect, useMemo } from 'react';
import { Button, Card, Checkbox, Divider, Form, Radio, Select, Space, Switch, Typography, message, Image, Row, Col } from 'antd';
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

        <Form.Item label="滑动窗口" name="sliding_window">
          <Checkbox.Group>
            {selectedModel?.supports?.sliding_window?.map((s) => (
              <Checkbox key={s} value={s}>{s}</Checkbox>
            )) || (
              <Text type="secondary">请先选择模型</Text>
            )}
          </Checkbox.Group>
        </Form.Item>

        <Form.Item label="融合方式" name="fusion_method">
          <Radio.Group>
            {selectedModel?.supports?.fusion_method?.map((f) => (
              <Radio key={f} value={f}>{f}</Radio>
            )) || (
              <Text type="secondary">请先选择模型</Text>
            )}
          </Radio.Group>
        </Form.Item>

        <Form.Item label="使用 MoE" name="use_moe" valuePropName="checked">
          <Switch disabled={selectedModel?.supports?.use_moe === undefined} />
        </Form.Item>

        <Divider />

        <Space direction="vertical" size={8} style={{ width: '100%' }}>
          <Space>
            <Button onClick={handleRandom}>随机抽取 ID</Button>
            <Text type="secondary">当前 ID：{target.targetId || '-'}</Text>
          </Space>

          {/* 三模态图片展示 */}
          <div>
            <Text strong>目标图片：</Text>
            <Row gutter={16} style={{ marginTop: 8 }}>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '8px' }}>
                  <Text type="secondary">RGB</Text>
                  <div style={{ marginTop: 4 }}>
                    {target.targetId && target.images?.RGB ? (
                      <Image
                        src={target.images.RGB}
                        alt="RGB"
                        style={{ width: 128, height: 256, objectFit: 'cover', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
                        fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                      />
                    ) : (
                      <div style={{ width: 128, height: 256, background: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #d9d9d9', objectFit: 'cover', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                        <Text type="secondary">待加载</Text>
                      </div>
                    )}
                  </div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '8px' }}>
                  <Text type="secondary">NIR</Text>
                  <div style={{ marginTop: 4 }}>
                    {target.targetId && target.images?.NIR ? (
                      <Image
                        src={target.images.NIR}
                        alt="NIR"
                        style={{ width: 128, height: 256, objectFit: 'cover', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
                        fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                      />
                    ) : (
                      <div style={{ width: 128, height: 256, background: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #d9d9d9', objectFit: 'cover', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                        <Text type="secondary">待加载</Text>
                      </div>
                    )}
                  </div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '8px' }}>
                  <Text type="secondary">TI</Text>
                  <div style={{ marginTop: 4 }}>
                    {target.targetId && target.images?.TI ? (
                      <Image
                        src={target.images.TI}
                        alt="TI"
                        style={{ width: 128, height: 256, objectFit: 'cover', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
                        fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                      />
                    ) : (
                      <div style={{ width: 128, height: 256, background: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #d9d9d9', objectFit: 'cover', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                        <Text type="secondary">待加载</Text>
                      </div>
                    )}
                  </div>
                </div>
              </Col>
            </Row>
          </div>

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


