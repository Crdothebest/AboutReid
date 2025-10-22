import { useEffect, useState } from 'react';
import { Card, Checkbox, Divider, Form, Radio, Select, Switch, Typography, Button, Space, Row, Col, Image } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { getModels, getRandomTargetId } from '../api/reid';
import { useConfigStore } from '../store/config';

const { Title, Text } = Typography;

export function ConfigPanel({ onSubmit }: { onSubmit: () => void }) {
  const [form] = Form.useForm();
  const [target, setTarget] = useState<{ targetId?: string; images?: { RGB?: string; NIR?: string; TI?: string } }>({});
  const [selectedModality, setSelectedModality] = useState<'RGB' | 'NIR' | 'TI'>('RGB');
  
  const {
    modelId,
    slidingWindow,
    fusionMethod,
    useMoe,
    queryModality,
    setModelId,
    setSlidingWindow,
    setFusionMethod,
    setUseMoe,
    setQueryModality,
  } = useConfigStore();

  const { isLoading, error } = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    staleTime: 5 * 60 * 1000, // 5分钟内数据不会重新获取
    gcTime: 10 * 60 * 1000, // 10分钟后清除缓存
    retry: 3, // 失败时重试3次
    retryDelay: 1000, // 重试间隔1秒
  });

  // 检查 API 连接状态
  useEffect(() => {
    if (error) {
      console.error('API 请求失败:', error);
    }
  }, [error]);


  useEffect(() => {
    form.setFieldsValue({
      model_id: modelId,
      sliding_window: modelId === 'baseline' ? [] : slidingWindow,
      fusion_method: modelId === 'baseline' ? undefined : fusionMethod,
      use_moe: modelId === 'baseline' ? false : useMoe,
      query_modality: queryModality,
    });
  }, [form, modelId, slidingWindow, fusionMethod, useMoe, queryModality]);

  const handleRandom = async () => {
    try {
      const data = await getRandomTargetId();
      setTarget({ targetId: data.target_id, images: data.images });
    } catch (e) {
      console.error('获取随机目标失败', e);
    }
  };

  const handleRankQuery = () => {
    if (!target.targetId) {
      console.error('请先抽取目标图片');
      return;
    }
    if (!modelId) {
      console.error('请先选择模型');
      return;
    }
    if (!selectedModality) {
      console.error('请先选择查询模态');
      return;
    }
    
    // 设置查询模态并触发提交
    setQueryModality(selectedModality);
    onSubmit();
  };

  return (
    <Card
      title={<Title level={4} style={{ margin: 0, color: '#000000' }}>🔧 模型配置</Title>}
      styles={{
        body: {
          padding: '20px',
          height: 'calc(100vh - 80px)',
          overflow: 'auto'
        },
        header: {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderBottom: 'none'
        }
      }}
      style={{ height: '100%', border: 'none' }}
    >
      <Form
        form={form}
        layout="vertical"
        onValuesChange={(changed, all) => {
          if ('model_id' in changed) {
            setModelId(all.model_id);
            // 如果选择 Baseline 模型，清空其他选项
            if (all.model_id === 'baseline') {
              setSlidingWindow([]);
              setFusionMethod(undefined);
              setUseMoe(false);
              form.setFieldsValue({
                sliding_window: [],
                fusion_method: undefined,
                use_moe: false
              });
            } else if (all.model_id === 'optimized') {
              // 选择优化模型时，重置为默认值
              setUseMoe(false);
              form.setFieldsValue({
                use_moe: false
              });
            }
          }
          if ('sliding_window' in changed) setSlidingWindow(all.sliding_window);
          if ('fusion_method' in changed) setFusionMethod(all.fusion_method);
          if ('use_moe' in changed) setUseMoe(all.use_moe);
          if ('query_modality' in changed) setQueryModality(all.query_modality);
        }}
      >
        <Form.Item label={<span style={{ fontWeight: 'bold' }}>✨ 模型选择</span>} name="model_id" rules={[{ required: true, message: '请选择模型' }]} required={false}>
          <Select
            placeholder={isLoading ? "正在加载模型..." : "选择模型"}
            loading={isLoading}
            options={[
              { label: 'Baseline 模型', value: 'baseline' },
              { label: '优化模型', value: 'optimized' }
            ]}
            notFoundContent={isLoading ? "正在加载..." : "暂无模型数据"}
            showSearch
            filterOption={(input, option) =>
              (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
            }
          />
        </Form.Item>

        <Form.Item label={<span style={{ fontWeight: 'bold' }}>✨ 滑动窗口</span>} name="sliding_window">
          <Checkbox.Group disabled={modelId === 'baseline'} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {modelId === 'baseline' ? (
              <Text type="secondary">Baseline 模型不支持滑动窗口</Text>
            ) : modelId === 'optimized' ? (
              <>
                <Checkbox value={4} style={{ marginBottom: '8px' }}>4×4 小窗口</Checkbox>
                <Checkbox value={8} style={{ marginBottom: '8px' }}>8×8 中窗口</Checkbox>
                <Checkbox value={16}>16×16 大窗口</Checkbox>
              </>
            ) : (
              <Text type="secondary">请先选择模型</Text>
            )}
          </Checkbox.Group>
        </Form.Item>

        <Form.Item label={<span style={{ fontWeight: 'bold' }}>✨ 融合方式</span>} name="fusion_method">
          <Radio.Group disabled={modelId === 'baseline'} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {modelId === 'baseline' ? (
              <Text type="secondary">Baseline 模型使用默认融合方式</Text>
            ) : modelId === 'optimized' ? (
              <>
                <Radio value="concat" style={{ marginBottom: '8px' }}>拼接融合</Radio>
                <Radio value="mlp" style={{ marginBottom: '8px' }}>MLP 融合</Radio>
                <Radio value="attention_fusion">注意力融合</Radio>
              </>
            ) : (
              <Text type="secondary">请先选择模型</Text>
            )}
          </Radio.Group>
        </Form.Item>

        <Form.Item label={<span style={{ fontWeight: 'bold' }}>✨ 使用 MoE</span>} name="use_moe" valuePropName="checked">
          <div>
            <Switch
              disabled={modelId === 'baseline' || modelId === undefined}
              checkedChildren="启用"
              unCheckedChildren="禁用"
              onChange={(checked) => {
                setUseMoe(checked);
              }}
            />
            {modelId === 'baseline' && (
              <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
                Baseline 模型不支持 MoE
              </Text>
            )}
            {modelId === 'optimized' && (
              <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
                优化模型支持 MoE 多专家网络
              </Text>
            )}
          </div>
        </Form.Item>

        <Divider />

        <div style={{
          padding: '16px',
          background: 'linear-gradient(135deg, #f6f9fc 0%, #e9f4ff 100%)',
          borderRadius: '8px',
          border: '1px solid #d9d9d9',
          textAlign: 'center'
        }}>
          <Text type="secondary" style={{ fontSize: '14px', lineHeight: '1.5' }}>
            配置完成后，检索结果将显示在右侧"检索结果"卡片中
          </Text>
        </div>
      </Form>
    </Card>
  );
}

export default ConfigPanel;


