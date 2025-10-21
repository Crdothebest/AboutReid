import { create } from 'zustand';
import type { QueryConfig } from '../api/types';

type Modality = 'RGB' | 'NIR' | 'TI';

type TargetState = {
  targetId?: string;
  images?: { RGB?: string; NIR?: string; TI?: string };
};

type ConfigState = {
  modelId?: string;
  slidingWindow?: number[]; // 改为多选
  fusionMethod?: 'concat' | 'mlp' | 'attention_fusion';
  useMoe?: boolean;
  queryModality?: Modality;
  target: TargetState;
  setModelId: (v?: string) => void;
  setSlidingWindow: (v?: number[]) => void; // 改为多选
  setFusionMethod: (v?: 'concat' | 'mlp' | 'attention_fusion') => void;
  setUseMoe: (v?: boolean) => void;
  setQueryModality: (v?: Modality) => void;
  setTarget: (t: TargetState) => void;
  toQueryConfig: () => QueryConfig | undefined;
};

export const useConfigStore = create<ConfigState>((set, get) => ({
  modelId: undefined,
  slidingWindow: undefined,
  fusionMethod: undefined,
  useMoe: false, // 默认为 false
  queryModality: undefined,
  target: {},
  setModelId: (v) => set({ modelId: v, slidingWindow: undefined, fusionMethod: undefined, useMoe: false }),
  setSlidingWindow: (v) => set({ slidingWindow: v }),
  setFusionMethod: (v) => set({ fusionMethod: v }),
  setUseMoe: (v) => set({ useMoe: v }),
  setQueryModality: (v) => set({ queryModality: v }),
  setTarget: (t) => set({ target: t }),
  toQueryConfig: () => {
    const { modelId, slidingWindow, fusionMethod, useMoe } = get();
    if (!modelId) return undefined;
    const cfg: QueryConfig = { model_id: modelId };
    if (slidingWindow !== undefined && slidingWindow.length > 0) {
      // 如果选择了多个滑动窗口，取第一个作为主要参数
      cfg.sliding_window = slidingWindow[0];
    }
    if (fusionMethod !== undefined) cfg.fusion_method = fusionMethod;
    if (useMoe !== undefined) cfg.use_moe = useMoe;
    return cfg;
  },
}));


