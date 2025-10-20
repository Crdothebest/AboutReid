export type ModelOption = {
  id: string; // model_id
  supports: {
    sliding_window?: number[]; // e.g., [4,8,16]
    fusion_method?: Array<'concat' | 'mlp' | 'attention_fusion'>;
    use_moe?: boolean; // whether toggle is meaningful
  };
};

export type GetModelsResponse = {
  models: ModelOption[];
};

export type RandomTarget = {
  target_id: string;
  images: {
    RGB?: string;
    NIR?: string;
    TI?: string;
  };
};

export type RankItem = {
  id: string; // gallery sample id
  image_url: string;
  score: number; // similarity or distance normalized
  timestamp?: string;
  camera_id?: string;
};

export type Metrics = {
  mAP?: number;
  rank1?: number;
};

export type QueryConfig = {
  model_id: string;
  sliding_window?: number;
  fusion_method?: 'concat' | 'mlp' | 'attention_fusion';
  use_moe?: boolean;
};

export type ReidQueryPayload = {
  target_id: string;
  query_modality: 'RGB' | 'NIR' | 'TI';
  config: QueryConfig;
};

export type ReidQueryResponse = {
  metrics?: Metrics;
  rank_list: RankItem[];
  // echo parameters for summary if backend provides
  echo?: ReidQueryPayload;
};


