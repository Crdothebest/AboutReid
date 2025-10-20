import { apiClient } from './client';
import type {
  GetModelsResponse,
  RandomTarget,
  ReidQueryPayload,
  ReidQueryResponse,
} from './types';

export async function getModels() {
  const { data } = await apiClient.get<GetModelsResponse>('/get_models');
  return data;
}

export async function getRandomTargetId() {
  const { data } = await apiClient.get<RandomTarget>('/get_random_target_id');
  return data;
}

export async function reidRankQuery(payload: ReidQueryPayload) {
  const { data } = await apiClient.post<ReidQueryResponse>('/reid_rank_query', payload);
  return data;
}


