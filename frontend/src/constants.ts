import type { ModelKey, PredictionPayload } from './types';

export const MODEL_LABELS: Record<ModelKey, string> = {
  rf_bombom: 'BOMBOM MORANGUETE 13G 160UN',
  rf_teta_bel: 'TETA BEL TRADICIONAL 50UN',
  rf_topbel_tradicional: 'TOPBEL TRADICIONAL 50UN',
  rf_topbel_leite: 'TOPBEL LEITE CONDENSADO 50UN'
};

export const MODEL_OPTIONS = Object.entries(MODEL_LABELS).map(([value, label]) => ({
  value: value as ModelKey,
  label
}));

export const DEFAULT_PAYLOAD: PredictionPayload = {
  model_type: 'rf_bombom',
  year: new Date().getFullYear(),
  month: new Date().getMonth() + 1,
  campaign: null,
  seasonality: null
};
