export type ModelKey =
  | 'rf_bombom'
  | 'rf_teta_bel'
  | 'rf_topbel_tradicional'
  | 'rf_topbel_leite';

export interface PredictionPayload {
  model_type: ModelKey;
  year: number;
  month: number;
  campaign: number | null;
  seasonality: number | null;
  productQuery?: string;
}

export interface HistoryPrediction {
  year: number;
  month: number;
  campaign: number;
  seasonality: number;
  quantity: number;
}

export interface UserPrediction {
  timestamp?: string;
  model_type?: ModelKey;
  year?: number;
  month?: number;
  campaign?: number;
  seasonality?: number;
  prediction?: number;
  product?: string;
}

export interface PredictionResponse {
  prediction: number;
  model_used?: string;
  details?: string;
  history?: HistoryPrediction[];
  user_predictions?: UserPrediction[];
  flag_source?: 'user' | 'predicted' | 'history';
  applied_flags?: {
    campaign: number;
    seasonality: number;
  };
}
