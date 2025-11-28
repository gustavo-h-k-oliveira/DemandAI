export type ModelKey =
  | 'rf_bombom'
  | 'rf_teta_bel'
  | 'rf_topbel_tradicional'
  | 'rf_topbel_leite';

interface PredictionBase {
  year: number;
  month: number;
  campaign: number | null;
  seasonality: number | null;
}

export interface PredictionPayload extends PredictionBase {
  model_types: ModelKey[];
  productQuery?: string;
}

export interface SinglePredictionPayload extends PredictionBase {
  model_type: ModelKey;
}

export interface BatchPredictionEntry {
  model_type: ModelKey;
  prediction: number;
  details?: string;
  flag_source?: 'user' | 'predicted' | 'history';
  applied_flags?: {
    campaign: number;
    seasonality: number;
  };
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
  batch_predictions?: BatchPredictionEntry[];
}
