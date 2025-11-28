import type { PredictionPayload, PredictionResponse, UserPrediction } from '../types';

export async function getPredictions(): Promise<UserPrediction[]> {
  const res = await fetch('/predictions');
  if (!res.ok) throw new Error(`Failed to fetch predictions: ${res.status}`);
  const payload = (await res.json()) as { user_predictions?: UserPrediction[] };
  return payload.user_predictions ?? [];
}

export async function postPredict(payload: PredictionPayload): Promise<PredictionResponse> {
  const { productQuery: _productQuery, ...payloadToSend } = payload;
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payloadToSend)
  });
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  return (await res.json()) as PredictionResponse;
}
