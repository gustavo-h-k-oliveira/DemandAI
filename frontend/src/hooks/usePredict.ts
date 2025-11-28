import { useCallback, useState } from 'react';
import { postPredict } from '../api/api';
import type {
  PredictionPayload,
  PredictionResponse,
  SinglePredictionPayload,
  UserPrediction,
  BatchPredictionEntry
} from '../types';

export function usePredict() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (payload: PredictionPayload): Promise<PredictionResponse> => {
    setIsLoading(true);
    setError(null);
    try {
      if (!payload.model_types.length) {
        throw new Error('Selecione ao menos um produto para prever.');
      }

      const basePayload: Omit<SinglePredictionPayload, 'model_type'> = {
        year: payload.year,
        month: payload.month,
        campaign: payload.campaign,
        seasonality: payload.seasonality
      };

      const responses = await Promise.all(
        payload.model_types.map(async (model_type) => ({
          model_type,
          response: await postPredict({ ...basePayload, model_type })
        }))
      );

      if (!responses.length) {
        throw new Error('Nenhuma resposta recebida do serviço de previsão.');
      }

      const combinedUserPredictions = responses
        .flatMap(({ response }) => response.user_predictions ?? [])
        .reduce<Map<string, UserPrediction>>((acc, prediction) => {
          const key = [
            prediction.timestamp ?? 'na',
            prediction.model_type ?? 'unknown',
            prediction.year ?? 'na',
            prediction.month ?? 'na',
            prediction.prediction ?? 'na'
          ].join('|');
          acc.set(key, prediction);
          return acc;
        }, new Map());

      const batchPredictions: BatchPredictionEntry[] = responses.map(({ model_type, response }) => ({
        model_type,
        prediction: response.prediction,
        details: response.details,
        flag_source: response.flag_source,
        applied_flags: response.applied_flags
      }));

      const latestResponse = responses[responses.length - 1]?.response ?? responses[0].response;

      const aggregated: PredictionResponse = {
        ...latestResponse,
        batch_predictions: batchPredictions,
        user_predictions: Array.from(combinedUserPredictions.values())
      };

      setResult(aggregated);
      return aggregated;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erro inesperado';
      setError(message);
      setResult(null);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, isLoading, error, predict, reset };
}
