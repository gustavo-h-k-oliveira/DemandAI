import { useCallback, useState } from 'react';
import { postPredict } from '../api/api';
import type { PredictionPayload, PredictionResponse } from '../types';

export function usePredict() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (payload: PredictionPayload): Promise<PredictionResponse> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await postPredict(payload);
      setResult(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erro inesperado';
      setError(message);
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
