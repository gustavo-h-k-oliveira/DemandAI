import { useCallback, useEffect, useState } from 'react';
import { getPredictions } from '../api/api';
import type { UserPrediction } from '../types';

interface UsePredictionsOptions {
  autoLoad?: boolean;
}

export function usePredictions({ autoLoad = true }: UsePredictionsOptions = {}) {
  const [predictions, setPredictions] = useState<UserPrediction[]>([]);
  const [isLoading, setIsLoading] = useState(autoLoad);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getPredictions();
      setPredictions(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erro inesperado';
      setError(message);
      setPredictions([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (autoLoad) {
      refresh();
    }
  }, [autoLoad, refresh]);

  return { predictions, isLoading, error, refresh, setPredictions };
}
