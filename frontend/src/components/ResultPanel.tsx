import type { PredictionResponse } from '../types';
import { MODEL_LABELS } from '../constants';

interface ResultPanelProps {
  result: PredictionResponse | null;
  error: string | null;
}

export default function ResultPanel({ result, error }: ResultPanelProps) {
  return (
    <section className="panel result">
      <h2>Resultado</h2>
      {error && <p className="error">{error}</p>}

      {!error && result && (
        <dl>
          <div>
            <dt>Previsão</dt>
            <dd>{result.prediction.toLocaleString('pt-BR')}</dd>
          </div>
          {result.model_used && (
            <div>
              <dt>Produto</dt>
              <dd>{MODEL_LABELS[result.model_used as keyof typeof MODEL_LABELS] ?? result.model_used}</dd>
            </div>
          )}
          {result.details && (
            <div>
              <dt>Detalhes</dt>
              <dd>{result.details}</dd>
            </div>
          )}
        </dl>
      )}

      {!error && !result && <p>Envie o formulário para ver a previsão.</p>}
    </section>
  );
}
