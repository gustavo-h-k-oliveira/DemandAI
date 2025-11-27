import type { PredictionResponse } from '../types';
import { MODEL_LABELS } from '../constants';

interface ResultPanelProps {
  result: PredictionResponse | null;
  error: string | null;
}

export default function ResultPanel({ result, error }: ResultPanelProps) {
  const flagSourceLabels: Record<string, string> = {
    user: 'Informado pelo usuário',
    predicted: 'Previsão automática',
    history: 'Derivado do histórico'
  };

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
          {result.applied_flags && (
            <>
              <div>
                <dt>Campanha aplicada</dt>
                <dd>{result.applied_flags.campaign ? 'Sim' : 'Não'}</dd>
              </div>
              <div>
                <dt>Sazonalidade prevista</dt>
                <dd>{result.applied_flags.seasonality ? 'Sim' : 'Não'}</dd>
              </div>
            </>
          )}
          {result.details && (
            <div>
              <dt>Detalhes</dt>
              <dd>{result.details}</dd>
            </div>
          )}
          {result.flag_source && (
            <div>
              <dt>Origem dos parâmetros</dt>
              <dd>{flagSourceLabels[result.flag_source] ?? result.flag_source}</dd>
            </div>
          )}
        </dl>
      )}

      {!error && !result && <p>Envie o formulário para ver a previsão.</p>}
    </section>
  );
}
