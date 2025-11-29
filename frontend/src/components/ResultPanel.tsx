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

  const batch = result?.batch_predictions ?? [];
  const hasBatch = batch.length > 0;

  return (
    <section className="panel result">
      <h2>Resultados</h2>
      {error && <p className="error">{error}</p>}

      {!error && result && hasBatch && (
        <div className="batch-results">
          <table className="batch-table">
            <thead>
              <tr>
                <th style={{textAlign: 'left'}}>Produto</th>
                <th style={{textAlign: 'left'}}>Previsão</th>
                <th style={{textAlign: 'center'}}>Campanha</th>
                <th style={{textAlign: 'center'}}>Sazonalidade</th>
                <th style={{textAlign: 'center'}}>Origem</th>
                <th style={{textAlign: 'left'}}>Detalhes</th>
              </tr>
            </thead>
            <tbody>
              {batch.map((entry, i) => {
                const label = MODEL_LABELS[entry.model_type] ?? entry.model_type;
                return (
                  <tr key={`${entry.model_type}-${i}`} style={{borderTop: '1px solid rgba(0,0,0,0.06)'}}>
                    <td>{label}</td>
                    <td className="prediction-value">{entry.prediction.toLocaleString('pt-BR')}</td>
                    <td style={{textAlign: 'center'}}>{entry.applied_flags ? (entry.applied_flags.campaign ? 'Sim' : 'Não') : '—'}</td>
                    <td style={{textAlign: 'center'}}>{entry.applied_flags ? (entry.applied_flags.seasonality ? 'Sim' : 'Não') : '—'}</td>
                    <td style={{textAlign: 'center'}}>{entry.flag_source ? flagSourceLabels[entry.flag_source] ?? entry.flag_source : '—'}</td>
                    <td>{entry.details ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {!error && result && !hasBatch && (
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
