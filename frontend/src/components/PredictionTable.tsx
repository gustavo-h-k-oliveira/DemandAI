import type { UserPrediction } from '../types';
import { MODEL_LABELS } from '../constants';

interface PredictionTableProps {
  preds: UserPrediction[] | null;
}

export default function PredictionTable({ preds }: PredictionTableProps) {
  const empty = !preds || preds.length === 0;

  if (empty) {
    return (
      <div className="history" aria-live="polite">
        <h2>Últimas predições realizadas</h2>
        <p className="error">Nenhuma predição disponível no momento.</p>
      </div>
    );
  }

  return (
    <>
      <div className="history">
        <h2>Últimas predições realizadas</h2>
        <table>
          <thead>
            <tr>
              <th>Data/Hora (UTC)</th>
              <th>Produto</th>
              <th>Ano</th>
              <th>Mês</th>
              <th>Campanha</th>
              <th>Sazonalidade</th>
              <th>Previsão</th>
            </tr>
          </thead>
          <tbody>
            {preds.slice().reverse().map((row, idx) => (
              <tr key={`${row.timestamp ?? 'ts'}-${idx}`}>
                <td>{row.timestamp ?? '-'}</td>
                <td>{row.product ?? (row.model_type ? MODEL_LABELS[row.model_type] ?? row.model_type : '-')}</td>
                <td>{row.year ?? '-'}</td>
                <td>{row.month ?? '-'}</td>
                <td>{row.campaign ?? '-'}</td>
                <td>{row.seasonality ?? '-'}</td>
                <td>
                  {typeof row.prediction === 'number'
                    ? row.prediction.toLocaleString('pt-BR')
                    : row.prediction ?? '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="history-footer">
        <a className="view-more-button" href="/predictions">Ver mais</a>
      </div>
    </>
  );
}
