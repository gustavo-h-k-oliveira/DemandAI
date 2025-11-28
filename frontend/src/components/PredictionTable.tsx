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
              <th>Data</th>
              <th>Hora</th>
              <th>Produto</th>
              <th>Ano</th>
              <th>Mês</th>
              <th>Campanha</th>
              <th>Sazonalidade</th>
              <th>Previsão</th>
            </tr>
          </thead>
          <tbody>
              {preds.slice().reverse().map((row, idx) => {
                const ts = row.timestamp;
                let dateDisplay = '-';
                let timeDisplay = '-';
                if (ts) {
                  try {
                    const d = new Date(ts);
                    if (!isNaN(d.getTime())) {
                      // Converter para fuso de Brasília e exibir sem segundos
                      dateDisplay = d.toLocaleDateString('pt-BR', { timeZone: 'America/Sao_Paulo' });
                      timeDisplay = d.toLocaleTimeString('pt-BR', {
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false,
                        timeZone: 'America/Sao_Paulo'
                      });
                    } else {
                      // fallback: show raw timestamp split if contains T
                      if (typeof ts === 'string' && ts.includes('T')) {
                        const [datePart, timePart] = ts.split('T');
                        dateDisplay = datePart;
                        // remove trailing Z and seconds if present
                        let t = (timePart || '').replace(/Z$/, '');
                        const parts = t.split(':');
                        if (parts.length >= 2) {
                          timeDisplay = `${parts[0].padStart(2, '0')}:${parts[1].padStart(2, '0')}`;
                        } else {
                          timeDisplay = t;
                        }
                      } else {
                        dateDisplay = ts as string;
                      }
                    }
                  } catch (e) {
                    dateDisplay = ts as string;
                  }
                }

                return (
                  <tr key={`${row.timestamp ?? 'ts'}-${idx}`}>
                    <td>{dateDisplay}</td>
                    <td>{timeDisplay}</td>
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
                );
              })}
          </tbody>
        </table>
      </div>
      <div className="history-footer">
        <a className="view-more-button" href="/predictions">Ver mais</a>
      </div>
    </>
  );
}
