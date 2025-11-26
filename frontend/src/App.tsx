import { FormEvent, useState, useEffect } from 'react';
import type { JSX } from 'react';

import logo from './static/zd-logo.png';
import { CaretLeft, SquaresFour, FileArrowUp, Files, Folder, Brain } from 'phosphor-react';

import SidebarLink from './components/SidebarLink';
import RadioOptionGroup from './components/RadioOption';

interface PredictionResponse {
  prediction: number;
  model_used?: string;
  details?: string;
  history?: Array<{
    year: number;
    month: number;
    campaign: number;
    seasonality: number;
    quantity: number;
  }>;
  user_predictions?: Array<{
    timestamp?: string;
    model_type?: string;
    year?: number;
    month?: number;
    campaign?: number;
    seasonality?: number;
    prediction?: number;
  }>;
}

const DEFAULT_PAYLOAD = {
  model_type: 'rf_bombom',
  year: new Date().getFullYear(),
  month: new Date().getMonth() + 1,
  campaign: 0,
  seasonality: 0
};

const MODEL_LABELS: Record<string, string> = {
  rf_bombom: 'BOMBOM MORANGUETE 13G 160UN',
  rf_teta_bel: 'TETA BEL TRADICIONAL 50UN',
  rf_topbel_tradicional: 'TOPBEL TRADICIONAL 50UN',
  rf_topbel_leite: 'TOPBEL LEITE CONDENSADO 50UN'
};

function App(): JSX.Element {
  const [formData, setFormData] = useState(DEFAULT_PAYLOAD);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [userPredictions, setUserPredictions] = useState<PredictionResponse['user_predictions']>([]);

  useEffect(() => {
    let mounted = true;
    fetch('/predictions')
      .then((res) => {
        if (!res.ok) throw new Error("failed");
        return res.json();
      })
      .then((data) => {
        if (!mounted) return;
        setUserPredictions(data.user_predictions || []);
      })
      .catch(() => {
        if (!mounted) return;
        setUserPredictions([]);
      });
    return () => {
      mounted = false;
    };
  }, []);

  function handleChange(event: FormEvent<HTMLInputElement | HTMLSelectElement>) {
    const { name, value } = event.currentTarget;

    setFormData((prev) => ({
      ...prev,
      [name]: name === 'model_type' ? value : Number(value)
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error(`Erro ${response.status}`);
      }

      const data = (await response.json()) as PredictionResponse;
      setResult(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erro inesperado';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  const predsToShow = result?.user_predictions ?? userPredictions;

  return (
    <div className="main">
      <nav className="sidebar">
        <div className="logo">
          <img src={logo} alt="DemandAI Logo" />
          <button className='toggle-nav'>
            <CaretLeft size={24} weight='bold' color='#615D5B' />
          </button>
        </div>
        <div className="menu">
          <p>Menu</p>
          <SidebarLink
            href="/"
            label="Dashboard"
            icon={<SquaresFour size={20} color='#615D5B' />}
            active
          />
          <SidebarLink
            href="/"
            label="Arquivos"
            icon={<FileArrowUp size={20} color='#615D5B' />}
          />
          <SidebarLink
            href="/"
            label="Relatórios"
            icon={<Files size={20} color='#615D5B' />}
          />
          <SidebarLink
            href="/"
            label="Histórico"
            icon={<Folder size={20} color='#615D5B' />}
          />
        </div>
      </nav>
      <div className="content">
        <header>
          <h1>Previsão de Demanda</h1>
        </header>
        <form className="panel" onSubmit={handleSubmit}>
          <label className='input-form'>
            Produto
            <select
              name="model_type"
              value={formData.model_type}
              onInput={handleChange}
            >
              <option value="rf_bombom">BOMBOM MORANGUETE 13G 160UN</option>
              <option value="rf_teta_bel">TETA BEL TRADICIONAL 50UN</option>
              <option value="rf_topbel_tradicional">TOPBEL TRADICIONAL 50UN</option>
              <option value="rf_topbel_leite">TOPBEL LEITE CONDENSADO 50UN</option>
            </select>
          </label>
            <div className='input-div'>
              <label className='input-form'>
                Ano
                <input
                  type="number"
                  name="year"
                  min={2000}
                  value={formData.year}
                  onInput={handleChange}
                />
              </label>
              <label className='input-form'>
                Mês
                <input
                  type="number"
                  name="month"
                  min={1}
                  max={12}
                  value={formData.month}
                  onInput={handleChange}
                />
              </label>
            </div>
            <div className='input-div'>
              <RadioOptionGroup
                legend="Campanha"
                name="campaign"
                options={[
                  { value: 1, label: 'Sim' },
                  { value: 0, label: 'Não' }
                ]}
                value={formData.campaign}
                onInput={handleChange}
              />
              <RadioOptionGroup
                legend="Sazonalidade"
                name="seasonality"
                options={[
                  { value: 1, label: 'Sim' },
                  { value: 0, label: 'Não' }
                ]}
                value={formData.seasonality}
                onInput={handleChange}
              />
            </div>
          <button type="submit" disabled={isLoading}>
            <Brain size={20} color='white' />
            <span>{isLoading ? 'Consultando…' : 'Realizar previsão'}</span>
          </button>
        </form>
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
                  <dd>{MODEL_LABELS[result.model_used ?? ''] ?? result.model_used}</dd>
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
        <section className="panel user-predictions">
          {!error && predsToShow && predsToShow.length > 0 && (
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
                  {predsToShow.slice().reverse().map((row, idx) => (
                      <tr key={`${row.timestamp ?? 'ts'}-${idx}`}>
                        <td>{row.timestamp ?? '-'}</td>
                        <td>{(row as any).product ?? MODEL_LABELS[(row.model_type ?? '') as string] ?? row.model_type ?? '-'}</td>
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
            <button type="submit" disabled={isLoading}>
              <span>Ver mais</span>
            </button>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
