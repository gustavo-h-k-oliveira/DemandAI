import { FormEvent, useState } from 'react';
import type { JSX } from 'react';

interface PredictionResponse {
  prediction: number;
  model_used?: string;
  details?: string;
}

const DEFAULT_PAYLOAD = {
  model_type: 'rf_bombom',
  year: new Date().getFullYear(),
  month: new Date().getMonth() + 1,
  campaign: 0,
  seasonality: 0
};

function App(): JSX.Element {
  const [formData, setFormData] = useState(DEFAULT_PAYLOAD);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

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

  return (
    <div className="page">
      <header>
        <h1>DemandAI</h1>
        <p>Simule previsões de demanda com os modelos Random Forest disponíveis.</p>
      </header>

      <form className="panel" onSubmit={handleSubmit}>
        <label>
          Modelo
          <select
            name="model_type"
            value={formData.model_type}
            onInput={handleChange}
          >
            <option value="rf_bombom">Bombom Moranguete</option>
            <option value="rf_teta_bel">Teta Bel</option>
            <option value="rf_topbel">TopBel Tradicional</option>
            <option value="rf_topbel_leite_condensado">TopBel Leite Condensado</option>
          </select>
        </label>

        <label>
          Ano
          <input
            type="number"
            name="year"
            min={2000}
            value={formData.year}
            onInput={handleChange}
          />
        </label>

        <label>
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

        <label>
          Campanha
          <input
            type="number"
            name="campaign"
            value={formData.campaign}
            onInput={handleChange}
          />
        </label>

        <label>
          Sazonalidade
          <input
            type="number"
            name="seasonality"
            value={formData.seasonality}
            onInput={handleChange}
          />
        </label>

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Consultando…' : 'Gerar previsão'}
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
                <dt>Modelo utilizado</dt>
                <dd>{result.model_used}</dd>
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
    </div>
  );
}

export default App;
