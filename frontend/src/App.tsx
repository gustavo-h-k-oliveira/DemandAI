import { FormEvent, useState } from 'react';
import type { JSX } from 'react';

import PredictionTable from './components/PredictionTable';
import PredictionForm from './components/PredictionForm';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ResultPanel from './components/ResultPanel';
import { DEFAULT_PAYLOAD } from './constants';
import type { PredictionPayload } from './types';
import { usePredictions } from './hooks/usePredictions';
import { usePredict } from './hooks/usePredict';

function App(): JSX.Element {
  const [formData, setFormData] = useState<PredictionPayload>(DEFAULT_PAYLOAD);
  const { predictions, refresh, setPredictions } = usePredictions();
  const { result, isLoading, error, predict } = usePredict();

  function handleChange(event: FormEvent<HTMLInputElement | HTMLSelectElement>) {
    const { name, value } = event.currentTarget;

    if (name === 'model_type') {
      setFormData((prev) => ({ ...prev, model_type: value as PredictionPayload['model_type'] }));
      return;
    }

    if (name === 'campaign' || name === 'seasonality') {
      setFormData((prev) => ({
        ...prev,
        [name]: value === '' ? null : Number(value)
      }));
      return;
    }

    setFormData((prev) => ({
      ...prev,
      [name]: Number(value)
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      const data = await predict(formData);
      if (data.user_predictions) {
        setPredictions(data.user_predictions);
      } else {
        refresh();
      }
    } catch (err) {
      console.error('Prediction failed', err);
    }
  }

  const predsToShow = result?.user_predictions ?? predictions;

  return (
    <div className="main">
      <Sidebar />
      <div className="content">
        <Header title="Previsão de Demanda" />
        <PredictionForm
          formData={formData}
          onInput={handleChange}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
        <ResultPanel result={result} error={error} />
        <section className="panel user-predictions">
          {!error && <PredictionTable preds={predsToShow ?? []} />}
        </section>
      </div>
    </div>
  );
}

export default App;
