import { FormEvent, useState } from 'react';
import type { JSX } from 'react';

import PredictionTable from './components/PredictionTable';
import PredictionForm from './components/PredictionForm';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ResultPanel from './components/ResultPanel';
import { DEFAULT_PAYLOAD } from './constants';
import type { ModelKey, PredictionPayload } from './types';
import { usePredictions } from './hooks/usePredictions';
import { usePredict } from './hooks/usePredict';

function App(): JSX.Element {
  const [formData, setFormData] = useState<PredictionPayload>(DEFAULT_PAYLOAD);
  const [formMessage, setFormMessage] = useState<string | null>(null);
  const { predictions, refresh, setPredictions } = usePredictions();
  const { result, isLoading, error, predict } = usePredict();

  function handleChange(event: FormEvent<HTMLInputElement | HTMLSelectElement>) {
    const { name, value } = event.currentTarget;
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

  function handleProductQueryChange(query: string) {
    setFormData((prev) => ({ ...prev, productQuery: query }));
  }

  function handleToggleModel(model: ModelKey) {
    setFormData((prev) => {
      const isSelected = prev.model_types.includes(model);
      const nextModels = isSelected
        ? prev.model_types.filter((item) => item !== model)
        : [...prev.model_types, model];
      if (!isSelected && nextModels.length > 0) {
        setFormMessage(null);
      }
      return { ...prev, model_types: nextModels };
    });
  }

  function handleRemoveModel(model: ModelKey) {
    setFormData((prev) => ({
      ...prev,
      model_types: prev.model_types.filter((item) => item !== model)
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!formData.model_types.length) {
      setFormMessage('Selecione ao menos um produto para realizar a previsão.');
      return;
    }
    setFormMessage(null);
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
          onFieldChange={handleChange}
          onProductQueryChange={handleProductQueryChange}
          onToggleModel={handleToggleModel}
          onRemoveModel={handleRemoveModel}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
        <ResultPanel result={result} error={error ?? formMessage} />
        <section className="panel user-predictions">
          {!error && <PredictionTable preds={predsToShow ?? []} />}
        </section>
      </div>
    </div>
  );
}

export default App;
