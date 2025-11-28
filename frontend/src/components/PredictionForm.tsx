import React, { FormEvent } from 'react';
import { Brain, CaretUp, MagnifyingGlass, Plus, X } from 'phosphor-react';
import type { ModelKey, PredictionPayload } from '../types';
import { MODEL_LABELS, MODEL_OPTIONS } from '../constants';

interface PredictionFormProps {
  formData: PredictionPayload;
  onFieldChange: (e: FormEvent<HTMLInputElement | HTMLSelectElement>) => void;
  onProductQueryChange: (value: string) => void;
  onToggleModel: (model: ModelKey) => void;
  onRemoveModel: (model: ModelKey) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
}

export default function PredictionForm({
  formData,
  onFieldChange,
  onProductQueryChange,
  onToggleModel,
  onRemoveModel,
  onSubmit,
  isLoading
}: PredictionFormProps) {
  const productQuery = formData.productQuery ?? '';
  const normalizedQuery = productQuery.trim().toLowerCase();
  const filteredOptions = normalizedQuery.length > 0
    ? MODEL_OPTIONS.filter((opt) =>
        opt.label.toLowerCase().includes(normalizedQuery) ||
        opt.value.toLowerCase().includes(normalizedQuery)
      )
    : MODEL_OPTIONS;
  const hasMatches = filteredOptions.length > 0;
  const displayOptions = hasMatches ? filteredOptions : MODEL_OPTIONS;
  const selectedModels = formData.model_types;

  return (
    <form className="panel" onSubmit={onSubmit}>
      <div>
        <CaretUp size={16} />
        Produto
      </div>
      <section className="input-form product-selector">
        <div className="search-row">
          <MagnifyingGlass size={20} />
          <input
            className="search"
            type="text"
            value={productQuery}
            onChange={(event) => onProductQueryChange(event.currentTarget.value)}
            placeholder="Pesquisar por nome ou marca"
            aria-label="Pesquisar produto"
          />
          <button type="button" className="filter-button" disabled>
            <Plus size={16} />
            Filtro
          </button>
        </div>
        {!hasMatches && normalizedQuery.length > 0 && (
          <p className="no-results">Nenhum produto encontrado para "{productQuery}".</p>
        )}
        <div className="selected-models" aria-live="polite">
          {selectedModels.length === 0 && (
            <span className="selected-empty">Selecione um ou mais produtos.</span>
          )}
          {selectedModels.map((model) => (
            <span key={model} className="selected-chip">
              {MODEL_LABELS[model]}
              <button
                type="button"
                onClick={() => onRemoveModel(model)}
                aria-label={`Remover ${MODEL_LABELS[model]}`}
              >
                <X size={16} />
              </button>
            </span>
          ))}
        </div>
        <div className="model-options">
          {displayOptions.map((opt) => {
            const isSelected = selectedModels.includes(opt.value);
            return (
              <label
                key={opt.value}
                className={`model-option${isSelected ? ' selected' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => onToggleModel(opt.value)}
                />
                <span>{opt.label}</span>
              </label>
            );
          })}
        </div>
      </section>
      <div className='input-div'>
        <label className='input-form'>
          Ano
          <input
            type="number"
            name="year"
            min={2000}
            value={formData.year}
            onInput={onFieldChange}
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
            onInput={onFieldChange}
          />
        </label>
      </div>
      <div className='input-div'>
        <label className='input-form'>
          Campanha
          <select
            name="campaign"
            value={formData.campaign ?? ''}
            onInput={onFieldChange}
          >
            <option value="">Automático (usar previsão)</option>
            <option value="1">Sim</option>
            <option value="0">Não</option>
          </select>
        </label>
        <label className='input-form'>
          Sazonalidade
          <select
            name="seasonality"
            value={formData.seasonality ?? ''}
            onInput={onFieldChange}
          >
            <option value="">Automático (usar previsão)</option>
            <option value="1">Sim</option>
            <option value="0">Não</option>
          </select>
        </label>
      </div>
      <button type="submit" disabled={isLoading}>
        <Brain size={20} color='white' />
        <span>{isLoading ? 'Consultando…' : 'Realizar previsão'}</span>
      </button>
    </form>
  );
}
