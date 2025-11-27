import React, { FormEvent } from 'react';
import { Brain } from 'phosphor-react';
import type { PredictionPayload } from '../types';
import { MODEL_OPTIONS } from '../constants';

interface PredictionFormProps {
  formData: PredictionPayload;
  onInput: (e: FormEvent<HTMLInputElement | HTMLSelectElement>) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
}

export default function PredictionForm({ formData, onInput, onSubmit, isLoading }: PredictionFormProps) {
  return (
    <form className="panel" onSubmit={onSubmit}>
      <label className='input-form'>
        Produto
        <select
          name="model_type"
          value={formData.model_type}
          onInput={onInput}
        >
          {MODEL_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
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
            onInput={onInput}
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
            onInput={onInput}
          />
        </label>
      </div>
      <div className='input-div'>
        <label className='input-form'>
          Campanha
          <select
            name="campaign"
            value={formData.campaign ?? ''}
            onInput={onInput}
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
            onInput={onInput}
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
