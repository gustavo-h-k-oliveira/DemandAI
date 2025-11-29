import React, { FormEvent, useState } from 'react';
import { Brain, CaretUp, MagnifyingGlass, X } from 'phosphor-react';
import Chip from './Chip';
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
  const [collapsed, setCollapsed] = useState(false);
  const BRANDS = ['BEL', 'COBERTOP', 'GOLDEN', 'DIATT', 'POP UP'];
  const CATEGORIES = [
    'MORANGUETE', 'BOMBONS', 'CANDY BAR', 'MARSHMALLOW', 'MOEDAS', 'BARRAS',
    'ZERO AÇÚCAR', 'CHIPS FORNEÁVEIS', 'CONFEITOS E PÓS', 'GOTAS', 'PARA TRANSFORMAR',
    'RECHEIOS FORNEÁVEIS', 'LÁCTEOS', 'PIPOCA'
  ];
  const [selectedBrands, setSelectedBrands] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [filtersOpen, setFiltersOpen] = useState(true);
  const normalize = (s: string) =>
    s
      .toLowerCase()
      .normalize('NFD')
      .replace(/\p{Diacritic}/gu, '')
      .replace(/\s+/g, '');
  const productQuery = formData.productQuery ?? '';
  // Use the same `normalize` function for the query so accents/spaces are handled
  const normalizedQuery = normalize(productQuery.trim());
  const baseFiltered = normalizedQuery.length > 0
    ? MODEL_OPTIONS.filter((opt) =>
        normalize(opt.label).includes(normalizedQuery) ||
        normalize(opt.value).includes(normalizedQuery)
      )
    : MODEL_OPTIONS;

  const filteredByBrandsAndCategories = baseFiltered.filter((opt) => {
    const labelNorm = normalize(opt.label);
    const matchBrand = selectedBrands.length === 0 || selectedBrands.some((b) => labelNorm.includes(normalize(b)));
    const matchCategory = selectedCategories.length === 0 || selectedCategories.some((c) => labelNorm.includes(normalize(c)));
    return matchBrand && matchCategory;
  });

  const hasMatches = filteredByBrandsAndCategories.length > 0;
  const displayOptions = hasMatches ? filteredByBrandsAndCategories : baseFiltered;
  const selectedModels = formData.model_types;

  return (
    <form className="panel collapsible" onSubmit={onSubmit}>
      <div
        className="panel-header"
        role="button"
        tabIndex={0}
        onClick={() => setCollapsed((c) => !c)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setCollapsed((c) => !c);
          }
        }}
        aria-expanded={!collapsed}
      >
        <CaretUp size={16} style={{ transform: collapsed ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s ease' }} />
        Produto
      </div>
      <div className={`panel-body ${collapsed ? 'collapsed' : ''}`}>
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
            <button
              type="button"
              className="filter-button"
              aria-expanded={filtersOpen}
              onClick={() => setFiltersOpen((b) => !b)}
            >
              <CaretUp size={16} style={{ transform: filtersOpen ? 'none' : 'rotate(180deg)', transition: 'transform 0.18s ease' }} />
              Filtro
            </button>
          </div>
          {!hasMatches && normalizedQuery.length > 0 && (
            <p className="no-results">Nenhum produto encontrado para "{productQuery}".</p>
          )}
          <div className={`filter-panel ${filtersOpen ? 'open' : 'collapsed'}`}>
            <div className={`brand-panel ${filtersOpen ? 'open' : 'collapsed'}`}>
              <span className="options-label">Marcas</span>
              <div className="selected-chips" role="list">
                {BRANDS.map((brand) => {
                  const active = selectedBrands.includes(brand);
                  return (
                    <Chip
                      key={brand}
                      label={brand}
                      selected={active}
                      onClick={() => setSelectedBrands((prev) => (prev.includes(brand) ? prev.filter((p) => p !== brand) : [...prev, brand]))}
                    />
                  );
                })}
              </div>
            </div>
            <div className={`category-panel ${filtersOpen ? 'open' : 'collapsed'}`}>
              <span className="options-label">Categorias</span>
              <div className="selected-chips" role="list">
                {CATEGORIES.map((cat) => {
                  const active = selectedCategories.includes(cat);
                  return (
                    <Chip
                      key={cat}
                      label={cat}
                      selected={active}
                      onClick={() => setSelectedCategories((prev) => (prev.includes(cat) ? prev.filter((p) => p !== cat) : [...prev, cat]))}
                    />
                  );
                })}
              </div>
            </div>
          </div>
          <span className="options-label">Produtos</span>
          <div className="selected-chips" aria-live="polite">
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
      </div>
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
      <button className='submit' type="submit" disabled={isLoading}>
        <Brain size={20} color='white' />
        <span>{isLoading ? 'Consultando…' : 'Realizar previsão'}</span>
      </button>
    </form>
  );
}
