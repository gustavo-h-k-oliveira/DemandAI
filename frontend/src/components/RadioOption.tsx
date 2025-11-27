import React from 'react';

export type RadioOptionValue = string | number;

export interface RadioOptionItem {
  value: RadioOptionValue;
  label: string;
  id?: string;
}

export interface RadioOptionGroupProps {
  /** Texto que aparece como legenda (colocado acima das opções) */
  legend?: string;
  /** Nome do input (name) enviado no form */
  name: string;
  /** Opções disponíveis (label + value) */
  options: RadioOptionItem[];
  /** Valor atual (controlado) */
  value: RadioOptionValue;
  /** Handler de onInput (preservar assinatura usada no projeto) */
  onInput?: (e: React.FormEvent<HTMLInputElement>) => void;
  /** Handler de onChange (opcional) */
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  /** Aplica classe extra no fieldset */
  className?: string;
  /** Render horizontal (padrão) ou vertical */
  inline?: boolean;
  /** Se true, marca o grupo como required */
  required?: boolean;
}

export default function RadioOptionGroup({
  legend,
  name,
  options,
  value,
  onInput,
  onChange,
  className,
  inline = true,
  required = false
}: RadioOptionGroupProps) {
  const handleOnChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (onChange) return onChange(e);
    if (onInput) return onInput(e as unknown as React.FormEvent<HTMLInputElement>);
    return undefined;
  };
  return (
    <fieldset className={`radio-group ${className ?? ''}`.trim()} aria-label={legend ?? name}>
      {legend && <legend className="radio-group-title">{legend}</legend>}

      <div className={`radio-options ${inline ? 'inline' : 'stacked'}`}>
        {options.map((opt, idx) => {
          const optId = opt.id ?? `${name}-${String(opt.value)}-${idx}`;
          const isChecked = String(opt.value) === String(value);
          return (
            <label key={optId} className={`radio-option ${isChecked ? 'selected' : ''}`}>
              <input
                id={optId}
                type="radio"
                name={name}
                value={String(opt.value)}
                checked={isChecked}
                onChange={handleOnChange}
                onInput={onInput}
                required={required}
              />
              <span>{opt.label}</span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}


