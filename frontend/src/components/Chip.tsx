import React from 'react';
import { X } from 'phosphor-react';

interface ChipProps {
  label: string;
  selected?: boolean;
  onClick?: () => void;
  removable?: boolean;
  onRemove?: () => void;
}

export default function Chip({ label, selected = false, onClick, removable = false, onRemove }: ChipProps) {
  return (
    <button
      type="button"
      className={"filter-chips" + (selected ? ' selected' : '')}
      onClick={onClick}
      aria-pressed={selected}
    >
      <span style={{ color: selected ? '#fff' : '#615D5B' }}>{label}</span>
      {removable && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove?.();
          }}
          aria-label={`Remover ${label}`}
          style={{ background: 'transparent', border: 'none', marginLeft: 8, color: 'inherit', display: 'inline-flex' }}
        >
          <X size={16} />
        </button>
      )}
    </button>
  );
}
