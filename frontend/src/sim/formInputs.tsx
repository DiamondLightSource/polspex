import React from 'react';
import { tooltips } from './formParameters';

interface InputProps {
  name: string;
  label: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  error?: string;
}

interface TextInputProps extends InputProps {
  value: string;
}

interface NumericInputProps extends InputProps {
  value: number;
}

interface VectorInputProps extends InputProps {
  value: [number, number, number];
}

const TextInput: React.FC<TextInputProps> = ({ name, label, value, onChange, error }) => (
  <div className="form-group">
    <label title={tooltips[name]}>{label}:</label>
    <input
      type="text"
      name={name}
      value={value}
      onChange={onChange}
      title={tooltips[name]}
    />
    {error && <span className="error">{error}</span>}
  </div>
);

const NumericInput: React.FC<NumericInputProps> = ({ name, label, value, onChange, error }) => (
  <div className="form-group">
    <label title={tooltips[name]}>{label}:</label>
    <input
      type="number"
      name={name}
      value={value}
      onChange={onChange}
      title={tooltips[name]}
    />
    {error && <span className="error">{error}</span>}
  </div>
);


const VectorInput: React.FC<VectorInputProps> = ({ name, label, value, onChange, error }) => (
  <div className="form-group">
    <label title={tooltips[name]}>{label}:</label>
    <div  className="vector-inputs">
    <input
        type="number"
        name={name + 'X'}
        data-index="0"
        value={value[0]}
        onChange={onChange}
        title={`${label} X direction`}
    />
    <input
        type="number"
        name={name + 'Y'}
        data-index="1"
        value={value[1]}
        onChange={onChange}
        title={`${label} Y direction`}
    />
    <input
        type="number"
        name={name + 'Z'}
        data-index="2"
        value={value[2]}
        onChange={onChange}
        title={`${label} Z direction`}
    />
    </div>
    {error && <span className="error">{error}</span>}
  </div>
);

export { TextInput, NumericInput, VectorInput };