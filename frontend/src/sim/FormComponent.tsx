import React, { useState } from 'react';

import type { FormErrors } from './formParameters';
import type { SimulationProps } from '../App';
import { tooltips, dq_labels } from './formParameters';
import { TextInput, NumericInput, VectorInput } from './formInputs';
import elementDescription from './elements';
import { handleSubmit } from './handleSubmit';


function SimulationInputs( props: SimulationProps) {
  console.log('SimulationProps', props);
  const [errors, setErrors] = useState<FormErrors>({});
  const formData = props.inputForm;
  const setFormData = props.setInputForm;
  const available_dq_values = props.config.available_dq_values;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    const stringOptions = ['ion', 'symmetry', 'charge', 'path']
    setFormData({
      ...formData,
      [name]: stringOptions.includes(name) ? value : parseFloat(value),  // ternary operator
    });
  };

  const handleElementChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const element = event.target.value;
    const charges = Object.keys(available_dq_values[element] || {});
    console.log('Element changed:', element, 'charges:', charges);
    setFormData({
      ...formData,
      ion: element,
      charge: '',
      symmetry: '',
      symmetries: [],
      charges: charges,
      tenDq: {'10Dq': 0.0},
    });
  };

  const handleChargeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const charge = event.target.value;
    console.log('Charge changed:', charge);
    const symmetries = Object.keys(available_dq_values[formData.ion][charge] || []);
    console.log('Charge changed:', charge, 'symmetries:', symmetries);
    setFormData({
      ...formData,
      charge: charge,
      symmetry: '',
      symmetries: symmetries,
      tenDq: {'10Dq': 0.0},
    });
  };

  const handleSymmetryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const symmetry = event.target.value;
    console.log('Symmetry changed:', symmetry);
    setFormData({
      ...formData,
      symmetry: symmetry,
      tenDq: available_dq_values[formData.ion][formData.charge][symmetry]['initial'] || {'10Dq': 0.0},
    });
  };

  return (
    <form className="form-container" onSubmit={(e) => handleSubmit(e, props, setErrors)}>
      <h2>Quanty Simulation</h2>
      <div className="form-group">
        <label title={tooltips.ion}>Element:</label>
        <select name="ion" title={tooltips.ion} value={formData.ion} onChange={handleElementChange}>
          <option value="">Select Element</option>
          {Object.keys(available_dq_values).map((element) => (
            <option key={element} title={elementDescription(element)} value={element}>
              {element}
            </option>
          ))}
        </select>
        {errors.ion && <span className="error">{errors.ion}</span>}
      </div>

      <div className="form-group">
        <label title={tooltips.charge}>Charge:</label>
        <select name="charge" title={tooltips.charge} value={formData.charge} onChange={handleChargeChange} disabled={!formData.ion}>
          <option value="">Select Charge</option>
          {formData.charges.map((charge) => (
            <option key={charge} value={charge}>
              {charge}
            </option>
          ))}
        </select>
        {errors.charge && <span className="error">{errors.charge}</span>}
      </div>
      
      <div className="form-group">
        <label title={tooltips.symmetry}>Symmetry:</label>
        <select name="symmetry" title={tooltips.symmetry} value={formData.symmetry} disabled={!formData.charge} onChange={handleSymmetryChange}>
          <option value="">Select Symmetry</option>
          {formData.symmetries.map((symmetry) => (
            <option key={symmetry} title={tooltips[symmetry]} value={symmetry}>
              {symmetry}
            </option>
          ))}
        </select>
        {errors.symmetry && <span className="error">{errors.symmetry}</span>}
      </div>

      <div className="form-group">
        <div className="tenDq-container">
          {Object.entries(formData.tenDq).map(([label, value]) => (
            <div key={label} className="tenDq-item">
              <label title={tooltips[label]}>{dq_labels[label]}:</label>
              <input
                type="number"
                name={`tenDq.${label}`}
                value={value}
                title={label}
                onChange={(e) => {
              const updatedTenDq = { ...formData.tenDq, [label]: Number(e.target.value) };
              setFormData({ ...formData, tenDq: updatedTenDq });
                }}
                disabled={!formData.symmetry}
              />
            </div>
          ))}
        </div>
        {errors.tenDq && <span className="error">{errors.tenDq}</span>}
      </div>
      <NumericInput name="beta" label="Beta" value={formData.beta} onChange={handleChange} error={errors.beta} />
      {/* <NumericInput name="tenDq" label="10Dq" value={formData.tenDq} onChange={handleChange} error={errors.tenDq} /> */}
      <VectorInput name="bField" label="Magnetic Field [T]" value={[formData.bFieldX, formData.bFieldY, formData.bFieldZ]} onChange={handleChange} error={errors.bField} />
      <VectorInput name="hField" label="Exchange Field [eV]" value={[formData.hFieldX, formData.hFieldY, formData.hFieldZ]} onChange={handleChange} error={errors.hField} />
      <NumericInput name="temperature" label="Temperature [K]" value={formData.temperature} onChange={handleChange} error={errors.temperature} />
      <TextInput name="path" label="Quanty Path" value={formData.path} onChange={handleChange} error={errors.path} />
      <button type="submit" className="submit-button">Submit</button>
    </form>
  );
};

export default SimulationInputs;