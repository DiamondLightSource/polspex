
import type { FormErrors } from './formParameters';
import type { SimulationInputForm } from '../App';

export const validate = (
  formData: SimulationInputForm,
  setErrors: React.Dispatch<React.SetStateAction<FormErrors>>
): boolean => {
  const newErrors: FormErrors = {};
  if (!formData.ion) newErrors.ion = 'Ion is required';
  if (!formData.charge) newErrors.charge = 'Charge is required';
  if (!formData.symmetry) newErrors.symmetry = 'Symmetry is required';
  if (formData.beta < 0) newErrors.beta = 'Beta must be greater than 0';
  // if (formData.tenDq < 0) newErrors.tenDq = '10Dq must be greater than 0';
  if (formData.temperature <= 0) newErrors.temperature = 'Temperature must be greater than 0';

  setErrors(newErrors);
  return Object.keys(newErrors).length === 0;
};