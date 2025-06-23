

import { join } from 'path-browserify';
import { decode } from 'messagepack';
import type { LinePlotProps } from '@diamondlightsource/davidia';

import type { MeasurementProps, MetaData } from '../App';
import { apiScanFiles, apiSimilarScans, measurement, apiMetadata } from '../api';


export interface ScanFiles {
  first_number: number;
  last_number: number;
  file_spec: string;
}

/**
 * Fetches scan files from the specified visit path.
 *
 * @param {string} visitPath - The path of the visit to fetch scan files from.
 * @returns {Promise<ScanFiles>} A promise that resolves to the scan files data.
 * @throws Will throw an error if the fetch operation fails.
 */
const fetchScanFiles = async ( visitPath: string ): Promise<ScanFiles> => {
  try {
    console.log('Fetching scan files from ', visitPath)
    const response = await fetch(apiScanFiles, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({'path': visitPath}),
    });
    const result = await response.json() as ScanFiles;
    return result;
  } catch (error) {
    console.error('Error fetching scan files:', error);
    throw error;
  }
};

interface SimilarFiles {
  files: string[];
  scan_numbers: number[];
}

/**
 * Fetches scan files in same directory with similar properties
 *
 * @param {string} filePath - The path of the visit to fetch scan files from.
 * @returns {Promise<string[]>} A promise that resolves to the scan files data.
 * @throws Will throw an error if the fetch operation fails.
 */
const fetchSimilarFiles = async ( scanNumber: number, filePath: string, fileSpec: string ): Promise<number[]> => {
  try {
    const files = generateFileList([scanNumber], filePath, fileSpec);
    console.log('Fetching similar scan files to ', files)
    const response = await fetch(apiSimilarScans, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({'path': files[0]}),
    });
    const result = await response.json() as SimilarFiles;
    console.log('similar scans result', result)
    return result.scan_numbers;
  } catch (error) {
    console.error('Error fetching scan files:', error);
    throw error;
  }
};


/**
 * Generates a list of file paths based on the selected numbers and file specifications.
 *
 * @param {number[]} selectedNumbers - The selected numbers for generating file paths.
 * @param {string} filePath - The base file path.
 * @param {string} fileSpec - The file specification with a placeholder for the number.
 * @returns {string[]} An array of generated file paths.
 * @throws Will throw an error if the file path or specification is invalid.
 */
function generateFileList( selectedNumbers: number[], filePath: string, fileSpec: string ): string[] {
  const files: string[] = [];
  for (let i = 0; i < selectedNumbers.length; i++) {
    files.push(
      join( 
        filePath.trim(),
        fileSpec.replace('{number}', selectedNumbers[i].toString())
      )
    );
  }
  return files;
}

/**
 * Fetches metadata strings from input data files and sets inputForm.fileMetadata
 *
 * @param {Object} params - The parameters for fetching polarization pairs.
 * @param {Object} params.inputForm - The input form data.
 * @param {Object} params.setInputForm - setter for input form data.
 * @throws Will throw an error if the fetch operation fails.
 */
const fetchFileMetadata = async ({ inputForm, setInputForm }: MeasurementProps) => {
  console.log('Fetching File Metadata', inputForm);
  const { filePath, fileSpec, selectedNumbers } = inputForm
  // don't load files already looked at
  const newNumbers = selectedNumbers.filter((number) => !(number in inputForm.fileMetadata));
  if (newNumbers.length === 0) return;

  try {
    const files = generateFileList(newNumbers, filePath, fileSpec);
    console.log('Files:', files);
    const fileObj = Object.fromEntries(newNumbers.map((number, index) => [number, files[index]]))
    console.log('FileObj:', fileObj);

    const response = await fetch(apiMetadata, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({'files': fileObj}),
    });
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    const result = await response.json() as MetaData;
    setInputForm({
      ...inputForm,
      fileMetadata: {
        ...inputForm.fileMetadata,
        ...result
      }
    })
  } catch (error) {
    console.error('Error fetching metadata:', error);
    throw error;
  }
};


interface MeasuredData {
    pol_pairs: LinePlotProps[];
    average: LinePlotProps;
    table: string;
    element: string;
    field: number[];
    temperature: number;
}


/**
 * Fetches polarization pairs from the server and updates the plots and comparison data.
 *
 * @param {Object} params - The parameters for fetching polarization pairs.
 * @param {React.FormEvent} e - The form event.
 * @param {Object} params.inputForm - The input form data.
 * @param {Function} params.setPlots - The function to update the plots state.
 * @param {Object} params.comparison - The comparison data.
 * @param {Function} params.setComparison - The function to update the comparison state.
 * @throws Will throw an error if the fetch operation fails.
 */
const fetchMeasurement = async (
  e: React.FormEvent,
  {inputForm, setPlots, setTable, comparison, setComparison, simulationInput, setSimulationInput, config}: MeasurementProps,
) => {
  e.preventDefault();
  console.log('Submiting Measurement', inputForm);
  const { filePath, fileSpec, selectedNumbers, background_type } = inputForm
  // if (!validate(formData, setErrors)) return;

  try {
    const files = generateFileList(selectedNumbers, filePath, fileSpec);
    console.log('Files:', files);

    const response = await fetch(measurement, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({files: files, background_type: background_type}),
    });
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer(); 
    const data = await decode(new Uint8Array(buffer)) as MeasuredData; 
    console.log('Measurement Response:', data);
    const charges = Object.keys(config.available_dq_values[data.element] || {});
    // update plots and table
    setPlots(data.pol_pairs);
    setTable(data.table);
    setComparison({...comparison, 'experiment': data.average});
    // update simulation input form
    setSimulationInput({ 
      ...simulationInput, 
      'ion': data.element, 
      'charges': charges,
      'bFieldX': data.field[0], 
      'bFieldY': data.field[1], 
      'bFieldZ': data.field[2], 
      'temperature': data.temperature 
    });
  } catch (error) {
    console.error('Error:', error);
  }
};

export { fetchScanFiles, fetchSimilarFiles, fetchFileMetadata, fetchMeasurement };