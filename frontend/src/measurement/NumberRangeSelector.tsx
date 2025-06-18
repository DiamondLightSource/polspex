import React, { useEffect, useState } from 'react';

import type { MeasurementProps } from '../App';
import { fetchFileMetadata, fetchSimilarFiles } from './getData';

// parse printer style input box for list of numbers
const parsePageRanges = (input: string): number[] => {
  const pages = new Set<number>();
  const parts = input.split(',');

  for (const part of parts) {
    const trimmed = part.trim();
    const match = trimmed.match(/^(\d+)(?:-(\d+)(?:x(\d+))?)?$/);

    if (!match) continue;

    const start = parseInt(match[1], 10);
    const end = match[2] ? parseInt(match[2], 10) : start;
    const step = match[3] ? parseInt(match[3], 10) : 1;

    if (start <= end && step > 0) {
      for (let i = start; i <= end; i += step) {
        pages.add(i);
      }
    }
  }

  return Array.from(pages).sort((a, b) => a - b);
  };


const NumberRangeSelector: React.FC<MeasurementProps> = ( measurementProps ) => {
  const maxRange = 20
  const [rangeError, setRangeError] = useState<string>('')
  const { inputForm, setInputForm } = measurementProps
  const { scanNumberRange, selectedNumbers, fileMetadata } = inputForm

  const handleRemove = (event: React.MouseEvent<HTMLButtonElement, MouseEvent>, number: number) => {
    event.preventDefault(); // Prevent default form submission
    setInputForm({
      ...inputForm,
      selectedNumbers: inputForm.selectedNumbers.filter((n) => n !== number)
    })
  };

  const removeAll = (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    event.preventDefault();
    setInputForm({ ...inputForm, selectedNumbers: [] });
  };

  const handleRangeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputForm({ ...inputForm, scanNumberRange: e.target.value})
  };

  const generateScanNumbers = async () => {
    let range = parsePageRanges(scanNumberRange)
    if ( range.length == 1 ) {
      console.log('single scan specified: ', range[0], ' find similar files')
      // fetch scan number with similar metadata
      range = await fetchSimilarFiles(range[0], inputForm.filePath, inputForm.fileSpec);
      console.log('Similar files: ', range)
      if ( range.length > 0 ) {
        setInputForm({ 
          ...inputForm, 
          scanNumberRange: `${range[0]}-${range[range.length-1]}`,
          selectedNumbers: [...new Set([...selectedNumbers, ...range])]
        })
        setRangeError('');
      } else {
        setRangeError('No files found');
      }
    } else if ( range.length > 0 && selectedNumbers.length + range.length  < maxRange ) {
      setInputForm({
        ...inputForm,
        selectedNumbers: [...new Set([...selectedNumbers, ...range])],  // remove duplicates
      });
      setRangeError('');
    } else if ( selectedNumbers.length + range.length  >= maxRange ) {
      setRangeError('Too many scans selected');
    }
  }

  const handleRangeKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault(); // Prevent form submission
      generateScanNumbers()
    }
  };
    
  const handleRangeSelect = (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    event.preventDefault();
    generateScanNumbers()
  };

  useEffect(() => {
    // update Tooltips for metadata
    fetchFileMetadata(measurementProps)
  }, [selectedNumbers]);

  return (
    <div className="number-range-selector">
      <div className="number-inputs">

        <input
          type="text"
          placeholder='e.g., 1-5,7-15x2'
          value={scanNumberRange}
          onChange={handleRangeChange}
          onKeyDown={handleRangeKeyDown}
        />
        <button type="button" onClick={(e) => handleRangeSelect(e)}>Select Range</button>
        <button type="button" onClick={(e) => removeAll(e)}>Remove All</button>
      </div>
      {rangeError && <span className="error">{rangeError}</span>}
      <div className="selected-numbers">
        {selectedNumbers.map((number) => (
          <button
            key={number}
            type="button"
            title={fileMetadata[number]}
            className="selected-number-button"
            onClick={(e) => handleRemove(e, number)}
          >
            {number} &times;
          </button>
        ))}
      </div>
    </div>
  );
};

export default NumberRangeSelector;
