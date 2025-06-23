import React from 'react';
import { useEffect } from 'react';

import type { MeasurementProps } from '../App';
import type { ScanFiles } from './getData';
import { fetchScanFiles, fetchMeasurement } from './getData';
import NumberRangeSelector from './NumberRangeSelector';


function MeasurementInputs( measurementProps: MeasurementProps ) {
  const { inputForm, setInputForm, config } = measurementProps;
  const { filePath, selectedInstrument, instruments, selectedVisit, visits } = inputForm;

  // load files from visit path on visitPath change
  useEffect(() => {
    const getScanFiles = async () => {
      if (!filePath) return;
      const scanFiles: ScanFiles = await fetchScanFiles(filePath);
      if (scanFiles.first_number) {
        setInputForm({
          ...inputForm, 
          fileSpec: scanFiles.file_spec,
          scanNumberRange: `${scanFiles.first_number}-${scanFiles.last_number}`
        });
      }
    };
    getScanFiles()
      .catch(console.error);
  }, [filePath]);

  // dropdown onChange handlers
  const handleInstrumentChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    // load files
    const instrument = event.target.value;
    setInputForm({...inputForm, selectedInstrument: instrument });
    const beamlineVisits = config.visits;
    console.log('Instrument: ', instrument, 'visits', beamlineVisits, instrument in beamlineVisits)
    console.log('inputForm', inputForm, inputForm.selectedInstrument)
    if (beamlineVisits && instrument in beamlineVisits) {
      console.log('visits:', Object.keys(beamlineVisits[instrument]))
      setInputForm({
        ...inputForm,
        selectedInstrument: instrument,
        visits: Object.keys(beamlineVisits[instrument]),
        selectedVisit: ''
      });
    };
  };

  const handleVisitChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedVisit = event.target.value;
    const beamlineVisits = config.visits;
    setInputForm({
      ...inputForm,
      selectedVisit: selectedVisit,
      filePath: selectedVisit ? beamlineVisits[selectedInstrument][selectedVisit] : '',
    });
  };

  const handlePathChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputForm({ ...inputForm, filePath: event.target.value });
  };

  const handleBackgroundChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value;
    setInputForm({ ...inputForm, background_type: value });
  }
  return (
    <form className="form-container" onSubmit={(e) => fetchMeasurement(e, measurementProps)}>
      <h2>Experiment Data</h2>
      {/* ---Instrument Selection--- */}
      { instruments.length > 0 &&  // only display if on /dls file system
        <div className="form-group">
          <label title='Select Instrument'>Instrument:</label>
          <select name="instrument" title='Select Instrument' value={selectedInstrument} onChange={handleInstrumentChange}>
            <option value="">Select Instrument</option>
            {instruments.map((instrument) => (
              <option key={instrument} value={instrument}>
                {instrument}
              </option>
            ))}
          </select>
        </div>
      }
      {/* ---Visit Selection--- */}
      { instruments.length > 0 &&
        <div className="form-group">
          <label title='Select Visit'>Visit:</label>
          <select name="visit" title='Select Visit' value={selectedVisit} onChange={handleVisitChange} disabled={!selectedInstrument}>
            <option value="">Select VisitID</option>
            {visits.map((visit) => (
              <option key={visit} value={visit}>
                {visit}
              </option>
            ))}
          </select>
        </div>
      }
      {/* ---Data Path & FileSpec--- */}
      <div className="form-group">
        <label title='Path'>Path:</label>
        <span>
          <input
            type="text"
            name="path"
            value={filePath}
            onChange={handlePathChange}
            title='file path of data files'
          />
        </span>
        <span>
          <label title='File Spec'>File Spec:</label>
          <input
            type="text"
            name="fileSpec"
            value={inputForm.fileSpec}
            title='file name pattern with {number} as placeholder'
            onChange={(e) => setInputForm({...inputForm, fileSpec: e.target.value})}
          />
        </span>
      </div>
      {/* ---NumberRangeSelectror.tsx--- */}
      <div className="form-group">
        <NumberRangeSelector {... measurementProps } />
      </div>
      {/* ---Background--- */}
      <div className="form-group">
        <label title='Select Background'>Background:</label>
        <select name="background" title='Select background subtraction' value={inputForm.background_type} onChange={handleBackgroundChange}>
          <option value="">Select Background</option>
          <option key="flat" value="flat">flat</option>
          <option key="linear" value="linear">linear</option>
          <option key="curve" value="curve">curved</option>
          <option key="exp" value="exp">exponential</option>
        </select>
      </div>
      <button type="submit" className="submit-button">Submit</button>
    </form>
  );
};

export default MeasurementInputs;