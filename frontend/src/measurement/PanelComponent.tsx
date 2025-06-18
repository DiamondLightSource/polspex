// Panel component in App.tsx

import type { MeasurementProps } from '../App';
import { DvDPlots } from '../DavidiaPlots';
import MeasurementInputs from './FormComponent';
import MarkdownPreview from '../MarkdownTextBox';


export default
function MeasurementPanel(props: MeasurementProps) {
  return (
    <div className='my-window-grid'>
        <div className='my-left-panel'>
          <MeasurementInputs {...props} />
        </div>
        
        <div className='my-right-panel'>
          <h3>Measurements</h3>
          {props.plots.map((plot, i) => (
            <DvDPlots key={i} lineProps={plot} />
          ))}
          <h3>Average</h3>
          { props.comparison.experiment.lineData.length > 0 &&
            <DvDPlots lineProps={props.comparison.experiment} />
          }
          <div style={{ marginTop: '10px', border: '1px solid #ddd', padding: '10px' }}>
            <MarkdownPreview markdown={props.table ? props.table : ''} />
          </div>
        </div>
    </div>
  )
};