// Panel component in App.tsx

import type { SimulationProps } from '../App';
import { DvDPlots } from '../DavidiaPlots';
import SimulationInputs from './FormComponent';
import MarkdownPreview from '../MarkdownTextBox';


export default
function SimulationPanel(props: SimulationProps) {
  return (
    <div className='my-window-grid'>
      <div className='my-left-panel'>
        <SimulationInputs {...props} />
      </div>
      <div className='my-right-panel'>
        {props.plots.map((plot, i) => (
          <DvDPlots key={i} lineProps={plot} />
        ))}
        <div style={{ marginTop: '10px', border: '1px solid #ddd', padding: '10px' }}>
          <MarkdownPreview markdown={props.table ? props.table : ''} />
        </div>
      </div>
    </div>
  )
};