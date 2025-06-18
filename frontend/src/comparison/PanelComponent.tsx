import { useState, useEffect } from 'react';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { GlyphType } from '@diamondlightsource/davidia';
import type { LinePlotProps, LineData, LineParams } from '@diamondlightsource/davidia';

import type { ComparisonProps } from '../App';
import MarkdownPreview from '../MarkdownTextBox';
import { DvDPlots } from '../DavidiaPlots';

export default function ComparisonPanel(comparison: ComparisonProps) {
  console.log('ComparisonPanel:', comparison);

  // defaults for line data
  const defaultLineData: LineData = {
    lineParams: {
      name: 'Default',
      colour: 'blue',
      pointSize: 0,
      lineOn: true,
      glyphType: GlyphType.Square,
    } as LineParams,
    x: ndarray(new Float32Array([0, 1, 2])),
    y: ndarray(new Float32Array([0, .1, .2])),
    key: 'Default',
    xDomain: [0, 1],
    yDomain: [0, 1],
  };
  const experimentLineData = comparison.experiment.lineData[2] || defaultLineData; // Check if lineData has at least 3 elements
  const simulationLineData = comparison.simulation.lineData[0] || defaultLineData; // Use the first line data from the simulation
  experimentLineData.key = 'Experiment'; // Set the label for the experiment line
  simulationLineData.key = 'Simulation'; // Set the label for the simulation line
  console.log('experimentLineData', experimentLineData);
  console.log('simulationLineData', simulationLineData);

  // determine approximate scale of the y-axis
  const expSum = Math.abs(ops.sum(ndarray(new Float32Array(experimentLineData.y.data))));
  const simSum = Math.abs(ops.sum(ndarray(new Float32Array(simulationLineData.y.data))));
  const scale = expSum / simSum; // Calculate the scale factor based on the sum of y-values
  console.log('expSum', expSum, 'simSum', simSum, 'scale:', scale);
  // const scale = 1.0; // Default scale factor

  // useRef (multipoint in Davidia)
  const [xOffset, setXOffset] = useState(4.0); // State to manage the x-axis offset
  const [yScale, setYScale] = useState(scale); // State to manage the y-axis scale
  const [invertY, setInvertY] = useState(false); // State to manage y-axis inversion
  const [adjustedSimulation, setAdjustedSimulation] = useState<LineData>(simulationLineData); // State for adjusted simulation data
  const [yDomain, setYDomain] = useState(experimentLineData.yDomain); // State for y-axis domain

  // create single plot
  const plot: LinePlotProps = {
    plotConfig: {
      xLabel: 'energy (eV)',
      yLabel: 'intensity (arb. units)',
    },
    lineData: [
      experimentLineData, // Check if lineData has at least 3 elements
      adjustedSimulation, // Use the first line data from the simulation
    ],
    xDomain: experimentLineData.xDomain,
    yDomain: yDomain,
  };

  // Update adjusted simulation data whenever xOffset changes
  useEffect(() => {
    const xArray = ndarray(new Float32Array(simulationLineData.x.data)); // Copy the original x-axis data
    const yArray = ndarray(new Float32Array(simulationLineData.y.data)); // Copy the original y-axis data
    const yMult = invertY ? -1 : 1; // Determine the multiplier for y-axis inversion
    ops.adds(xArray, xArray, xOffset); // Add offset to x-axis data
    ops.muls(yArray, yArray, yScale * yMult); // Scale the y-axis data

    // Calculate the minimum y-value
    let yMin = Math.min(
      ...yArray.data,
      ...experimentLineData.y.data
    ); 
    // Calculate the maximum y-value
    let yMax = Math.max(
      ...yArray.data,
      ...experimentLineData.y.data
    ); 
    // Adjust the y-axis domain based on the minimum and maximum values
    if (yMin < 0) {
      yMin = 1.1 * yMin;
    } else {
      yMin = 0.9 * yMin;
    }
    
    if (yMax < 0) {
      yMax = 0.9 * yMax;
    } else {
      yMax = 1.1 * yMax;
    } 

    setYDomain([yMin, yMax]);
    setAdjustedSimulation({
      ...simulationLineData,
      x: xArray, // Use the adjusted x-axis data
      y: yArray, // Use the adjusted y-axis data
      // key: 'Simulation', // Set the label for the simulation line
    });
  }, [xOffset, yScale, invertY, comparison.simulation]);
  return (
    <div className="my-window-grid">
      <div className="my-left-panel">
        <h3>Comparison</h3>
        <p>Adjust X-Axis Offset:</p>
        <input
          type="range"
          min={-10}
          max={10}
          step={0.1}
          value={xOffset}
          onChange={(e) => setXOffset(parseFloat(e.target.value))} // Update the offset
          style={{ width: '100%' }}
        />
        <p>Offset: {xOffset.toFixed(1)}</p>
        <p>Adjust Y-Axis Scale:</p>
        <input
          type="range"
          min={0.01}
          max={2*scale}
          step={0.01}
          value={yScale}
          onChange={(e) => setYScale(parseFloat(e.target.value))} // Update the scale
          style={{ width: '100%' }}
        />
        <p>Scale: {yScale.toFixed(2)}</p>
        <div>
          <label>
            <input
              type="checkbox"
              checked={invertY}
              onChange={(e) => setInvertY(e.target.checked)} // Toggle y-axis inversion
            />
            Invert simulation
          </label>
        </div>
      </div>
      <div className="my-right-panel">
        <DvDPlots lineProps={plot} />
        <MarkdownPreview markdown={comparison.table ? comparison.table : ''} />
      </div>
    </div>
  );
}