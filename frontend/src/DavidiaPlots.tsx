import { useState } from 'react';
import ndarray from 'ndarray';
import { GlyphType, LinePlot } from '@diamondlightsource/davidia';
import type {
  NDT,
  LineParams,
  LineData,
  LinePlotProps,
} from '@diamondlightsource/davidia';
import MarkdownPreview from './MarkdownTextBox';

// import { createLineData } from '/scratch/grp66007/web/davidia/client/component/src/utils'


export function ExampleData() {
  const x = ndarray(new Float32Array([])) as NDT;
  const y = ndarray(new Float32Array([])) as NDT;
  const lineProps = {
    plotConfig: {
      xLabel: 'energy (eV)',
      yLabel: 'intensity (arb. units)',
    },
    lineData: [
      {
        key: 'squares',
        lineParams: {
          colour: 'purple',
          pointSize: 6,
          lineOn: true,
          glyphType: GlyphType.Square,
        } as LineParams,
        x,
        xDomain: [1, 10],
        y,
        yDomain: [1, 100],
        defaultIndices: false,
      } as LineData,
    ],
    xDomain: [0, 11],
    yDomain: [0, 101],
  } as LinePlotProps;
  return lineProps
}

interface DvDPlotsProps {
  lineProps: LinePlotProps;
  table?: string;
}


export function DvDPlots( { lineProps, table }: DvDPlotsProps ) {
  const [isTableVisible, setIsTableVisible] = useState(false); 
  const [visibleLines, setVisibleLines] = useState(
    Object.fromEntries(lineProps.lineData.map((line) => [line.key, true]))
  );
  console.log('visibleLines:', visibleLines);

  const toggleLineVisibility = (key: string) => {
    setVisibleLines((prev) => {
      const updatedVisibility = {
        ...prev,
        [key]: !prev[key], // Toggle visibility for the selected line
      };

      // Update yDomain based on visible lines
      const visibleLinesData = lineProps.lineData.filter((line) => updatedVisibility[line.key]);
      if (visibleLinesData.length > 0) {
        lineProps.yDomain = [
          Math.min(...visibleLinesData.flatMap((line) => Array.from(line.y.data))),
          Math.max(...visibleLinesData.flatMap((line) => Array.from(line.y.data))),
        ];
      }

      return updatedVisibility;
    });
  };

  const filteredLineData = lineProps.lineData.filter((line) => visibleLines[line.key]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', flexDirection: 'row' }}>
        {/* LinePlot */}
        <div style={{ flex: 1 }}>
          <LinePlot {...lineProps} lineData={filteredLineData} updateSelection={null} />
        </div>

        {/* Legend */}
        <div style={{ marginLeft: '20px', display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <h4>Legend</h4>
          <ul style={{ listStyleType: 'none', padding: 0 }}>
            {lineProps.lineData.map((line) => (
              <li key={line.key} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <input
                  type="checkbox"
                  checked={visibleLines[line.key]}
                  onChange={() => toggleLineVisibility(line.key)}
                  style={{ marginRight: '10px' }}
                />
                <span
                  style={{
                    display: 'inline-block',
                    width: '20px',
                    height: '10px',
                    backgroundColor: line.lineParams.colour,
                    marginRight: '10px',
                  }}
                ></span>
                {line.key}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Table Button and Content */}
      {table && (
        <div>
          <button
            onClick={() => setIsTableVisible(!isTableVisible)}
            style={{
              margin: '10px 0',
              padding: '5px 10px',
              cursor: 'pointer',
              backgroundColor: '#007bff',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
            }}
          >
            {isTableVisible ? 'Hide Table' : 'Show Table'}
          </button>
          {isTableVisible && (
            <div style={{ marginTop: '10px', border: '1px solid #ddd', padding: '10px' }}>
              <MarkdownPreview markdown={table} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};


interface XasLines {
  simulationPol1?: LineData;
  simulationPol2?: LineData;
  experimentPol1?: LineData;
  experimentPol2?: LineData;
}

interface DiffLines {
  simulation?: LineData;
  experiment?: LineData;
}

export interface PolXasFigureProps {
  xasLines?: XasLines;
  diffLines?: DiffLines;
  table?: string;
}


export function PolXasFigure( {xasLines, diffLines, table}: PolXasFigureProps ) {
  const [isTableVisible, setIsTableVisible] = useState(false); // State to toggle table visibility
  const lineProps = {
    plotConfig: {
      xLabel: 'energy (eV)',
      yLabel: 'intensity (arb. units)',
    },
    lineData: [],
  } as LinePlotProps;
  if (xasLines) {
    if (xasLines.simulationPol1) {
      lineProps.lineData.push({
        key: 'simulationPol1',
        lineParams: {
          colour: 'blue',
          pointSize: 6,
          lineOn: true,
          glyphType: GlyphType.Square,
        } as LineParams,
        x: xasLines.simulationPol1.x,
        y: xasLines.simulationPol1.y,
      } as LineData);
    }
    if (xasLines.simulationPol2) {
      lineProps.lineData.push({
        key: 'simulationPol2',
        lineParams: {
          colour: 'red',
          pointSize: 6,
          lineOn: true,
          glyphType: GlyphType.Square,
        } as LineParams,
        x: xasLines.simulationPol2.x,
        y: xasLines.simulationPol2.y,
      } as LineData);
    }
    if (xasLines.experimentPol1) {
      lineProps.lineData.push({
        key: 'experimentPol1',
        lineParams: {
          colour: 'green',
          pointSize: 6,
          lineOn: true,
          glyphType: GlyphType.Square,
        } as LineParams,
        x: xasLines.experimentPol1.x,
        y: xasLines.experimentPol1.y,
      } as LineData);
    }
    if (xasLines.experimentPol2) {
      lineProps.lineData.push({
        key: 'experimentPol2',
        lineParams: {
          colour: 'orange',
          pointSize: 6,
          lineOn: true,
          glyphType: GlyphType.Square,
        } as LineParams,
        x: xasLines.experimentPol2.x,
        y: xasLines.experimentPol2.y,
      } as LineData);
    }
  }
  if (diffLines) {
    if (diffLines.simulation) {
      lineProps.lineData.push({
        key: 'simulation',
        lineParams: {
          colour: 'blue',
          pointSize: 6,
          lineOn: true,
          glyphType: GlyphType.Square,
        } as LineParams,
        x: diffLines.simulation.x,
        y: diffLines.simulation.y,
      } as LineData);
    }
    if (diffLines.experiment) {
      lineProps.lineData.push({
        key: 'experiment',
        lineParams: {
          colour: 'red',
          pointSize: 6,
          lineOn: true,
          glyphType: GlyphType.Square,
        } as LineParams,
        x: diffLines.experiment.x,
        y: diffLines.experiment.y,
      } as LineData);
    }
  }
  console.log('lineProps:', lineProps)
  
  return (
    <>
      <LinePlot {...lineProps} updateSelection={null} />
      {table && (
        <div>
          <button
            onClick={() => setIsTableVisible(!isTableVisible)}
            style={{
              margin: '10px 0',
              padding: '5px 10px',
              cursor: 'pointer',
              backgroundColor: '#007bff',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
            }}
          >
            {isTableVisible ? 'Hide Table' : 'Show Table'}
          </button>
          {isTableVisible && (
            <div style={{ marginTop: '10px', border: '1px solid #ddd', padding: '10px' }}>
              <h3>Table</h3>
              <pre>{table}</pre>
            </div>
          )}
        </div>
      )}
    </>
  );
};