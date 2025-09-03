
import type { LineData } from '@diamondlightsource/davidia';



function lineDataToCSV(data: LineData) {
  const length = data.x.shape[0];
  let csv = 'energy [eV], signal\n';
  for (let i = 0; i < length; i++) {
    const dataX = data.x.data[i] ?? '';
    const dataY = data.y.data[i] ?? '';
    csv += `${dataX},${dataY}\n`;
  }
  return csv;
}

// Function to trigger CSV download
function downloadCSV(name: string, data: LineData) {
  const csv = lineDataToCSV(data);
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export { downloadCSV };