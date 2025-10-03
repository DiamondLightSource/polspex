

const { protocol, hostname, port } = window.location;
const basePort = port ? `:${port}` : '';
export const host = `${protocol}//${hostname}${basePort}`;
export const api = host + '/api';
export const apiConfig = api + '/config';
export const elements = api + '/elements';
export const dq_values = api + '/dq-values';
export const apiScanFiles = api + '/scanfiles';
export const apiSimilarScans = api + '/similar_scans';
export const simulation = api + '/simulation';
export const measurement = api + '/measurement';
export const apiMetadata = api + '/metadata';

