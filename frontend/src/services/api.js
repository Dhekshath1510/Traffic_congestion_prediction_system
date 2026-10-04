import axios from 'axios';

const API_BASE = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ─── Health ────────────────────────────────────────────────────────────
export async function fetchLivenessCheck() {
  const { data } = await api.get('/health/live');
  return data;
}

export async function fetchReadinessCheck() {
  const { data } = await api.get('/health/ready');
  return data;
}

// ─── Predictions ───────────────────────────────────────────────────────
export async function fetchPrediction(sequence, timestamp) {
  const { data } = await api.post('/predictions', { sequence, timestamp });
  return data;
}

export default api;
