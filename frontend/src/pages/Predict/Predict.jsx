import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, Send, Shuffle, Clock, BarChart3,
  ChevronLeft, ChevronRight, Info, Layers, Zap,
} from 'lucide-react';
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  AreaChart, Area,
} from 'recharts';
import { fetchPrediction } from '../../services/api';
import './Predict.css';

const SENSOR_COUNT = 207;
const PAGE_SIZE = 20;

// Generate realistic random speeds between 10-70 mph
function generateRandomSequence() {
  return Array.from({ length: 12 }, () =>
    Array.from({ length: SENSOR_COUNT }, () =>
      parseFloat((Math.random() * 55 + 10).toFixed(2))
    )
  );
}

function CongestionIndex({ value, congestionClass }) {
  const cls = congestionClass.toLowerCase();
  return (
    <div className="ci-bar-wrapper">
      <div className="ci-bar-bg">
        <div
          className={`ci-bar-fill ${cls}`}
          style={{ width: `${Math.max(value * 100, 2)}%` }}
        />
      </div>
      <span className="ci-value">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function SHAPPanel({ explanation }) {
  if (!explanation) return null;
  const maxAbs = Math.max(...explanation.top_features.map(f => Math.abs(f.contribution)), 0.01);

  return (
    <tr>
      <td colSpan="6" style={{ padding: 0 }}>
        <div className="shap-panel">
          <h4>
            SHAP Feature Contributions — Predicted: {explanation.predicted_class}
          </h4>
          <div className="shap-feature-list">
            {explanation.top_features.map((f) => {
              const pct = (Math.abs(f.contribution) / maxAbs) * 100;
              const isPos = f.contribution >= 0;
              return (
                <div key={f.feature} className="shap-feature">
                  <span className="shap-feature-name">{f.feature}</span>
                  <div className="shap-bar-wrapper">
                    <div
                      className={`shap-bar ${isPos ? 'positive' : 'negative'}`}
                      style={{ width: `${Math.max(pct, 4)}%` }}
                    />
                  </div>
                  <span className={`shap-value ${isPos ? 'positive' : 'negative'}`}>
                    {isPos ? '+' : ''}{f.contribution.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </td>
    </tr>
  );
}

const CHART_COLORS = {
  LOW: '#10b981',
  MODERATE: '#f59e0b',
  SEVERE: '#ef4444',
};

export default function Predict() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedSensor, setExpandedSensor] = useState(null);
  const [filter, setFilter] = useState('ALL');
  const [page, setPage] = useState(0);
  const [timestamp, setTimestamp] = useState(() => {
    const now = new Date();
    return now.toISOString().slice(0, 16);
  });

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPage(0);
    setFilter('ALL');
    setExpandedSensor(null);
    try {
      const sequence = generateRandomSequence();
      const data = await fetchPrediction(sequence, new Date(timestamp).toISOString());
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  // Filtered and paginated sensors
  const filteredSensors = useMemo(() => {
    if (!result) return [];
    if (filter === 'ALL') return result.sensors;
    return result.sensors.filter(s => s.congestion_class === filter);
  }, [result, filter]);

  const totalPages = Math.ceil(filteredSensors.length / PAGE_SIZE);
  const pagedSensors = filteredSensors.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  // Summary stats
  const summary = useMemo(() => {
    if (!result) return null;
    const counts = { LOW: 0, MODERATE: 0, SEVERE: 0 };
    let totalSpeed = 0;
    let totalCI = 0;
    result.sensors.forEach(s => {
      counts[s.congestion_class]++;
      totalSpeed += s.predicted_speed;
      totalCI += s.congestion_index;
    });
    return {
      counts,
      avgSpeed: totalSpeed / result.sensors.length,
      avgCI: totalCI / result.sensors.length,
      pieData: Object.entries(counts).map(([name, value]) => ({ name, value })),
    };
  }, [result]);

  // Speed distribution for chart
  const speedDistribution = useMemo(() => {
    if (!result) return [];
    const buckets = {};
    result.sensors.forEach(s => {
      const bucket = Math.floor(s.predicted_speed / 5) * 5;
      const key = `${bucket}-${bucket + 5}`;
      buckets[key] = (buckets[key] || 0) + 1;
    });
    return Object.entries(buckets)
      .sort(([a], [b]) => parseInt(a) - parseInt(b))
      .map(([range, count]) => ({ range, count }));
  }, [result]);

  return (
    <div className="predict-page">
      <div className="container">
        <motion.div
          className="page-header"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1>
            <Activity size={28} />
            Traffic Prediction
          </h1>
          <p>
            Submit sensor data to forecast traffic congestion across 207 sensors
            with SHAP-powered explainability.
          </p>
        </motion.div>

        {/* Form */}
        <motion.div
          className="predict-form"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
        >
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="timestamp-input">Prediction Timestamp</label>
              <input
                id="timestamp-input"
                type="datetime-local"
                value={timestamp}
                onChange={(e) => setTimestamp(e.target.value)}
              />
              <div className="form-hint">Select the time for which to predict congestion</div>
            </div>
            <div className="form-group">
              <label>Sensor Configuration</label>
              <input
                type="text"
                value={`12 timesteps × ${SENSOR_COUNT} sensors`}
                readOnly
                style={{ opacity: 0.7 }}
              />
              <div className="form-hint">Historical speed sequence matrix dimensions</div>
            </div>
          </div>

          <div className="form-actions">
            <button
              className="btn-predict"
              onClick={handlePredict}
              disabled={loading}
              id="predict-btn"
            >
              {loading ? (
                <>
                  <div className="predict-spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                  Predicting...
                </>
              ) : (
                <>
                  <Send size={16} />
                  Run Prediction
                </>
              )}
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                setTimestamp(new Date().toISOString().slice(0, 16));
              }}
              id="reset-btn"
            >
              <Clock size={16} />
              Set Current Time
            </button>
          </div>
        </motion.div>

        {/* Error */}
        {error && (
          <motion.div
            className="error-card"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ marginBottom: 32 }}
          >
            <h3>⚠ Prediction Error</h3>
            <p>{error}</p>
          </motion.div>
        )}

        {/* Loading */}
        {loading && (
          <div className="predict-loading">
            <div className="predict-spinner" />
            <p>Running GRU forecasting & XGBoost classification...</p>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <motion.div
            className="results-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="results-header">
              <h2>
                <BarChart3 size={22} />
                Prediction Results
              </h2>
              <div className="results-meta">
                <div className="results-meta-item">
                  <Clock size={14} />
                  {new Date(result.timestamp).toLocaleString()}
                </div>
                <div className="results-meta-item">
                  <Zap size={14} />
                  {result.forecast_horizon}
                </div>
                <div className="results-meta-item">
                  <Layers size={14} />
                  {result.sensors.length} sensors
                </div>
              </div>
            </div>

            {/* Summary Cards */}
            <div className="summary-grid">
              {/* Congestion Distribution Pie */}
              <div className="summary-card">
                <h3>
                  <Info size={16} />
                  Congestion Distribution
                </h3>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={summary.pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={80}
                      paddingAngle={3}
                      dataKey="value"
                      stroke="none"
                    >
                      {summary.pieData.map((entry) => (
                        <Cell key={entry.name} fill={CHART_COLORS[entry.name]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: 'var(--bg-secondary)',
                        border: '1px solid var(--border-primary)',
                        borderRadius: 8,
                        fontSize: 13,
                        color: 'var(--text-primary)',
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 8 }}>
                  {Object.entries(summary.counts).map(([cls, cnt]) => (
                    <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem' }}>
                      <div style={{ width: 10, height: 10, borderRadius: '50%', background: CHART_COLORS[cls] }} />
                      <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{cls}: {cnt}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Speed Distribution Bar */}
              <div className="summary-card">
                <h3>
                  <BarChart3 size={16} />
                  Speed Distribution (mph)
                </h3>
                <ResponsiveContainer width="100%" height={230}>
                  <AreaChart data={speedDistribution}>
                    <defs>
                      <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--chart-1)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--chart-1)" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                    <XAxis dataKey="range" tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} />
                    <YAxis tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} />
                    <Tooltip
                      contentStyle={{
                        background: 'var(--bg-secondary)',
                        border: '1px solid var(--border-primary)',
                        borderRadius: 8,
                        fontSize: 13,
                        color: 'var(--text-primary)',
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="count"
                      stroke="var(--chart-1)"
                      fill="url(#speedGrad)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Avg Stats */}
              <div className="summary-card">
                <h3>
                  <Zap size={16} />
                  Averages
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 8 }}>
                  <div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>
                      Avg Predicted Speed
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                      {summary.avgSpeed.toFixed(1)} <span style={{ fontSize: '0.9rem', fontWeight: 500, color: 'var(--text-tertiary)' }}>mph</span>
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>
                      Avg Congestion Index
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                      {(summary.avgCI * 100).toFixed(1)}<span style={{ fontSize: '0.9rem', fontWeight: 500, color: 'var(--text-tertiary)' }}>%</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {Object.entries(summary.counts).map(([cls, cnt]) => (
                      <div key={cls} style={{
                        flex: 1,
                        textAlign: 'center',
                        padding: '8px 4px',
                        borderRadius: 8,
                        background: CHART_COLORS[cls] + '18',
                      }}>
                        <div style={{ fontSize: '1.3rem', fontWeight: 800, color: CHART_COLORS[cls] }}>{cnt}</div>
                        <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>{cls}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Sensor Table */}
            <div className="sensor-section">
              <h3>Sensor Predictions</h3>
              <div className="sensor-filters">
                {['ALL', 'LOW', 'MODERATE', 'SEVERE'].map(f => (
                  <button
                    key={f}
                    className={`filter-btn ${filter === f ? 'active' : ''}`}
                    onClick={() => { setFilter(f); setPage(0); }}
                    id={`filter-${f.toLowerCase()}`}
                  >
                    {f === 'ALL' ? `All (${result.sensors.length})` : `${f} (${summary.counts[f]})`}
                  </button>
                ))}
              </div>

              <div className="sensor-table-wrapper">
                <table className="sensor-table" id="sensor-results-table">
                  <thead>
                    <tr>
                      <th>Sensor</th>
                      <th>Speed (mph)</th>
                      <th>Congestion Index</th>
                      <th>Class</th>
                      <th>Probabilities</th>
                      <th>SHAP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedSensors.map((sensor) => (
                      <>
                        <tr key={sensor.sensor_index}>
                          <td style={{ fontWeight: 700, fontFamily: 'monospace' }}>
                            #{sensor.sensor_index.toString().padStart(3, '0')}
                          </td>
                          <td style={{ fontWeight: 600 }}>{sensor.predicted_speed.toFixed(2)}</td>
                          <td>
                            <CongestionIndex
                              value={sensor.congestion_index}
                              congestionClass={sensor.congestion_class}
                            />
                          </td>
                          <td>
                            <span className={`congestion-badge ${sensor.congestion_class}`}>
                              {sensor.congestion_class}
                            </span>
                          </td>
                          <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>
                            {Object.entries(sensor.probabilities).map(([cls, prob]) => (
                              <span key={cls} style={{ marginRight: 8 }}>
                                <span style={{ color: CHART_COLORS[cls], fontWeight: 700 }}>
                                  {(prob * 100).toFixed(0)}%
                                </span>
                              </span>
                            ))}
                          </td>
                          <td>
                            <button
                              className="shap-toggle"
                              onClick={() =>
                                setExpandedSensor(
                                  expandedSensor === sensor.sensor_index ? null : sensor.sensor_index
                                )
                              }
                            >
                              {expandedSensor === sensor.sensor_index ? 'Hide' : 'View'}
                            </button>
                          </td>
                        </tr>
                        {expandedSensor === sensor.sensor_index && (
                          <SHAPPanel key={`shap-${sensor.sensor_index}`} explanation={sensor.explanation} />
                        )}
                      </>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="pagination">
                  <button
                    className="page-btn"
                    onClick={() => setPage(p => p - 1)}
                    disabled={page === 0}
                  >
                    <ChevronLeft size={16} />
                  </button>
                  {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                    let pageNum;
                    if (totalPages <= 7) {
                      pageNum = i;
                    } else if (page < 4) {
                      pageNum = i;
                    } else if (page > totalPages - 5) {
                      pageNum = totalPages - 7 + i;
                    } else {
                      pageNum = page - 3 + i;
                    }
                    return (
                      <button
                        key={pageNum}
                        className={`page-btn ${page === pageNum ? 'active' : ''}`}
                        onClick={() => setPage(pageNum)}
                      >
                        {pageNum + 1}
                      </button>
                    );
                  })}
                  <button
                    className="page-btn"
                    onClick={() => setPage(p => p + 1)}
                    disabled={page >= totalPages - 1}
                  >
                    <ChevronRight size={16} />
                  </button>
                  <span className="page-info">
                    {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filteredSensors.length)} of {filteredSensors.length}
                  </span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
