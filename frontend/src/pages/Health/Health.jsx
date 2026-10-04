import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Heart, Wifi, Database, Server, RefreshCw, CheckCircle, XCircle, Loader } from 'lucide-react';
import { fetchLivenessCheck, fetchReadinessCheck } from '../../services/api';
import './Health.css';

export default function Health() {
  const [liveness, setLiveness] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadHealth = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const [live, ready] = await Promise.allSettled([
        fetchLivenessCheck(),
        fetchReadinessCheck(),
      ]);
      setLiveness(live.status === 'fulfilled' ? live.value : null);
      setReadiness(ready.status === 'fulfilled' ? ready.value : null);
      if (live.status === 'rejected' && ready.status === 'rejected') {
        setError('Unable to connect to the backend API. Ensure the server is running on port 8000.');
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadHealth();
    const interval = setInterval(loadHealth, 15000);
    return () => clearInterval(interval);
  }, [loadHealth]);

  const StatusIcon = ({ status }) => {
    if (status === 'healthy') return <CheckCircle size={16} style={{ color: 'var(--status-healthy)' }} />;
    if (status === 'unhealthy') return <XCircle size={16} style={{ color: 'var(--status-danger)' }} />;
    return <Loader size={16} style={{ color: 'var(--status-warning)', animation: 'spin 1s linear infinite' }} />;
  };

  const statusClass = (s) => s === 'healthy' ? 'healthy' : s === 'unhealthy' ? 'unhealthy' : 'loading';

  return (
    <div className="health-page">
      <div className="container">
        <motion.div
          className="page-header"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1>
            <Heart size={28} />
            System Health
          </h1>
          <p>Real-time monitoring of API services and infrastructure components.</p>
        </motion.div>

        <button
          className={`refresh-btn ${refreshing ? 'spinning' : ''}`}
          onClick={loadHealth}
          disabled={refreshing}
          id="refresh-health"
        >
          <RefreshCw size={16} />
          {refreshing ? 'Checking...' : 'Refresh Status'}
        </button>

        <div className="health-grid" style={{ marginTop: '24px' }}>
          {/* Liveness Card */}
          <motion.div
            className="health-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
          >
            <div className="health-card-header">
              <div className="health-card-title">
                <div className="health-card-icon live">
                  <Wifi size={20} />
                </div>
                <h3>Liveness Probe</h3>
              </div>
              <span className={`status-badge ${loading ? 'loading' : liveness ? statusClass(liveness.status) : 'unhealthy'}`}>
                <StatusIcon status={loading ? 'loading' : liveness?.status || 'unhealthy'} />
                {loading ? 'Checking' : liveness?.status || 'Unreachable'}
              </span>
            </div>
            <div className="health-details">
              <div className="health-detail-row">
                <span className="health-detail-label">Endpoint</span>
                <span className="health-detail-value">/api/v1/health/live</span>
              </div>
              <div className="health-detail-row">
                <span className="health-detail-label">Service</span>
                <span className="health-detail-value">{liveness?.service || '—'}</span>
              </div>
              <div className="health-detail-row">
                <span className="health-detail-label">Version</span>
                <span className="health-detail-value">{liveness?.version || '—'}</span>
              </div>
            </div>
          </motion.div>

          {/* Readiness Card */}
          <motion.div
            className="health-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <div className="health-card-header">
              <div className="health-card-title">
                <div className="health-card-icon ready">
                  <Server size={20} />
                </div>
                <h3>Readiness Probe</h3>
              </div>
              <span className={`status-badge ${loading ? 'loading' : readiness ? statusClass(readiness.status) : 'unhealthy'}`}>
                <StatusIcon status={loading ? 'loading' : readiness?.status || 'unhealthy'} />
                {loading ? 'Checking' : readiness?.status || 'Unreachable'}
              </span>
            </div>
            <div className="health-details">
              <div className="health-detail-row">
                <span className="health-detail-label">Endpoint</span>
                <span className="health-detail-value">/api/v1/health/ready</span>
              </div>
              <div className="health-detail-row">
                <span className="health-detail-label">Service</span>
                <span className="health-detail-value">{readiness?.service || '—'}</span>
              </div>
              <div className="health-detail-row">
                <span className="health-detail-label">Version</span>
                <span className="health-detail-value">{readiness?.version || '—'}</span>
              </div>
            </div>

            {readiness && (
              <div className="components-section">
                <h4>Infrastructure Components</h4>
                <div className="component-item">
                  <span className="component-name">
                    <Database size={16} />
                    PostgreSQL
                  </span>
                  <span className={`status-badge ${statusClass(readiness.postgres?.status)}`}>
                    <StatusIcon status={readiness.postgres?.status} />
                    {readiness.postgres?.status}
                  </span>
                </div>
                <div className="component-item">
                  <span className="component-name">
                    <Server size={16} />
                    Redis
                  </span>
                  <span className={`status-badge ${statusClass(readiness.redis?.status)}`}>
                    <StatusIcon status={readiness.redis?.status} />
                    {readiness.redis?.status}
                  </span>
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {error && (
          <motion.div
            className="error-card"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <h3>⚠ Connection Error</h3>
            <p>{error}</p>
          </motion.div>
        )}
      </div>
    </div>
  );
}
