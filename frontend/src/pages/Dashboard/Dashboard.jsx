import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Activity, Cpu, Heart, Zap, TrendingUp, ShieldCheck,
  BarChart3, Brain, ArrowUpRight, ArrowDownRight, Server
} from 'lucide-react';
import { fetchLivenessCheck } from '../../services/api';
import './Dashboard.css';

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
  }),
};

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLivenessCheck()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false));
  }, []);

  const stats = [
    {
      icon: Cpu,
      color: 'indigo',
      value: '207',
      label: 'Traffic Sensors',
      change: 'All active',
      changeType: 'positive',
    },
    {
      icon: TrendingUp,
      color: 'emerald',
      value: '5 min',
      label: 'Forecast Horizon',
      change: 'Real-time',
      changeType: 'positive',
    },
    {
      icon: Brain,
      color: 'amber',
      value: '3',
      label: 'Congestion Classes',
      change: 'LOW · MOD · SEV',
      changeType: 'positive',
    },
    {
      icon: ShieldCheck,
      color: loading ? 'amber' : health ? 'emerald' : 'rose',
      value: loading ? '...' : health ? 'Online' : 'Offline',
      label: 'API Status',
      change: health ? `v${health.version}` : 'Unavailable',
      changeType: health ? 'positive' : 'negative',
    },
  ];

  const features = [
    {
      to: '/predict',
      icon: Activity,
      title: 'Traffic Prediction',
      desc: 'Submit a 12-timestep sensor matrix and get per-sensor speed forecasts with congestion classification.',
    },
    {
      to: '/health',
      icon: Heart,
      title: 'System Health',
      desc: 'Monitor API liveness, readiness, database, and Redis connectivity in real-time.',
    },
    {
      to: '/predict',
      icon: BarChart3,
      title: 'SHAP Explanations',
      desc: 'Understand model decisions through top-5 feature contributions for each sensor prediction.',
    },
  ];

  return (
    <div className="dashboard">
      <div className="container">
        <motion.div
          className="dashboard-header"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1>Traffic Congestion Intelligence</h1>
          <p>
            AI-powered traffic prediction system utilizing GRU neural networks and
            XGBoost classification with SHAP explainability across 207 sensors.
          </p>
        </motion.div>

        <div className="stats-grid">
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              className="stat-card"
              custom={i}
              initial="hidden"
              animate="visible"
              variants={fadeUp}
            >
              <div className={`stat-icon ${stat.color}`}>
                <stat.icon size={24} />
              </div>
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
              <span className={`stat-change ${stat.changeType}`}>
                {stat.changeType === 'positive' ? (
                  <ArrowUpRight size={12} />
                ) : (
                  <ArrowDownRight size={12} />
                )}
                {stat.change}
              </span>
            </motion.div>
          ))}
        </div>

        <motion.div
          className="info-banner"
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          <div className="info-banner-icon">
            <Zap size={26} />
          </div>
          <div>
            <h3>Powered by Deep Learning & XGBoost</h3>
            <p>
              Our pipeline uses a GRU recurrent network for speed forecasting, followed by
              XGBoost classification into LOW, MODERATE, or SEVERE congestion. SHAP values
              provide transparent, per-sensor explanations of each prediction.
            </p>
          </div>
        </motion.div>

        <div className="features-section">
          <h2>Explore Features</h2>
          <div className="features-grid">
            {features.map((feat, i) => (
              <motion.div
                key={feat.title}
                custom={i + 4}
                initial="hidden"
                animate="visible"
                variants={fadeUp}
              >
                <Link to={feat.to} className="feature-card" id={`feature-${feat.title.toLowerCase().replace(/\s+/g, '-')}`}>
                  <div className="feature-icon">
                    <feat.icon size={24} />
                  </div>
                  <div>
                    <h3>{feat.title}</h3>
                    <p>{feat.desc}</p>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
