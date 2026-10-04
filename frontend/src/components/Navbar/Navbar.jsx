import { NavLink } from 'react-router-dom';
import { Activity, BarChart3, Heart, Sun, Moon, Monitor } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import './Navbar.css';

const navItems = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/predict', label: 'Predict', icon: Activity },
  { to: '/health', label: 'System Health', icon: Heart },
];

const themeOptions = [
  { value: 'light', icon: Sun, title: 'Light' },
  { value: 'dark', icon: Moon, title: 'Dark' },
  { value: 'system', icon: Monitor, title: 'System' },
];

export default function Navbar() {
  const { preference, setTheme } = useTheme();

  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand">
          <div className="navbar-logo">
            <Activity size={22} strokeWidth={2.5} />
          </div>
          <div>
            <div className="navbar-title">TrafficPulse</div>
            <div className="navbar-subtitle">AI Congestion Prediction</div>
          </div>
        </NavLink>

        <ul className="navbar-nav">
          {navItems.map(({ to, label, icon: Icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={to === '/'}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                id={`nav-${label.toLowerCase().replace(/\s+/g, '-')}`}
              >
                <Icon size={16} />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="navbar-actions">
          <div className="theme-switcher" id="theme-switcher">
            {themeOptions.map(({ value, icon: Icon, title }) => (
              <button
                key={value}
                className={`theme-btn ${preference === value ? 'active' : ''}`}
                onClick={() => setTheme(value)}
                title={`${title} theme`}
                id={`theme-${value}`}
              >
                <Icon size={15} />
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
