import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import Navbar from './components/Navbar/Navbar';
import Dashboard from './pages/Dashboard/Dashboard';
import Predict from './pages/Predict/Predict';
import Health from './pages/Health/Health';
import './index.css';

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Navbar />
        <main style={{ flex: 1 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/health" element={<Health />} />
          </Routes>
        </main>
        <footer style={{
          textAlign: 'center',
          padding: 'var(--space-6) var(--space-4)',
          color: 'var(--text-tertiary)',
          fontSize: '0.8rem',
          borderTop: '1px solid var(--border-primary)',
        }}>
          TrafficPulse — AI-Powered Traffic Congestion Prediction System
        </footer>
      </BrowserRouter>
    </ThemeProvider>
  );
}
