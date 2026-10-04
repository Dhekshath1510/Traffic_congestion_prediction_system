import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const ThemeContext = createContext(null);

const THEME_KEY = 'tcp-theme-preference';

function getSystemTheme() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function resolveTheme(preference) {
  if (preference === 'system') return getSystemTheme();
  return preference;
}

export function ThemeProvider({ children }) {
  const [preference, setPreference] = useState(() => {
    const stored = localStorage.getItem(THEME_KEY);
    return stored || 'system';
  });

  const [resolvedTheme, setResolvedTheme] = useState(() => resolveTheme(preference));

  const applyTheme = useCallback((pref) => {
    const theme = resolveTheme(pref);
    setResolvedTheme(theme);
    document.documentElement.setAttribute('data-theme', theme);
  }, []);

  useEffect(() => {
    applyTheme(preference);
    localStorage.setItem(THEME_KEY, preference);
  }, [preference, applyTheme]);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      if (preference === 'system') {
        applyTheme('system');
      }
    };
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, [preference, applyTheme]);

  const setTheme = (newPref) => {
    setPreference(newPref);
  };

  return (
    <ThemeContext.Provider value={{ preference, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}
