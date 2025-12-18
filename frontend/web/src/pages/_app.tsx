// HealthSync AI - Next.js App Component
import type { AppProps } from 'next/app';
import { useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import { useAuthStore } from '../stores/authStore';
import { AuthProvider } from '../contexts/AuthContext';
import '../styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  const { initialize } = useAuthStore();

  useEffect(() => {
    // Initialize authentication on app start
    initialize();
  }, [initialize]);

  return (
    <AuthProvider>
      <Component {...pageProps} />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e1e2e',
            color: '#e4e4e7',
            border: '1px solid #27272a',
            borderRadius: '12px',
            padding: '16px',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
          },
          success: {
            style: {
              background: '#1e1e2e',
              border: '1px solid #10b981',
              color: '#10b981',
            },
            iconTheme: {
              primary: '#10b981',
              secondary: '#1e1e2e',
            },
          },
          error: {
            style: {
              background: '#1e1e2e',
              border: '1px solid #ef4444',
              color: '#ef4444',
            },
            iconTheme: {
              primary: '#ef4444',
              secondary: '#1e1e2e',
            },
          },
        }}
      />
    </AuthProvider>
  );
}