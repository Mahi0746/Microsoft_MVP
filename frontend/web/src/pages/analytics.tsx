import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { useRouter } from 'next/router';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function AnalyticsPage() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => { fetchStats(); }, []);

  const fetchStats = async () => {
    const token = localStorage.getItem('token');
    if (!token) return router.push('/auth/login');
    try {
      const api = await import('../utils/apiClient');
      const data = await api.default.get('/api/marketplace/stats');
      setStats(data);
    } catch (e) { console.error('Error fetching stats', e); }
    finally { setLoading(false); }
  };

  if (loading) return (<DashboardLayout><div className="min-h-screen flex items-center justify-center"><LoadingSpinner /></div></DashboardLayout>);

  return (
    <DashboardLayout>
      <h1 className="text-2xl font-bold mb-4">Analytics</h1>
      {!stats && <div className="text-sm text-gray-400">No analytics available.</div>}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="card">
            <h3 className="font-semibold">Overview</h3>
            <pre className="text-xs mt-2">{JSON.stringify(stats.overview || stats, null, 2)}</pre>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
