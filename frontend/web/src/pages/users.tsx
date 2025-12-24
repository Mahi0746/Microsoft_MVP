import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { useRouter } from 'next/router';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function UsersPage() {
  const [doctors, setDoctors] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => { fetchDoctors(); }, []);

  const fetchDoctors = async () => {
    const token = localStorage.getItem('token');
    if (!token) return router.push('/auth/login');
    try {
      const api = await import('../utils/apiClient');
      const data = await api.default.get('/api/doctors/search');
      setDoctors(Array.isArray(data) ? data : data.doctors || []);
    } catch (e) { console.error('Error fetching doctors', e); }
    finally { setLoading(false); }
  };

  if (loading) return (<DashboardLayout><div className="min-h-screen flex items-center justify-center"><LoadingSpinner /></div></DashboardLayout>);

  return (
    <DashboardLayout>
      <h1 className="text-2xl font-bold mb-4">Users (Doctors)</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {doctors.map((d: any, i: number) => (
          <div key={d.id || i} className="card">
            <div className="font-semibold">{d.first_name || d.user_name || d.name || d.id}</div>
            <div className="text-sm text-dark-text-secondary">{d.specialization || d.specialty || ''}</div>
          </div>
        ))}
      </div>
    </DashboardLayout>
  );
}
