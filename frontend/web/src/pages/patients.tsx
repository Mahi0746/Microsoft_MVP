import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { useRouter } from 'next/router';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function PatientsPage() {
  const [patients, setPatients] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = async () => {
    const token = localStorage.getItem('token');
    if (!token) return router.push('/auth/login');

    try {
      const api = await import('../utils/apiClient');
      const data = await api.default.get('/api/doctors/dashboard/appointments');
      // Extract unique patients
      const unique: Record<string, any> = {};
      (data || []).forEach((a: any) => {
        if (a.patient_name) unique[a.patient_name] = a;
      });
      setPatients(Object.keys(unique).map(k => ({ name: k, latest: unique[k] })));
    } catch (e) {
      console.error('Error fetching patients', e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return (
    <DashboardLayout>
      <div className="min-h-screen flex items-center justify-center"><LoadingSpinner /></div>
    </DashboardLayout>
  );

  return (
    <DashboardLayout>
      <h1 className="text-2xl font-bold mb-4">Patients</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {patients.length === 0 && <div className="text-sm text-gray-400">No patients found.</div>}
        {patients.map((p) => (
          <div key={p.name} className="card">
            <h3 className="font-semibold">{p.name}</h3>
            <p className="text-sm text-dark-text-secondary">Latest appointment: {p.latest?.created_at || '—'}</p>
          </div>
        ))}
      </div>
    </DashboardLayout>
  );
}
