import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { useRouter } from 'next/router';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function AppointmentsPage() {
  const [appointments, setAppointments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => { fetchAppointments(); }, []);

  const fetchAppointments = async () => {
    const token = localStorage.getItem('token');
    if (!token) return router.push('/auth/login');

    try {
      const api = await import('../utils/apiClient');
      const data = await api.default.get('/api/doctors/dashboard/appointments');
      setAppointments(data || []);
    } catch (e) {
      console.error('Error fetching appointments', e);
    } finally { setLoading(false); }
  };

  if (loading) return (
    <DashboardLayout><div className="min-h-screen flex items-center justify-center"><LoadingSpinner /></div></DashboardLayout>
  );

  return (
    <DashboardLayout>
      <h1 className="text-2xl font-bold mb-4">Appointments</h1>
      <div className="space-y-3">
        {appointments.length === 0 && <div className="text-sm text-gray-400">No appointments found.</div>}
        {appointments.map((a: any) => (
          <div key={a.id || a.appointment_id} className="card">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold">{a.patient_name || a.patient_id}</div>
                <div className="text-sm text-dark-text-secondary">{a.symptoms_summary || '—'}</div>
              </div>
              <div className="text-sm text-dark-text-secondary">{a.status || a.urgency_level || '—'}</div>
            </div>
          </div>
        ))}
      </div>
    </DashboardLayout>
  );
}
