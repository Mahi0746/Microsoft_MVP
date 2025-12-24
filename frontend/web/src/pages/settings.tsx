import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { useRouter } from 'next/router';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function SettingsPage() {
  const [profile, setProfile] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => { fetchProfile(); }, []);

  const fetchProfile = async () => {
    const token = localStorage.getItem('token');
    if (!token) return router.push('/auth/login');

    try {
      const api = await import('../utils/apiClient');
      const data = await api.default.get('/api/auth/me');
      setProfile(data);
    } catch (e) { console.error('Error fetching profile', e); }
    finally { setLoading(false); }
  };

  const save = async () => {
    const token = localStorage.getItem('token');
    if (!token) return router.push('/auth/login');
    try {
      const api = await import('../utils/apiClient');
      await api.default.put('/api/auth/profile', profile);
      alert('Profile saved');
    } catch (e) { console.error('Error saving profile', e); }
  };

  if (loading) return (<DashboardLayout><div className="min-h-screen flex items-center justify-center"><LoadingSpinner /></div></DashboardLayout>);

  return (
    <DashboardLayout>
      <h1 className="text-2xl font-bold mb-4">Settings</h1>
      {!profile && <div className="text-sm text-gray-400">No profile loaded.</div>}
      {profile && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium">First name</label>
            <input className="input" value={profile.first_name || ''} onChange={e => setProfile({ ...profile, first_name: e.target.value })} />
          </div>
          <div>
            <label className="block text-sm font-medium">Last name</label>
            <input className="input" value={profile.last_name || ''} onChange={e => setProfile({ ...profile, last_name: e.target.value })} />
          </div>
          <div>
            <button className="btn" onClick={save}>Save</button>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
