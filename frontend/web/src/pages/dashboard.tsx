import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { useAuthStore } from '../stores/authStore';
import DashboardLayout from '../components/layout/DashboardLayout';
import PatientDashboard from '../components/dashboard/PatientDashboard';
import DoctorDashboard from '../components/dashboard/DoctorDashboard';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const DashboardPage: React.FC = () => {
  const { user, isLoading: loading } = useAuthStore();
  const router = useRouter();
  const [dataLoading, setDataLoading] = useState(true);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/');
      return;
    }

    if (user) {
      // Simulate data loading or fetch any general dashboard data
      setDataLoading(false);
    }
  }, [user, loading, router]);

  if (loading || dataLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-dark-bg-primary">
        <LoadingSpinner />
      </div>
    );
  }

  if (!user) {
    return null; // Will redirect to login
  }

  // Render role-specific dashboard
  return (
    <DashboardLayout>
      <div className="animate-fade-in">
        {user.role === 'doctor' ? (
          <DoctorDashboard user={user} />
        ) : (
          <PatientDashboard user={user} />
        )}
      </div>
    </DashboardLayout>
  );
};

export default DashboardPage;