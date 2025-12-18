import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import DashboardOverview from '../components/dashboard/DashboardOverview';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const DashboardPage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [dashboardData, setDashboardData] = useState(null);
  const [dataLoading, setDataLoading] = useState(true);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
      return;
    }

    if (user) {
      fetchDashboardData();
    }
  }, [user, loading, router]);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/dashboard/stats', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setDashboardData(data);
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setDataLoading(false);
    }
  };

  if (loading || dataLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (!user) {
    return null; // Will redirect to login
  }

  return (
    <DashboardLayout>
      <div className="space-y-6 animate-fade-in">
        {/* Welcome Header */}
        <div className="card-glow">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-dark-text-primary">
                Welcome back, {user.firstName}!
              </h1>
              <p className="text-dark-text-secondary mt-1">
                Here's what's happening with your health today.
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-full text-sm font-medium shadow-lg glow-green">
                {user.role === 'patient' ? 'Patient' : user.role === 'doctor' ? 'Healthcare Provider' : 'Administrator'}
              </div>
            </div>
          </div>
        </div>

        {/* Dashboard Overview */}
        <DashboardOverview />

        {/* Quick Actions */}
        <div className="card">
          <h2 className="text-lg font-semibold text-dark-text-primary mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            
            {/* Voice AI Doctor */}
            <button
              onClick={() => router.push('/voice-doctor')}
              className="card hover:border-primary-500 group transition-all text-left"
            >
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-3 rounded-lg shadow-lg glow-blue group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-medium text-dark-text-primary">Voice AI Doctor</h3>
                  <p className="text-sm text-dark-text-secondary">Start consultation</p>
                </div>
              </div>
            </button>

            {/* AR Scanner */}
            <button
              onClick={() => router.push('/ar-scanner')}
              className="card hover:border-green-500 group transition-all text-left"
            >
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-br from-green-500 to-emerald-600 p-3 rounded-lg shadow-lg glow-green group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-medium text-dark-text-primary">AR Scanner</h3>
                  <p className="text-sm text-dark-text-secondary">Scan documents</p>
                </div>
              </div>
            </button>

            {/* Therapy Games */}
            <button
              onClick={() => router.push('/therapy-game')}
              className="card hover:border-purple-500 group transition-all text-left"
            >
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-br from-purple-500 to-pink-600 p-3 rounded-lg shadow-lg glow-purple group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.5a2.5 2.5 0 110 5H9m4.5-1.206a11.955 11.955 0 01-2.5 2.5M15 6.5a11.955 11.955 0 01-2.5-2.5M9 6.5a11.955 11.955 0 00-2.5-2.5m1.5 2.5h3m-3 0h-.5a2.5 2.5 0 00-2.5 2.5V12a2.5 2.5 0 002.5 2.5H9m-3-6h3m-3 0h-.5a2.5 2.5 0 00-2.5 2.5v3a2.5 2.5 0 002.5 2.5H9" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-medium text-dark-text-primary">Therapy Games</h3>
                  <p className="text-sm text-dark-text-secondary">Start rehabilitation</p>
                </div>
              </div>
            </button>

            {/* Doctor Marketplace */}
            <button
              onClick={() => router.push('/marketplace')}
              className="card hover:border-yellow-500 group transition-all text-left"
            >
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-br from-yellow-500 to-orange-600 p-3 rounded-lg shadow-lg group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-medium text-dark-text-primary">Find Doctors</h3>
                  <p className="text-sm text-dark-text-secondary">Book appointments</p>
                </div>
              </div>
            </button>

          </div>
        </div>

        {/* Recent Activity */}
        <div className="card">
          <h2 className="text-lg font-semibold text-dark-text-primary mb-4">Recent Activity</h2>
          <div className="space-y-3">
            <div className="flex items-center space-x-3 p-3 bg-dark-bg-tertiary rounded-lg border border-dark-border-primary">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-2 rounded-full shadow-lg">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-dark-text-primary">Account created successfully</p>
                <p className="text-xs text-dark-text-secondary">Welcome to HealthSync AI!</p>
              </div>
              <span className="text-xs text-dark-text-tertiary">Just now</span>
            </div>
          </div>
        </div>

        {/* Health Insights */}
        <div className="gradient-primary shadow-xl rounded-xl p-6 text-white relative overflow-hidden">
          <div className="absolute inset-0 bg-black opacity-10"></div>
          <div className="relative z-10">
            <h2 className="text-lg font-semibold mb-2">Health Insights</h2>
            <p className="text-white/90 mb-4">
              Discover personalized health recommendations and track your wellness journey with our AI-powered platform.
            </p>
            <button
              onClick={() => router.push('/future-simulator')}
              className="bg-white text-primary-600 px-4 py-2 rounded-lg font-medium hover:bg-gray-100 transition-all transform hover:scale-105 shadow-lg"
            >
              Try Future-You Simulator
            </button>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default DashboardPage;