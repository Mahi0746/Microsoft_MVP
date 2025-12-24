import React from 'react';
import { useRouter } from 'next/router';

interface DoctorDashboardProps {
    user: {
        id: string;
        email: string;
        firstName?: string;
        lastName?: string;
        role: string;
    };
}

const DoctorDashboard: React.FC<DoctorDashboardProps> = ({ user }) => {
    const router = useRouter();

    return (
        <div className="space-y-6">
            {/* Welcome Section */}
            <div className="card-glow">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-dark-text-primary">
                            Welcome, Dr. {user.lastName || user.firstName || 'Provider'}! 🩺
                        </h1>
                        <p className="text-dark-text-secondary mt-1">
                            Here's your practice overview for today.
                        </p>
                    </div>
                    <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-full text-sm font-medium shadow-lg glow-purple">
                        Healthcare Provider
                    </div>
                </div>
            </div>

            {/* Stats Overview */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-dark-text-secondary text-sm">Today's Appointments</p>
                            <p className="text-2xl font-bold text-dark-text-primary mt-1">8</p>
                        </div>
                        <div className="bg-blue-900 bg-opacity-30 p-3 rounded-xl">
                            <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-dark-text-secondary text-sm">Active Patients</p>
                            <p className="text-2xl font-bold text-dark-text-primary mt-1">156</p>
                        </div>
                        <div className="bg-green-900 bg-opacity-30 p-3 rounded-xl">
                            <svg className="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                            </svg>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-dark-text-secondary text-sm">Pending Reviews</p>
                            <p className="text-2xl font-bold text-dark-text-primary mt-1">12</p>
                        </div>
                        <div className="bg-yellow-900 bg-opacity-30 p-3 rounded-xl">
                            <svg className="w-6 h-6 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-dark-text-secondary text-sm">This Month</p>
                            <p className="text-2xl font-bold text-dark-text-primary mt-1">$12.4k</p>
                        </div>
                        <div className="bg-purple-900 bg-opacity-30 p-3 rounded-xl">
                            <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                    </div>
                </div>
            </div>

            {/* Quick Actions */}
            <div className="card">
                <h2 className="text-lg font-semibold text-dark-text-primary mb-4">Quick Actions</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

                    {/* View Patients */}
                    <button
                        onClick={() => router.push('/marketplace')}
                        className="card hover:border-primary-500 group transition-all text-left"
                    >
                        <div className="flex items-center space-x-3">
                            <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-3 rounded-lg shadow-lg glow-blue group-hover:scale-110 transition-transform">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                                </svg>
                            </div>
                            <div>
                                <h3 className="font-medium text-dark-text-primary">Patient Records</h3>
                                <p className="text-sm text-dark-text-secondary">View patient list</p>
                            </div>
                        </div>
                    </button>

                    {/* Review Scans */}
                    <button
                        onClick={() => router.push('/ar-scanner-rag')}
                        className="card hover:border-green-500 group transition-all text-left"
                    >
                        <div className="flex items-center space-x-3">
                            <div className="bg-gradient-to-br from-green-500 to-emerald-600 p-3 rounded-lg shadow-lg glow-green group-hover:scale-110 transition-transform">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                            </div>
                            <div>
                                <h3 className="font-medium text-dark-text-primary">Review Documents</h3>
                                <p className="text-sm text-dark-text-secondary">Scan analysis</p>
                            </div>
                        </div>
                    </button>

                    {/* AI Assistant */}
                    <button
                        onClick={() => router.push('/voice-doctor')}
                        className="card hover:border-purple-500 group transition-all text-left"
                    >
                        <div className="flex items-center space-x-3">
                            <div className="bg-gradient-to-br from-purple-500 to-pink-600 p-3 rounded-lg shadow-lg glow-purple group-hover:scale-110 transition-transform">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                                </svg>
                            </div>
                            <div>
                                <h3 className="font-medium text-dark-text-primary">AI Assistant</h3>
                                <p className="text-sm text-dark-text-secondary">Voice consultation</p>
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
                            <p className="text-sm font-medium text-dark-text-primary">New patient consultation completed</p>
                            <p className="text-xs text-dark-text-secondary">AI analysis reviewed and approved</p>
                        </div>
                        <span className="text-xs text-dark-text-tertiary">2 min ago</span>
                    </div>

                    <div className="flex items-center space-x-3 p-3 bg-dark-bg-tertiary rounded-lg border border-dark-border-primary">
                        <div className="bg-gradient-to-br from-green-500 to-emerald-600 p-2 rounded-full shadow-lg">
                            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </div>
                        <div className="flex-1">
                            <p className="text-sm font-medium text-dark-text-primary">Medical document scanned</p>
                            <p className="text-xs text-dark-text-secondary">Lab report for Patient #1234</p>
                        </div>
                        <span className="text-xs text-dark-text-tertiary">15 min ago</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DoctorDashboard;
