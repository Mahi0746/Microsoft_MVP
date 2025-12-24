import React from 'react';
import { useRouter } from 'next/router';

interface PatientDashboardProps {
    user: {
        id: string;
        email: string;
        firstName?: string;
        lastName?: string;
        role: string;
    };
}

const PatientDashboard: React.FC<PatientDashboardProps> = ({ user }) => {
    const router = useRouter();

    return (
        <div className="space-y-6">
            {/* Welcome Section */}
            <div className="card-glow">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-dark-text-primary">
                            Welcome back, {user.firstName || 'Patient'}! 👋
                        </h1>
                        <p className="text-dark-text-secondary mt-1">
                            Here's your health overview for today.
                        </p>
                    </div>
                    <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-full text-sm font-medium shadow-lg glow-green">
                        Patient
                    </div>
                </div>
            </div>

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
                                <h3 className="font-medium text-dark-text-primary">Scan Documents</h3>
                                <p className="text-sm text-dark-text-secondary">Upload medical docs</p>
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
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <div>
                                <h3 className="font-medium text-dark-text-primary">Therapy Games</h3>
                                <p className="text-sm text-dark-text-secondary">Start rehabilitation</p>
                            </div>
                        </div>
                    </button>

                    {/* Find Doctors */}
                    <button
                        onClick={() => router.push('/marketplace')}
                        className="card hover:border-yellow-500 group transition-all text-left"
                    >
                        <div className="flex items-center space-x-3">
                            <div className="bg-gradient-to-br from-yellow-500 to-orange-600 p-3 rounded-lg shadow-lg group-hover:scale-110 transition-transform">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
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
    );
};

export default PatientDashboard;
