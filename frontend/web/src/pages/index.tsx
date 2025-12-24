// HealthSync AI - Landing Page
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Link from 'next/link';
import { useAuthStore } from '../stores/authStore';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ParticleBackground from '../components/ui/ParticleBackground';

export default function HomePage() {
  const router = useRouter();
  const { user, isLoading: loading } = useAuthStore();

  // Removed auto-redirect to prevent Fast Refresh loops
  // User can manually navigate to dashboard via the "Get Started" or "Sign In" buttons logic if needed

  // if (loading) ... (keep loading spinner if desired, or remove to show landing page immediately)

  return (
    <>
      <Head>
        <title>HealthSync AI - Intelligent Healthcare Platform</title>
        <meta name="description" content="Revolutionary AI-powered healthcare platform with voice consultations, AR scanning, and personalized health insights" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-dark relative overflow-hidden">
        <ParticleBackground />
        {/* Navigation */}
        <nav className="glass-strong border-b border-dark-border-primary sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <div className="flex-shrink-0 flex items-center">
                  <div className="h-8 w-8 bg-gradient-primary rounded-lg flex items-center justify-center shadow-lg glow-blue">
                    <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                    </svg>
                  </div>
                  <span className="ml-2 text-xl font-bold text-gradient">HealthSync AI</span>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <Link href="/auth/login" className="text-dark-text-secondary hover:text-dark-text-primary px-3 py-2 rounded-md text-sm font-medium transition-colors">
                  Sign In
                </Link>
                <Link href="/auth/register" className="btn-primary">
                  Get Started
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 animate-fade-in">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-extrabold text-dark-text-primary mb-6">
              The Future of
              <span className="text-gradient"> Healthcare</span>
              <br />
              is Here
            </h1>
            <p className="text-xl text-dark-text-secondary mb-8 max-w-3xl mx-auto">
              Experience revolutionary AI-powered healthcare with voice consultations, AR medical scanning,
              gamified therapy, and personalized health predictions.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/auth/register" className="btn-primary px-8 py-3 text-lg">
                Start Your Health Journey
              </Link>
              <Link href="/auth/login" className="btn-secondary px-8 py-3 text-lg">
                Sign In
              </Link>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-dark-text-primary mb-4">
              12 Revolutionary Features
            </h2>
            <p className="text-lg text-dark-text-secondary">
              Everything you need for comprehensive healthcare management
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Voice AI Doctor */}
            <div className="card-glow group">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4 shadow-lg glow-blue group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-dark-text-primary mb-2">Voice AI Doctor</h3>
              <p className="text-dark-text-secondary">Real-time AI consultations with voice analysis and intelligent health recommendations.</p>
            </div>

            {/* AR Medical Scanner */}
            <div className="card-glow group">
              <div className="bg-gradient-to-br from-green-500 to-emerald-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4 shadow-lg glow-green group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-dark-text-primary mb-2">AR Medical Scanner</h3>
              <p className="text-dark-text-secondary">Scan prescriptions, lab reports, and medical documents with AI-powered OCR analysis.</p>
            </div>

            {/* Pain-to-Game Therapy */}
            <div className="card-glow group">
              <div className="bg-gradient-to-br from-purple-500 to-pink-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4 shadow-lg glow-purple group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.5a2.5 2.5 0 110 5H9m4.5-1.206a11.955 11.955 0 01-2.5 2.5M15 6.5a11.955 11.955 0 01-2.5-2.5M9 6.5a11.955 11.955 0 00-2.5-2.5m1.5 2.5h3m-3 0h-.5a2.5 2.5 0 00-2.5 2.5V12a2.5 2.5 0 002.5 2.5H9m-3-6h3m-3 0h-.5a2.5 2.5 0 00-2.5 2.5v3a2.5 2.5 0 002.5 2.5H9" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-dark-text-primary mb-2">Gamified Therapy</h3>
              <p className="text-dark-text-secondary">Transform rehabilitation into engaging games with motion tracking and progress rewards.</p>
            </div>

            {/* Doctor Marketplace */}
            <div className="card-glow group">
              <div className="bg-gradient-to-br from-yellow-500 to-orange-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4 shadow-lg group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-dark-text-primary mb-2">Doctor Marketplace</h3>
              <p className="text-dark-text-secondary">Find and book appointments with specialists using AI-powered matching and bidding.</p>
            </div>

            {/* Future-You Simulator */}
            <div className="card-glow group">
              <div className="bg-gradient-to-br from-red-500 to-pink-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4 shadow-lg group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-dark-text-primary mb-2">Future-You Simulator</h3>
              <p className="text-dark-text-secondary">See your future health with AI age progression and personalized lifestyle predictions.</p>
            </div>

            {/* Health Analytics */}
            <div className="card-glow group">
              <div className="bg-gradient-to-br from-indigo-500 to-purple-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4 shadow-lg glow-purple group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-dark-text-primary mb-2">Health Analytics</h3>
              <p className="text-dark-text-secondary">Comprehensive health insights with ML-powered disease prediction and family health graphs.</p>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="relative z-10 gradient-primary py-16 overflow-hidden">
          <div className="absolute inset-0 bg-black opacity-20"></div>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Transform Your Healthcare?
            </h2>
            <p className="text-xl text-white/90 mb-8">
              Join thousands of users already experiencing the future of healthcare
            </p>
            <Link href="/auth/register" className="bg-white text-primary-600 hover:bg-gray-100 px-8 py-3 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 shadow-xl inline-block">
              Get Started Free
            </Link>
          </div>
        </div>

        {/* Footer */}
        <footer className="relative z-10 bg-dark-bg-secondary border-t border-dark-border-primary py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <div className="flex items-center justify-center mb-4">
                <div className="h-8 w-8 bg-gradient-primary rounded-lg flex items-center justify-center shadow-lg">
                  <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                </div>
                <span className="ml-2 text-xl font-bold text-gradient">HealthSync AI</span>
              </div>
              <p className="text-dark-text-tertiary">
                © 2024 HealthSync AI. Revolutionizing healthcare with artificial intelligence.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
}