// HealthSync AI - Login Page
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Link from 'next/link';
import { useForm } from 'react-hook-form';
import { toast } from 'react-hot-toast';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';
import LoadingSpinner from '../../components/ui/LoadingSpinner';

interface LoginForm {
  email: string;
  password: string;
}

export default function LoginPage() {
  const router = useRouter();
  const { login, isAuthenticated, isLoading, error, clearError } = useAuthStore();
  const [showPassword, setShowPassword] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginForm>();

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, router]);

  useEffect(() => {
    if (error) {
      toast.error(error);
      clearError();
    }
  }, [error, clearError]);

  const onSubmit = async (data: LoginForm) => {
    const success = await login(data.email, data.password);
    if (success) {
      toast.success('Login successful!');
      router.push('/dashboard');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Login - HealthSync AI</title>
        <meta name="description" content="Login to HealthSync AI Dashboard" />
      </Head>

      <div className="min-h-screen flex items-center justify-center bg-dark-bg-primary py-12 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-primary-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
        </div>

        <div className="max-w-md w-full space-y-8 relative z-10">
          <div>
            <div className="mx-auto h-16 w-16 flex items-center justify-center rounded-2xl bg-gradient-primary shadow-lg glow-blue">
              <svg className="h-9 w-9 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
            </div>
            <h2 className="mt-6 text-center text-3xl font-extrabold text-gradient">
              Sign in to HealthSync AI
            </h2>
            <p className="mt-2 text-center text-sm text-dark-text-secondary">
              Secure healthcare platform for medical professionals
            </p>
          </div>

          <form className="mt-8 space-y-6 glass-strong p-8 rounded-2xl border border-dark-border-primary" onSubmit={handleSubmit(onSubmit)}>
            <div className="space-y-5">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-dark-text-primary mb-2">
                  Email address
                </label>
                <input
                  {...register('email', {
                    required: 'Email is required',
                    pattern: {
                      value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                      message: 'Invalid email address',
                    },
                  })}
                  type="email"
                  autoComplete="email"
                  className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  placeholder="Enter your email"
                />
                {errors.email && (
                  <p className="mt-2 text-sm text-red-400">{errors.email.message}</p>
                )}
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-dark-text-primary mb-2">
                  Password
                </label>
                <div className="relative">
                  <input
                    {...register('password', {
                      required: 'Password is required',
                      minLength: {
                        value: 6,
                        message: 'Password must be at least 6 characters',
                      },
                    })}
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    className="w-full px-4 py-3 pr-12 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                    placeholder="Enter your password"
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeSlashIcon className="h-5 w-5" />
                    ) : (
                      <EyeIcon className="h-5 w-5" />
                    )}
                  </button>
                </div>
                {errors.password && (
                  <p className="mt-2 text-sm text-red-400">{errors.password.message}</p>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="text-sm">
                <Link href="/auth/forgot-password" className="font-medium text-primary-400 hover:text-primary-300 transition-colors">
                  Forgot your password?
                </Link>
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={isSubmitting}
                className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-semibold rounded-xl text-white bg-gradient-primary hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 focus:ring-offset-dark-bg-primary disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg glow-blue"
              >
                {isSubmitting ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : null}
                Sign in
              </button>
            </div>

            <div className="text-center">
              <p className="text-sm text-dark-text-secondary">
                Don't have an account?{' '}
                <Link href="/auth/register" className="font-medium text-primary-400 hover:text-primary-300 transition-colors">
                  Contact administrator
                </Link>
              </p>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}