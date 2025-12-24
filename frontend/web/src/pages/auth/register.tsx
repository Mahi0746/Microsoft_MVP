import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { useAuth } from '../../contexts/AuthContext';

const RegisterPage: React.FC = () => {
  const router = useRouter();
  const { login, setSession } = useAuth();
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'patient',
    age: '',
    phone: '',
    dateOfBirth: '',
    gender: 'other'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Basic validation
    if (!formData.firstName.trim()) {
      setError('First name is required');
      setLoading(false);
      return;
    }

    if (!formData.lastName.trim()) {
      setError('Last name is required');
      setLoading(false);
      return;
    }

    if (!formData.email.trim()) {
      setError('Email is required');
      setLoading(false);
      return;
    }

    if (!formData.password) {
      setError('Password is required');
      setLoading(false);
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters');
      setLoading(false);
      return;
    }

    // Ensure password byte-length <= 72 (bcrypt limitation)
    if (new TextEncoder().encode(formData.password).length > 72) {
      setError('Password must be at most 72 bytes when UTF-8 encoded (bcrypt limitation)');
      setLoading(false);
      return;
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const requestBody = {
        first_name: formData.firstName,
        last_name: formData.lastName,
        email: formData.email,
        password: formData.password,
        role: formData.role,
        phone: formData.phone || null,
        date_of_birth: formData.dateOfBirth || null,
        gender: formData.gender || null
      };

      console.log('Sending registration request:', requestBody);

      const response = await fetch(`${apiUrl}/api/auth/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData.detail || errorData.message || 'Registration failed');
        return;
      }

      const tokenData = await response.json();
      console.log('Registration response:', tokenData);

      // Store tokens
      localStorage.setItem('token', tokenData.access_token);
      localStorage.setItem('refresh_token', tokenData.refresh_token);
      
      // Get user profile
      const profileResponse = await fetch(`${apiUrl}/api/auth/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${tokenData.access_token}`,
          'Content-Type': 'application/json',
        },
      });

      if (profileResponse.ok) {
        const userProfile = await profileResponse.json();
        // Prefer directly setting the session in AuthContext to avoid re-calling login
        if (setSession) {
          setSession({ access_token: tokenData.access_token, refresh_token: tokenData.refresh_token }, userProfile);
          router.push('/dashboard');
        } else {
          // Fallback: perform a normal login
          const loginSuccess = await login(formData.email, formData.password);
          if (loginSuccess) {
            router.push('/dashboard');
          } else {
            setError('Registration successful but login failed. Please try logging in manually.');
          }
        }
      } else {
        setError('Registration successful but failed to get user profile. Please try logging in manually.');
      }
    } catch (err) {
      setError('Network error. Please try again.');
      console.error('Registration error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-dark-bg-primary py-12 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-primary-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      <div className="max-w-2xl w-full space-y-8 relative z-10">
        <div>
          <div className="mx-auto h-16 w-16 flex items-center justify-center rounded-2xl bg-gradient-primary shadow-lg glow-blue">
            <svg className="h-9 w-9 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gradient">
            Create your account
          </h2>
          <p className="mt-2 text-center text-sm text-dark-text-secondary">
            Join HealthSync AI for personalized healthcare
          </p>
        </div>

        <form className="mt-8 space-y-6 glass-strong p-8 rounded-2xl border border-dark-border-primary" onSubmit={handleSubmit}>
          {error && (
            <div className="bg-red-900 bg-opacity-30 border border-red-500 text-red-300 px-4 py-3 rounded-xl">
              {error}
            </div>
          )}

          <div className="space-y-5">
            {/* Name Fields */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="firstName" className="block text-sm font-medium text-dark-text-primary mb-2">
                  First Name
                </label>
                <input
                  id="firstName"
                  name="firstName"
                  type="text"
                  required
                  value={formData.firstName}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  placeholder="First name"
                />
              </div>
              <div>
                <label htmlFor="lastName" className="block text-sm font-medium text-dark-text-primary mb-2">
                  Last Name
                </label>
                <input
                  id="lastName"
                  name="lastName"
                  type="text"
                  required
                  value={formData.lastName}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  placeholder="Last name"
                />
              </div>
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-dark-text-primary mb-2">
                Email Address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                value={formData.email}
                onChange={handleInputChange}
                className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                placeholder="Email address"
              />
            </div>

            {/* Password Fields */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-dark-text-primary mb-2">
                  Password
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  maxLength={72}
                  value={formData.password}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  placeholder="Password"
                />
              </div>
              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-dark-text-primary mb-2">
                  Confirm Password
                </label>
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type="password"
                  required
                  maxLength={72}
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  placeholder="Confirm password"
                />
              </div>
            </div>

            {/* Role Selection */}
            <div>
              <label htmlFor="role" className="block text-sm font-medium text-dark-text-primary mb-2">
                Account Type
              </label>
              <select
                id="role"
                name="role"
                value={formData.role}
                onChange={handleInputChange}
                className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              >
                <option value="patient">Patient</option>
                <option value="doctor">Healthcare Provider</option>
                <option value="admin">Administrator</option>
              </select>
            </div>

            {/* Additional Info */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="age" className="block text-sm font-medium text-dark-text-primary mb-2">
                  Age
                </label>
                <input
                  id="age"
                  name="age"
                  type="number"
                  min="1"
                  max="120"
                  value={formData.age}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  placeholder="Age"
                />
              </div>
              <div>
                <label htmlFor="gender" className="block text-sm font-medium text-dark-text-primary mb-2">
                  Gender
                </label>
                <select
                  id="gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                  <option value="prefer_not_to_say">Prefer not to say</option>
                </select>
              </div>
            </div>

            {/* Phone */}
            <div>
              <label htmlFor="phone" className="block text-sm font-medium text-dark-text-primary mb-2">
                Phone Number (Optional)
              </label>
              <input
                id="phone"
                name="phone"
                type="tel"
                value={formData.phone}
                onChange={handleInputChange}
                className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                placeholder="Phone number"
              />
            </div>

            {/* Date of Birth */}
            <div>
              <label htmlFor="dateOfBirth" className="block text-sm font-medium text-dark-text-primary mb-2">
                Date of Birth (Optional)
              </label>
              <input
                id="dateOfBirth"
                name="dateOfBirth"
                type="date"
                value={formData.dateOfBirth}
                onChange={handleInputChange}
                className="w-full px-4 py-3 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-semibold rounded-xl text-white bg-gradient-primary hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 focus:ring-offset-dark-bg-primary disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg glow-blue"
            >
              {loading ? (
                <div className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creating Account...
                </div>
              ) : (
                'Create Account'
              )}
            </button>
          </div>

          <div className="text-center">
            <span className="text-sm text-dark-text-secondary">
              Already have an account?{' '}
              <Link href="/auth/login" className="font-medium text-primary-400 hover:text-primary-300 transition-colors">
                Sign in here
              </Link>
            </span>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RegisterPage;