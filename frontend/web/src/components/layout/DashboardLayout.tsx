// HealthSync AI - Dashboard Layout Component
import { useState, ReactNode } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import {
  HomeIcon,
  UserGroupIcon,
  CalendarIcon,
  ChartBarIcon,
  CogIcon,
  BellIcon,
  UserCircleIcon,
  Bars3Icon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';
import { motion, AnimatePresence } from 'framer-motion';

interface DashboardLayoutProps {
  children: ReactNode;
}

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  roles: string[];
}

const navigation: NavigationItem[] = [
  { name: 'Dashboard', href: '/', icon: HomeIcon, roles: ['doctor', 'admin'] },
  { name: 'Patients', href: '/patients', icon: UserGroupIcon, roles: ['doctor'] },
  { name: 'Appointments', href: '/appointments', icon: CalendarIcon, roles: ['doctor'] },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon, roles: ['doctor', 'admin'] },
  { name: 'Marketplace', href: '/marketplace', icon: ChartBarIcon, roles: ['admin'] },
  { name: 'Users', href: '/users', icon: UserGroupIcon, roles: ['admin'] },
  { name: 'Settings', href: '/settings', icon: CogIcon, roles: ['doctor', 'admin'] },
];

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { user, logout } = useAuthStore();
  const router = useRouter();

  const filteredNavigation = navigation.filter(item => 
    user?.role && item.roles.includes(user.role)
  );

  const handleLogout = async () => {
    await logout();
    router.push('/auth/login');
  };

  return (
    <div className="min-h-screen bg-dark-bg-primary">
      {/* Mobile sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 lg:hidden"
            >
              <div
                className="fixed inset-0 bg-black bg-opacity-75"
                onClick={() => setSidebarOpen(false)}
              />
            </motion.div>

            <motion.div
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed inset-y-0 left-0 z-50 w-64 glass-strong lg:hidden border-r border-dark-border-primary"
            >
              <div className="flex h-16 items-center justify-between px-4 border-b border-dark-border-primary">
                <div className="flex items-center">
                  <div className="h-8 w-8 bg-gradient-primary rounded-lg flex items-center justify-center shadow-lg">
                    <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                    </svg>
                  </div>
                  <span className="ml-2 text-lg font-bold text-gradient">HealthSync AI</span>
                </div>
                <button
                  type="button"
                  className="text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                  onClick={() => setSidebarOpen(false)}
                >
                  <XMarkIcon className="h-6 w-6" />
                </button>
              </div>
              <nav className="mt-8 px-4">
                <ul className="space-y-2">
                  {filteredNavigation.map((item) => (
                    <li key={item.name}>
                      <Link
                        href={item.href}
                        className={`group flex items-center px-3 py-3 text-sm font-medium rounded-lg transition-all ${
                          router.pathname === item.href
                            ? 'bg-gradient-primary text-white shadow-lg glow-blue'
                            : 'text-dark-text-secondary hover:bg-dark-bg-hover hover:text-dark-text-primary'
                        }`}
                        onClick={() => setSidebarOpen(false)}
                      >
                        <item.icon className={`mr-3 h-5 w-5 ${router.pathname === item.href ? 'text-white' : ''}`} />
                        {item.name}
                      </Link>
                    </li>
                  ))}
                </ul>
              </nav>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-72 lg:flex-col">
        <div className="flex min-h-0 flex-1 flex-col glass-strong border-r border-dark-border-primary">
          <div className="flex h-20 flex-shrink-0 items-center px-6 gradient-primary border-b border-dark-border-primary">
            <div className="flex items-center">
              <div className="h-10 w-10 bg-white bg-opacity-20 rounded-lg flex items-center justify-center shadow-lg backdrop-blur-sm">
                <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
              </div>
              <span className="ml-3 text-white font-bold text-xl">HealthSync AI</span>
            </div>
          </div>
          <div className="flex flex-1 flex-col overflow-y-auto">
            <nav className="flex-1 px-4 py-6">
              <ul className="space-y-2">
                {filteredNavigation.map((item) => (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className={`group flex items-center px-4 py-3 text-sm font-medium rounded-xl transition-all ${
                        router.pathname === item.href
                          ? 'bg-gradient-primary text-white shadow-lg glow-blue'
                          : 'text-dark-text-secondary hover:bg-dark-bg-hover hover:text-dark-text-primary'
                      }`}
                    >
                      <item.icon className={`mr-3 h-5 w-5 ${router.pathname === item.href ? 'text-white' : ''}`} />
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
            <div className="px-4 py-4 border-t border-dark-border-primary">
              <div className="flex items-center space-x-3 p-3 bg-dark-bg-tertiary rounded-lg">
                <div className="h-10 w-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <span className="text-white font-semibold text-sm">
                    {user?.firstName?.[0]}{user?.lastName?.[0]}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-dark-text-primary truncate">
                    {user?.firstName} {user?.lastName}
                  </p>
                  <p className="text-xs text-dark-text-secondary truncate">{user?.email}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-72">
        {/* Top navigation */}
        <div className="sticky top-0 z-10 glass-strong border-b border-dark-border-primary">
          <div className="flex h-20 items-center justify-between px-4 sm:px-6 lg:px-8">
            <button
              type="button"
              className="text-dark-text-secondary hover:text-dark-text-primary lg:hidden transition-colors"
              onClick={() => setSidebarOpen(true)}
            >
              <Bars3Icon className="h-6 w-6" />
            </button>

            <div className="flex items-center space-x-4">
              {/* Real-time Status Indicator */}
              <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 bg-dark-bg-tertiary rounded-lg border border-dark-border-primary">
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs font-medium text-dark-text-secondary">Live</span>
              </div>

              {/* Notifications */}
              <button
                type="button"
                className="relative rounded-xl bg-dark-bg-tertiary p-2.5 text-dark-text-secondary hover:text-dark-text-primary hover:bg-dark-bg-hover focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-dark-bg-primary transition-all border border-dark-border-primary"
              >
                <BellIcon className="h-5 w-5" />
                <span className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-gradient-to-r from-red-500 to-pink-600 text-xs text-white flex items-center justify-center font-semibold shadow-lg">
                  3
                </span>
              </button>

              {/* Logout button */}
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-4 py-2 border border-dark-border-primary text-sm font-medium rounded-xl text-dark-text-primary bg-dark-bg-tertiary hover:bg-dark-bg-hover focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-dark-bg-primary transition-all"
              >
                Logout
              </button>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1">
          <div className="py-8">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}