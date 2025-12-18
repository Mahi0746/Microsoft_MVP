// HealthSync AI - Dashboard Overview Component
import { useEffect, useState } from 'react';
import { 
  UserGroupIcon, 
  CalendarIcon, 
  ChartBarIcon,
  CurrencyDollarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';

interface DashboardStats {
  totalPatients?: number;
  totalAppointments?: number;
  totalRevenue?: number;
  averageRating?: number;
  pendingAppointments?: number;
  completedAppointments?: number;
}

interface RecentActivity {
  id: string;
  type: 'appointment' | 'consultation' | 'review' | 'payment';
  title: string;
  description: string;
  timestamp: string;
  status: 'completed' | 'pending' | 'cancelled';
}

export default function DashboardOverview() {
  const { user } = useAuthStore();
  const [stats, setStats] = useState<DashboardStats>({});
  const [isLoading, setIsLoading] = useState(true);
  const [recentActivities, setRecentActivities] = useState<RecentActivity[]>([]);

  useEffect(() => {
    loadDashboardStats();
    loadRecentActivities();
  }, [user]);

  const loadDashboardStats = async () => {
    try {
      // Mock data for now - would fetch from API
      const mockStats: DashboardStats = {
        totalPatients: 156,
        totalAppointments: 89,
        totalRevenue: 12450,
        averageRating: 4.8,
        pendingAppointments: 12,
        completedAppointments: 77,
      };

      setStats(mockStats);
    } catch (error) {
      console.error('Error loading dashboard stats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadRecentActivities = async () => {
    try {
      // Mock data for recent activities
      const mockActivities: RecentActivity[] = [
        {
          id: '1',
          type: 'appointment',
          title: 'New Appointment Scheduled',
          description: 'John Doe scheduled for tomorrow at 2:00 PM',
          timestamp: '2 hours ago',
          status: 'pending'
        },
        {
          id: '2',
          type: 'consultation',
          title: 'Consultation Completed',
          description: 'Video call with Sarah Johnson completed',
          timestamp: '4 hours ago',
          status: 'completed'
        },
        {
          id: '3',
          type: 'review',
          title: 'New Patient Review',
          description: 'Michael Brown left a 5-star review',
          timestamp: '6 hours ago',
          status: 'completed'
        }
      ];

      setRecentActivities(mockActivities);
    } catch (error) {
      console.error('Error loading recent activities:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="animate-pulse">
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="h-8 w-8 bg-gray-300 rounded"></div>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <div className="h-4 bg-gray-300 rounded w-3/4 mb-2"></div>
                    <div className="h-6 bg-gray-300 rounded w-1/2"></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const statCards = [
    {
      name: 'Total Patients',
      value: stats.totalPatients || 0,
      icon: UserGroupIcon,
      change: '+12%',
      changeType: 'increase' as const,
      color: 'bg-blue-500'
    },
    {
      name: 'Today\'s Appointments',
      value: stats.pendingAppointments || 0,
      icon: CalendarIcon,
      change: '+5%',
      changeType: 'increase' as const,
      color: 'bg-green-500'
    },
    {
      name: 'Monthly Revenue',
      value: `$${(stats.totalRevenue || 0).toLocaleString()}`,
      icon: CurrencyDollarIcon,
      change: '+8%',
      changeType: 'increase' as const,
      color: 'bg-yellow-500'
    },
    {
      name: 'Completed Consultations',
      value: stats.completedAppointments || 0,
      icon: ChartBarIcon,
      change: '+15%',
      changeType: 'increase' as const,
      color: 'bg-purple-500'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Welcome back, Dr. {user?.firstName || 'Doctor'}!
            </h1>
            <p className="mt-1 text-sm text-gray-600">
              Here's what's happening with your practice today.
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Online
            </span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat) => (
          <div key={stat.name} className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`p-3 rounded-md ${stat.color}`}>
                    <stat.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {stat.name}
                    </dt>
                    <dd className="flex items-baseline">
                      <div className="text-2xl font-semibold text-gray-900">
                        {stat.value}
                      </div>
                      <div className={`ml-2 flex items-baseline text-sm font-semibold ${
                        stat.changeType === 'increase' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {stat.changeType === 'increase' ? (
                          <ArrowTrendingUpIcon className="self-center flex-shrink-0 h-4 w-4 text-green-500" />
                        ) : (
                          <ArrowTrendingDownIcon className="self-center flex-shrink-0 h-4 w-4 text-red-500" />
                        )}
                        <span className="ml-1">{stat.change}</span>
                      </div>
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Activities */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
            Recent Activities
          </h3>
          <div className="flow-root">
            <ul className="-mb-8">
              {recentActivities.map((activity, activityIdx) => (
                <li key={activity.id}>
                  <div className="relative pb-8">
                    {activityIdx !== recentActivities.length - 1 ? (
                      <span
                        className="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200"
                        aria-hidden="true"
                      />
                    ) : null}
                    <div className="relative flex space-x-3">
                      <div>
                        <span className={`h-8 w-8 rounded-full flex items-center justify-center ring-8 ring-white ${
                          activity.status === 'completed' ? 'bg-green-500' : 'bg-yellow-500'
                        }`}>
                          {activity.type === 'appointment' && <CalendarIcon className="h-4 w-4 text-white" />}
                          {activity.type === 'consultation' && <UserGroupIcon className="h-4 w-4 text-white" />}
                          {activity.type === 'review' && <ChartBarIcon className="h-4 w-4 text-white" />}
                        </span>
                      </div>
                      <div className="min-w-0 flex-1 pt-1.5 flex justify-between space-x-4">
                        <div>
                          <p className="text-sm text-gray-900 font-medium">{activity.title}</p>
                          <p className="text-sm text-gray-500">{activity.description}</p>
                        </div>
                        <div className="text-right text-sm whitespace-nowrap text-gray-500">
                          {activity.timestamp}
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
            Quick Actions
          </h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <button className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-blue-500 rounded-lg border border-gray-300 hover:border-gray-400 transition-colors">
              <div>
                <span className="rounded-lg inline-flex p-3 bg-blue-50 text-blue-700 ring-4 ring-white">
                  <CalendarIcon className="h-6 w-6" />
                </span>
              </div>
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900">
                  Schedule Appointment
                </h3>
                <p className="mt-2 text-sm text-gray-500">
                  Book a new appointment for a patient
                </p>
              </div>
            </button>

            <button className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-green-500 rounded-lg border border-gray-300 hover:border-gray-400 transition-colors">
              <div>
                <span className="rounded-lg inline-flex p-3 bg-green-50 text-green-700 ring-4 ring-white">
                  <UserGroupIcon className="h-6 w-6" />
                </span>
              </div>
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900">
                  Add New Patient
                </h3>
                <p className="mt-2 text-sm text-gray-500">
                  Register a new patient in the system
                </p>
              </div>
            </button>

            <button className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-purple-500 rounded-lg border border-gray-300 hover:border-gray-400 transition-colors">
              <div>
                <span className="rounded-lg inline-flex p-3 bg-purple-50 text-purple-700 ring-4 ring-white">
                  <ChartBarIcon className="h-6 w-6" />
                </span>
              </div>
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900">
                  View Analytics
                </h3>
                <p className="mt-2 text-sm text-gray-500">
                  Check your practice performance metrics
                </p>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}