// HealthSync AI - Web Authentication Store
import { create } from 'zustand';

interface User {
  id: string;
  email: string;
  role: 'patient' | 'doctor' | 'admin';
  firstName?: string;
  lastName?: string;
  isActive: boolean;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
  initialize: () => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,

  login: async (email: string, password: string) => {
    set({ isLoading: true, error: null });
    
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Enable cookie support
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        set({ error: errorData.detail || 'Login failed', isLoading: false });
        return false;
      }

      const tokenData = await response.json();
      
      // Store tokens in localStorage as backup (cookies are primary now)
      localStorage.setItem('token', tokenData.access_token);
      if (tokenData.refresh_token) {
        localStorage.setItem('refresh_token', tokenData.refresh_token);
      }
      
      // Use user data from response
      if (tokenData.user) {
        const user: User = {
          id: tokenData.user.id,
          email: tokenData.user.email,
          role: tokenData.user.role,
          firstName: tokenData.user.firstName,
          lastName: tokenData.user.lastName,
          isActive: true,
        };

        // Store user data in localStorage for persistence
        localStorage.setItem('user', JSON.stringify(user));

        set({
          user,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
        return true;
      }

      set({ error: 'Login failed - no user data', isLoading: false });
      return false;
    } catch (error) {
      console.error('Login error:', error);
      set({ error: 'Network error - please check if backend is running', isLoading: false });
      return false;
    }
  },

  logout: async () => {
    try {
      // Call logout endpoint on backend
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      await fetch(`${apiUrl}/api/auth/logout`, {
        method: 'POST',
        credentials: 'include', // Include cookies
        headers: {
          'Content-Type': 'application/json',
        },
      }).catch(() => {}); // Ignore errors
    } catch (error) {
      // Ignore logout endpoint errors
    }
    
    // Clear local storage
    localStorage.removeItem('token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    
    set({
      user: null,
      isAuthenticated: false,
      error: null,
    });
  },

  initialize: async () => {
    set({ isLoading: true });
    
    try {
      // Check for existing session
      const token = localStorage.getItem('token');
      const userData = localStorage.getItem('user');
      
      if (token && userData) {
        try {
          const user = JSON.parse(userData);
          
          // Verify token is still valid by making a request to /me endpoint
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
          const response = await fetch(`${apiUrl}/api/auth/me`, {
            method: 'GET',
            credentials: 'include', // Include cookies
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          });

          if (response.ok) {
            // Token is valid, restore session
            set({
              user,
              isAuthenticated: true,
              isLoading: false,
            });
            return;
          } else if (response.status === 401) {
            // Token expired, try to refresh
            const refreshResponse = await fetch(`${apiUrl}/api/auth/refresh`, {
              method: 'POST',
              credentials: 'include', // Include cookies
              headers: {
                'Content-Type': 'application/json',
              },
            });

            if (refreshResponse.ok) {
              const tokenData = await refreshResponse.json();
              localStorage.setItem('token', tokenData.access_token);
              
              set({
                user,
                isAuthenticated: true,
                isLoading: false,
              });
              return;
            }
          }
          
          // Token refresh failed, clear storage
          localStorage.removeItem('token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user');
        } catch (error) {
          // Error parsing user data or validating token
          localStorage.removeItem('token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user');
        }
      }

      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
      });
    } catch (error) {
      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: 'Failed to initialize auth',
      });
    }
  },

  clearError: () => {
    set({ error: null });
  },
}));