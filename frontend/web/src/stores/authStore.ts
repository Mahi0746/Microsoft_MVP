// HealthSync AI - Web Authentication Store
import { create } from 'zustand';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        set({ error: errorData.detail || errorData.message || 'Login failed', isLoading: false });
        return false;
      }

      const tokenData = await response.json();
      
      // Store tokens
      localStorage.setItem('token', tokenData.access_token);
      localStorage.setItem('refresh_token', tokenData.refresh_token);
      
      // Get user profile
      const profileResponse = await fetch(`${API_URL}/api/auth/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${tokenData.access_token}`,
          'Content-Type': 'application/json',
        },
      });

      if (profileResponse.ok) {
        const userProfile = await profileResponse.json();
        const user: User = {
          id: userProfile.id,
          email: userProfile.email,
          role: userProfile.role,
          firstName: userProfile.first_name,
          lastName: userProfile.last_name,
          isActive: userProfile.is_active,
        };

        set({
          user,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
        return true;
      }

      set({ error: 'Login failed', isLoading: false });
      return false;
    } catch (error) {
      set({ error: 'Network error', isLoading: false });
      return false;
    }
  },

  logout: async () => {
    localStorage.removeItem('token');
    localStorage.removeItem('refresh_token');
    set({
      user: null,
      isAuthenticated: false,
      error: null,
    });
  },

  initialize: async () => {
    set({ isLoading: true });
    
    try {
      const token = localStorage.getItem('token');
      
      if (token) {
        const profileResponse = await fetch(`${API_URL}/api/auth/me`, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });

        if (profileResponse.ok) {
          const userProfile = await profileResponse.json();
          const user: User = {
            id: userProfile.id,
            email: userProfile.email,
            role: userProfile.role,
            firstName: userProfile.first_name,
            lastName: userProfile.last_name,
            isActive: userProfile.is_active,
          };

          set({
            user,
            isAuthenticated: true,
            isLoading: false,
          });
          return;
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