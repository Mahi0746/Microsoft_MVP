// HealthSync AI - Authentication Store (Zustand)
import { create } from 'zustand';
import { AuthService } from '../services/AuthService';
import { User } from '../types/health';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login: (email: string, password: string) => Promise<boolean>;
  register: (credentials: any) => Promise<boolean>;
  logout: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<boolean>;
  resetPassword: (email: string) => Promise<boolean>;
  clearError: () => void;
  initialize: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,

  login: async (email: string, password: string) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await AuthService.login({ email, password });
      
      if (response.success && response.user) {
        set({
          user: response.user,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
        return true;
      } else {
        set({
          error: response.error || 'Login failed',
          isLoading: false,
        });
        return false;
      }
    } catch (error) {
      set({
        error: 'Network error',
        isLoading: false,
      });
      return false;
    }
  },

  register: async (credentials) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await AuthService.register(credentials);
      
      if (response.success && response.user) {
        set({
          user: response.user,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
        return true;
      } else {
        set({
          error: response.error || 'Registration failed',
          isLoading: false,
        });
        return false;
      }
    } catch (error) {
      set({
        error: 'Network error',
        isLoading: false,
      });
      return false;
    }
  },

  logout: async () => {
    set({ isLoading: true });
    
    try {
      await AuthService.logout();
      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      set({
        error: 'Logout failed',
        isLoading: false,
      });
    }
  },

  updateProfile: async (updates: Partial<User>) => {
    const { user } = get();
    if (!user) return false;

    set({ isLoading: true, error: null });
    
    try {
      const response = await AuthService.updateProfile(user.id, updates);
      
      if (response.success && response.user) {
        set({
          user: response.user,
          isLoading: false,
          error: null,
        });
        return true;
      } else {
        set({
          error: response.error || 'Update failed',
          isLoading: false,
        });
        return false;
      }
    } catch (error) {
      set({
        error: 'Network error',
        isLoading: false,
      });
      return false;
    }
  },

  resetPassword: async (email: string) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await AuthService.resetPassword(email);
      
      if (response.success) {
        set({
          isLoading: false,
          error: null,
        });
        return true;
      } else {
        set({
          error: response.error || 'Password reset failed',
          isLoading: false,
        });
        return false;
      }
    } catch (error) {
      set({
        error: 'Network error',
        isLoading: false,
      });
      return false;
    }
  },

  clearError: () => {
    set({ error: null });
  },

  initialize: async () => {
    set({ isLoading: true });
    
    try {
      const user = await AuthService.getCurrentUser();
      
      if (user) {
        set({
          user,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
      } else {
        set({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          error: null,
        });
      }
    } catch (error) {
      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    }
  },
}));