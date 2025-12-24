// HealthSync AI - Authentication Context
import { createContext, useContext, useEffect, ReactNode, useState } from 'react';
import { useAuthStore } from '../stores/authStore';

interface User {
  id: string;
  email: string;
  role: 'patient' | 'doctor' | 'admin';
  firstName?: string;
  lastName?: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  // Allows setting session directly with tokens and profile (used after signup)
  setSession?: (tokenData: { access_token: string; refresh_token: string }, userProfile: any) => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: false,
  login: async () => false,
  logout: () => {},
});

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const { initialize } = useAuthStore();

  useEffect(() => {
    // Check for existing session
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      try {
        setUser(JSON.parse(userData));
      } catch (error) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
      }
    }
    
    setLoading(false);
  }, []);

  // Set session directly with tokens and user profile
  const setSession = (tokenData: { access_token: string; refresh_token: string }, userProfile: any) => {
    try {
      localStorage.setItem('token', tokenData.access_token);
      localStorage.setItem('refresh_token', tokenData.refresh_token);

      const userData = {
        id: userProfile.id,
        email: userProfile.email,
        role: userProfile.role,
        firstName: userProfile.first_name,
        lastName: userProfile.last_name,
      };

      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));

      // Ensure zustand store is initialized so the rest of the app sees auth state
      try {
        initialize();
      } catch (err) {
        console.warn('Failed to initialize auth store after setSession:', err);
      }
    } catch (err) {
      console.error('Failed to set session:', err);
    }
  };

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Login failed:', errorData);
        return false;
      }

      const tokenData = await response.json();
      
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
        const userData = {
          id: userProfile.id,
          email: userProfile.email,
          role: userProfile.role,
          firstName: userProfile.first_name,
          lastName: userProfile.last_name,
        };
        
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, setSession }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);