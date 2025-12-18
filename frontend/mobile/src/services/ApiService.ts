// HealthSync AI - API Service
import { supabase } from './AuthService';

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

class ApiService {
  private async getAuthHeaders(): Promise<HeadersInit> {
    const { data: { session } } = await supabase.auth.getSession();
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (session?.access_token) {
      headers['Authorization'] = `Bearer ${session.access_token}`;
    }

    return headers;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const headers = await this.getAuthHeaders();
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
          ...headers,
          ...options.headers,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: data.message || data.detail || 'Request failed',
        };
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
      };
    }
  }

  // Health Metrics
  async getHealthMetrics(userId: string): Promise<ApiResponse> {
    return this.request(`/api/v1/health/metrics/${userId}`);
  }

  async addHealthMetric(metric: any): Promise<ApiResponse> {
    return this.request('/api/v1/health/metrics', {
      method: 'POST',
      body: JSON.stringify(metric),
    });
  }

  // Voice AI Doctor
  async startVoiceSession(): Promise<ApiResponse> {
    return this.request('/api/v1/voice/session/start', {
      method: 'POST',
    });
  }

  async uploadVoiceChunk(sessionId: string, audioData: string): Promise<ApiResponse> {
    return this.request('/api/v1/voice/upload-chunk', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        audio_data: audioData,
      }),
    });
  }

  async getVoiceAnalysis(sessionId: string): Promise<ApiResponse> {
    return this.request(`/api/v1/voice/analysis/${sessionId}`);
  }

  // AR Scanner
  async uploadScanImage(imageUri: string, scanType: string): Promise<ApiResponse> {
    const formData = new FormData();
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'scan.jpg',
    } as any);
    formData.append('scan_type', scanType);

    const headers = await this.getAuthHeaders();
    delete (headers as any)['Content-Type']; // Let browser set multipart boundary

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ar-scanner/scan`, {
        method: 'POST',
        headers,
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: data.message || 'Upload failed',
        };
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      return {
        success: false,
        error: 'Network error',
      };
    }
  }

  async getScanHistory(): Promise<ApiResponse> {
    return this.request('/api/v1/ar-scanner/history');
  }

  // Therapy Game
  async startTherapySession(exerciseType: string): Promise<ApiResponse> {
    return this.request('/api/v1/therapy-game/session/start', {
      method: 'POST',
      body: JSON.stringify({ exercise_type: exerciseType }),
    });
  }

  async submitTherapySession(sessionData: any): Promise<ApiResponse> {
    return this.request('/api/v1/therapy-game/session/complete', {
      method: 'POST',
      body: JSON.stringify(sessionData),
    });
  }

  async getTherapyProgress(): Promise<ApiResponse> {
    return this.request('/api/v1/therapy-game/progress');
  }

  async getTherapyLeaderboard(): Promise<ApiResponse> {
    return this.request('/api/v1/therapy-game/leaderboard');
  }

  // Future Simulator
  async uploadFutureSimImage(imageUri: string): Promise<ApiResponse> {
    const formData = new FormData();
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'future_sim.jpg',
    } as any);

    const headers = await this.getAuthHeaders();
    delete (headers as any)['Content-Type'];

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/future-simulator/upload-image`, {
        method: 'POST',
        headers,
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: data.message || 'Upload failed',
        };
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      return {
        success: false,
        error: 'Network error',
      };
    }
  }

  async generateAgeProgression(imagePath: string, targetAge: number): Promise<ApiResponse> {
    return this.request('/api/v1/future-simulator/age-progression', {
      method: 'POST',
      body: JSON.stringify({
        image_path: imagePath,
        target_age_years: targetAge,
      }),
    });
  }

  async generateHealthProjections(targetAge: number, scenario: string): Promise<ApiResponse> {
    return this.request('/api/v1/future-simulator/health-projections', {
      method: 'POST',
      body: JSON.stringify({
        target_age_years: targetAge,
        lifestyle_scenario: scenario,
      }),
    });
  }

  async getSimulationHistory(): Promise<ApiResponse> {
    return this.request('/api/v1/future-simulator/history');
  }

  async compareLifestyleScenarios(targetAge: number): Promise<ApiResponse> {
    return this.request(`/api/v1/future-simulator/compare-scenarios?target_age_years=${targetAge}`, {
      method: 'POST',
    });
  }

  // Doctor Marketplace
  async searchDoctors(params: any): Promise<ApiResponse> {
    const queryString = new URLSearchParams(params).toString();
    return this.request(`/api/v1/doctors/search?${queryString}`);
  }

  async matchSpecialists(symptoms: string, urgency: string): Promise<ApiResponse> {
    return this.request('/api/v1/doctors/match-specialists', {
      method: 'POST',
      body: JSON.stringify({
        symptoms_summary: symptoms,
        urgency_level: urgency,
        consultation_type: 'video',
      }),
    });
  }

  async createAppointment(appointmentData: any): Promise<ApiResponse> {
    return this.request('/api/v1/doctors/appointments', {
      method: 'POST',
      body: JSON.stringify(appointmentData),
    });
  }

  async getAppointmentBids(appointmentId: string): Promise<ApiResponse> {
    return this.request(`/api/v1/doctors/appointments/${appointmentId}/bids`);
  }

  async acceptBid(appointmentId: string, bidId: string): Promise<ApiResponse> {
    return this.request(`/api/v1/doctors/appointments/${appointmentId}/accept-bid/${bidId}`, {
      method: 'PUT',
    });
  }

  async getDoctorProfile(doctorId: string): Promise<ApiResponse> {
    return this.request(`/api/v1/doctors/${doctorId}/profile`);
  }

  // Health Predictions
  async getHealthPredictions(): Promise<ApiResponse> {
    return this.request('/api/v1/health/predictions');
  }

  async updateHealthData(healthData: any): Promise<ApiResponse> {
    return this.request('/api/v1/health/metrics', {
      method: 'POST',
      body: JSON.stringify(healthData),
    });
  }

  // Family Health Graph
  async getFamilyHealthGraph(): Promise<ApiResponse> {
    return this.request('/api/v1/health/family-graph');
  }

  async updateFamilyMember(memberData: any): Promise<ApiResponse> {
    return this.request('/api/v1/health/family-member', {
      method: 'POST',
      body: JSON.stringify(memberData),
    });
  }

  // Notifications
  async getNotifications(): Promise<ApiResponse> {
    return this.request('/api/v1/notifications');
  }

  async markNotificationRead(notificationId: string): Promise<ApiResponse> {
    return this.request(`/api/v1/notifications/${notificationId}/read`, {
      method: 'PUT',
    });
  }

  // Health Insights
  async getHealthInsights(): Promise<ApiResponse> {
    return this.request('/api/v1/future-simulator/health-insights');
  }

  // Marketplace Stats
  async getMarketplaceStats(): Promise<ApiResponse> {
    return this.request('/api/v1/doctors/marketplace/stats');
  }
}

export const apiService = new ApiService();