// HealthSync AI - Health Data Store (Zustand)
import { create } from 'zustand';
import { apiService } from '../services/ApiService';
import { 
  HealthMetric, 
  HealthPrediction, 
  TherapyProgress, 
  FamilyHealthGraph,
  VoiceAnalysis,
  ARScanResult,
  FutureSimulation
} from '../types/health';

interface HealthState {
  // Data
  healthMetrics: HealthMetric[];
  predictions: HealthPrediction[];
  therapyProgress: TherapyProgress | null;
  familyGraph: FamilyHealthGraph | null;
  voiceAnalyses: VoiceAnalysis[];
  arScans: ARScanResult[];
  futureSimulations: FutureSimulation[];
  
  // UI State
  isLoading: boolean;
  error: string | null;
  
  // Actions
  initializeHealth: (userId: string) => Promise<void>;
  addHealthMetric: (metric: Omit<HealthMetric, 'id' | 'userId'>) => Promise<boolean>;
  refreshPredictions: () => Promise<void>;
  updateTherapyProgress: () => Promise<void>;
  addVoiceAnalysis: (analysis: VoiceAnalysis) => void;
  addARScan: (scan: ARScanResult) => void;
  addFutureSimulation: (simulation: FutureSimulation) => void;
  clearError: () => void;
}

export const useHealthStore = create<HealthState>((set, get) => ({
  // Initial state
  healthMetrics: [],
  predictions: [],
  therapyProgress: null,
  familyGraph: null,
  voiceAnalyses: [],
  arScans: [],
  futureSimulations: [],
  isLoading: false,
  error: null,

  initializeHealth: async (userId: string) => {
    set({ isLoading: true, error: null });
    
    try {
      // Load health metrics
      const metricsResponse = await apiService.getHealthMetrics(userId);
      if (metricsResponse.success) {
        set({ healthMetrics: metricsResponse.data || [] });
      }

      // Load predictions
      const predictionsResponse = await apiService.getHealthPredictions();
      if (predictionsResponse.success) {
        set({ predictions: predictionsResponse.data || [] });
      }

      // Load therapy progress
      const therapyResponse = await apiService.getTherapyProgress();
      if (therapyResponse.success) {
        set({ therapyProgress: therapyResponse.data });
      }

      // Load family graph
      const familyResponse = await apiService.getFamilyHealthGraph();
      if (familyResponse.success) {
        set({ familyGraph: familyResponse.data });
      }

      // Load AR scan history
      const scanResponse = await apiService.getScanHistory();
      if (scanResponse.success) {
        set({ arScans: scanResponse.data || [] });
      }

      // Load future simulations
      const simulationResponse = await apiService.getSimulationHistory();
      if (simulationResponse.success) {
        set({ futureSimulations: simulationResponse.data || [] });
      }

      set({ isLoading: false });
    } catch (error) {
      set({
        error: 'Failed to load health data',
        isLoading: false,
      });
    }
  },

  addHealthMetric: async (metric) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.addHealthMetric(metric);
      
      if (response.success) {
        const newMetric = response.data;
        set(state => ({
          healthMetrics: [...state.healthMetrics, newMetric],
          isLoading: false,
        }));
        return true;
      } else {
        set({
          error: response.error || 'Failed to add health metric',
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

  refreshPredictions: async () => {
    try {
      const response = await apiService.getHealthPredictions();
      
      if (response.success) {
        set({ predictions: response.data || [] });
      }
    } catch (error) {
      console.error('Failed to refresh predictions:', error);
    }
  },

  updateTherapyProgress: async () => {
    try {
      const response = await apiService.getTherapyProgress();
      
      if (response.success) {
        set({ therapyProgress: response.data });
      }
    } catch (error) {
      console.error('Failed to update therapy progress:', error);
    }
  },

  addVoiceAnalysis: (analysis: VoiceAnalysis) => {
    set(state => ({
      voiceAnalyses: [analysis, ...state.voiceAnalyses].slice(0, 50), // Keep last 50
    }));
  },

  addARScan: (scan: ARScanResult) => {
    set(state => ({
      arScans: [scan, ...state.arScans],
    }));
  },

  addFutureSimulation: (simulation: FutureSimulation) => {
    set(state => ({
      futureSimulations: [simulation, ...state.futureSimulations],
    }));
  },

  clearError: () => {
    set({ error: null });
  },
}));