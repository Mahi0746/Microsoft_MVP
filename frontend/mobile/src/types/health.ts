// HealthSync AI - Health Data Types
export interface User {
  id: string;
  email: string;
  role: 'patient' | 'doctor' | 'admin';
  firstName?: string;
  lastName?: string;
  dateOfBirth?: string;
  gender?: string;
  phone?: string;
  profileImageUrl?: string;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface HealthMetric {
  id: string;
  userId: string;
  metricType: 'blood_pressure' | 'heart_rate' | 'weight' | 'bmi' | 'blood_sugar' | 'temperature' | 'oxygen_saturation';
  value: number;
  unit: string;
  measuredAt: string;
  notes?: string;
}

export interface Symptom {
  id: string;
  userId: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  bodyPart?: string;
  duration?: string;
  triggers?: string[];
  timestamp: string;
}

export interface HealthPrediction {
  id: string;
  userId: string;
  disease: string;
  probability: number;
  confidenceScore: number;
  riskFactors: string[];
  recommendations: string[];
  createdAt: string;
  expiresAt: string;
}

export interface VoiceAnalysis {
  transcript: string;
  voiceFeatures: {
    stressLevel: number;
    speechRate: number;
    pauseFrequency: number;
    energyLevel: number;
    confidenceLevel: number;
  };
  assessment: {
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    urgencyFlag: boolean;
    recommendedActions: string[];
    suggestedSpecialists: string[];
    confidenceScore: number;
  };
}

export interface ARScanResult {
  id: string;
  userId: string;
  imageUrl: string;
  scanType: 'prescription' | 'lab_report' | 'medical_document';
  ocrResults: {
    text: string;
    confidence: number;
    extractedMedications?: MedicationInfo[];
  };
  imageAnalysis: {
    description: string;
    detectedObjects: string[];
    confidence: number;
  };
  warnings: string[];
  createdAt: string;
}

export interface MedicationInfo {
  name: string;
  dosage: string;
  frequency: string;
  instructions: string;
  confidence: number;
}

export interface TherapySession {
  id: string;
  userId: string;
  exerciseType: 'arm_raises' | 'knee_bends' | 'neck_rotations' | 'shoulder_rolls';
  durationSeconds: number;
  pointsEarned: number;
  accuracyScore: number;
  painDetected: boolean;
  painLevel: number;
  movementData: {
    keyPoints: number[][];
    angles: number[];
    correctForm: boolean[];
  };
  createdAt: string;
}

export interface TherapyProgress {
  id: string;
  userId: string;
  weekStartDate: string;
  totalSessions: number;
  totalPoints: number;
  averageAccuracy: number;
  streakDays: number;
  exerciseBreakdown: {
    [key: string]: {
      sessions: number;
      points: number;
      accuracy: number;
    };
  };
  achievements: Achievement[];
}

export interface Achievement {
  id: string;
  name: string;
  description: string;
  iconName: string;
  unlockedAt: string;
  category: 'consistency' | 'accuracy' | 'endurance' | 'milestone';
}

export interface FutureSimulation {
  id: string;
  userId: string;
  originalImageUrl: string;
  agedImageUrl: string;
  ageProgressionYears: number;
  healthProjections: {
    lifeExpectancy: number;
    diseaseRisks: {
      [disease: string]: {
        probability: number;
        riskLevel: 'low' | 'medium' | 'high';
        potentialComplications: string[];
      };
    };
  };
  lifestyleScenarios: {
    current: LifestyleScenario;
    improved: LifestyleScenario;
    declined: LifestyleScenario;
  };
  aiNarrative: {
    currentPath: string;
    improvedPath: string;
  };
  recommendations: string[];
  createdAt: string;
}

export interface LifestyleScenario {
  description: string;
  lifeExpectancy: number;
  diseaseRisks: { [disease: string]: number };
  qualityOfLife: number;
  recommendations: string[];
}

export interface Doctor {
  id: string;
  userId: string;
  firstName: string;
  lastName: string;
  specialization: string;
  subSpecializations: string[];
  yearsExperience: number;
  rating: number;
  totalReviews: number;
  baseConsultationFee: number;
  bio?: string;
  languages: string[];
  isVerified: boolean;
  isAcceptingPatients: boolean;
  location?: string;
  consultationTypes: ('video' | 'audio' | 'chat' | 'in_person')[];
}

export interface Appointment {
  id: string;
  patientId: string;
  doctorId?: string;
  status: 'pending' | 'bidding' | 'confirmed' | 'completed' | 'cancelled';
  symptomsSummary: string;
  urgencyLevel: 'low' | 'medium' | 'high' | 'critical';
  preferredDate?: string;
  consultationType: 'video' | 'audio' | 'chat' | 'in_person';
  finalFee?: number;
  scheduledAt?: string;
  createdAt: string;
}

export interface AppointmentBid {
  id: string;
  appointmentId: string;
  doctorId: string;
  doctorName: string;
  specialization: string;
  bidAmount: number;
  estimatedDuration: number;
  availableSlots: string[];
  message?: string;
  isSelected: boolean;
  createdAt: string;
}

export interface FamilyMember {
  id: string;
  relation: string;
  healthConditions: string[];
  ageOfOnset: { [condition: string]: number };
  isDeceased: boolean;
  ageAtDeath?: number;
  causeOfDeath?: string;
}

export interface FamilyHealthGraph {
  userId: string;
  familyMembers: FamilyMember[];
  inheritedRisks: {
    [condition: string]: {
      probability: number;
      confidence: number;
      contributingRelatives: string[];
    };
  };
  updatedAt: string;
}