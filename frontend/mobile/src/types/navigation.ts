// HealthSync AI - Navigation Types
export type RootStackParamList = {
  Auth: undefined;
  MainTabs: undefined;
  TherapyGame: undefined;
  FutureSimulator: undefined;
  DoctorMarketplace: undefined;
};

export type MainTabParamList = {
  Home: undefined;
  Health: undefined;
  VoiceDoctor: undefined;
  ARScanner: undefined;
  Profile: undefined;
};

export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
  ForgotPassword: undefined;
};

export type HealthStackParamList = {
  Dashboard: undefined;
  Metrics: undefined;
  Predictions: undefined;
  FamilyGraph: undefined;
};

export type VoiceDoctorStackParamList = {
  Consultation: undefined;
  History: undefined;
  Results: { sessionId: string };
};

export type ARScannerStackParamList = {
  Scanner: undefined;
  Results: { scanId: string };
  History: undefined;
};

export type TherapyGameStackParamList = {
  Menu: undefined;
  Exercise: { exerciseType: string };
  Progress: undefined;
  Leaderboard: undefined;
};

export type FutureSimulatorStackParamList = {
  Upload: undefined;
  Processing: { imageId: string };
  Results: { simulationId: string };
  History: undefined;
  Compare: undefined;
};

export type DoctorMarketplaceStackParamList = {
  Search: undefined;
  DoctorProfile: { doctorId: string };
  Appointment: { doctorId: string };
  Bidding: { appointmentId: string };
  Booking: { appointmentId: string; bidId: string };
};

export type ProfileStackParamList = {
  Overview: undefined;
  Settings: undefined;
  HealthData: undefined;
  Privacy: undefined;
  Support: undefined;
};