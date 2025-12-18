// HealthSync AI - Voice Doctor Screen
import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

export default function VoiceDoctorScreen() {
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleStartRecording = () => {
    setIsRecording(true);
    // TODO: Implement voice recording
    Alert.alert('Voice Recording', 'Voice recording feature will be implemented here');
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    setIsAnalyzing(true);
    
    // Simulate analysis
    setTimeout(() => {
      setIsAnalyzing(false);
      Alert.alert('Analysis Complete', 'Voice analysis results will be shown here');
    }, 3000);
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#FF6B6B', '#FF8E8E']}
        style={styles.header}
      >
        <Ionicons name="mic" size={60} color="white" />
        <Text style={styles.headerTitle}>AI Voice Doctor</Text>
        <Text style={styles.headerSubtitle}>
          Describe your symptoms and get instant AI analysis
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        <View style={styles.recordingArea}>
          {isAnalyzing ? (
            <View style={styles.analyzingContainer}>
              <View style={styles.pulseCircle} />
              <Text style={styles.analyzingText}>Analyzing your voice...</Text>
            </View>
          ) : (
            <TouchableOpacity
              style={[
                styles.recordButton,
                isRecording && styles.recordButtonActive
              ]}
              onPress={isRecording ? handleStopRecording : handleStartRecording}
            >
              <Ionicons
                name={isRecording ? 'stop' : 'mic'}
                size={60}
                color="white"
              />
            </TouchableOpacity>
          )}
        </View>

        <Text style={styles.instructionText}>
          {isRecording
            ? 'Tap to stop recording'
            : 'Tap the microphone to start recording'
          }
        </Text>

        <View style={styles.featuresContainer}>
          <Text style={styles.featuresTitle}>AI Analysis Includes:</Text>
          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
            <Text style={styles.featureText}>Voice stress detection</Text>
          </View>
          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
            <Text style={styles.featureText}>Symptom analysis</Text>
          </View>
          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
            <Text style={styles.featureText}>Risk assessment</Text>
          </View>
          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
            <Text style={styles.featureText}>Specialist recommendations</Text>
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    padding: 30,
    paddingTop: 60,
    alignItems: 'center',
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 15,
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    marginTop: 10,
  },
  content: {
    flex: 1,
    padding: 30,
    alignItems: 'center',
  },
  recordingArea: {
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 50,
  },
  recordButton: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#FF6B6B',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  recordButtonActive: {
    backgroundColor: '#F44336',
  },
  analyzingContainer: {
    alignItems: 'center',
  },
  pulseCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#FF6B6B',
    opacity: 0.6,
  },
  analyzingText: {
    fontSize: 16,
    color: '#666',
    marginTop: 20,
  },
  instructionText: {
    fontSize: 18,
    color: '#333',
    textAlign: 'center',
    marginBottom: 40,
  },
  featuresContainer: {
    width: '100%',
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  featuresTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  featureText: {
    fontSize: 16,
    color: '#666',
    marginLeft: 10,
  },
});