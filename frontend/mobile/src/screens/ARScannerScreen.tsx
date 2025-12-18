// HealthSync AI - AR Scanner Screen
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

export default function ARScannerScreen() {
  const handleScanPrescription = () => {
    Alert.alert('AR Scanner', 'Prescription scanning feature will be implemented here');
  };

  const handleScanLabReport = () => {
    Alert.alert('AR Scanner', 'Lab report scanning feature will be implemented here');
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#4ECDC4', '#44A08D']}
        style={styles.header}
      >
        <Ionicons name="camera" size={60} color="white" />
        <Text style={styles.headerTitle}>AR Medical Scanner</Text>
        <Text style={styles.headerSubtitle}>
          Scan prescriptions and medical documents with AI
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        <TouchableOpacity style={styles.scanOption} onPress={handleScanPrescription}>
          <LinearGradient
            colors={['#667eea', '#764ba2']}
            style={styles.scanGradient}
          >
            <Ionicons name="document-text" size={40} color="white" />
            <Text style={styles.scanTitle}>Scan Prescription</Text>
            <Text style={styles.scanSubtitle}>Extract medication information</Text>
          </LinearGradient>
        </TouchableOpacity>

        <TouchableOpacity style={styles.scanOption} onPress={handleScanLabReport}>
          <LinearGradient
            colors={['#f093fb', '#f5576c']}
            style={styles.scanGradient}
          >
            <Ionicons name="analytics" size={40} color="white" />
            <Text style={styles.scanTitle}>Scan Lab Report</Text>
            <Text style={styles.scanSubtitle}>Analyze test results</Text>
          </LinearGradient>
        </TouchableOpacity>
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
    justifyContent: 'center',
  },
  scanOption: {
    marginBottom: 20,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  scanGradient: {
    padding: 30,
    alignItems: 'center',
  },
  scanTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 15,
  },
  scanSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 5,
  },
});