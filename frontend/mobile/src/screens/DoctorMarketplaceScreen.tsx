// HealthSync AI - Doctor Marketplace Screen
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

export default function DoctorMarketplaceScreen() {
  const handleSearchDoctors = () => {
    Alert.alert('Doctor Marketplace', 'Doctor search feature will be implemented here');
  };

  const handleSymptomAnalysis = () => {
    Alert.alert('Doctor Marketplace', 'Symptom analysis feature will be implemented here');
  };

  const handleViewAppointments = () => {
    Alert.alert('Doctor Marketplace', 'Appointments view will be implemented here');
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#FFEAA7', '#DDA0DD']}
        style={styles.header}
      >
        <Ionicons name="people" size={60} color="white" />
        <Text style={styles.headerTitle}>Doctor Marketplace</Text>
        <Text style={styles.headerSubtitle}>
          Find specialists and book appointments with AI matching
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        <TouchableOpacity style={styles.featureCard} onPress={handleSymptomAnalysis}>
          <Ionicons name="medical" size={30} color="#FF6B6B" />
          <View style={styles.featureContent}>
            <Text style={styles.featureTitle}>Symptom Analysis</Text>
            <Text style={styles.featureSubtitle}>AI-powered specialist matching</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard} onPress={handleSearchDoctors}>
          <Ionicons name="search" size={30} color="#4ECDC4" />
          <View style={styles.featureContent}>
            <Text style={styles.featureTitle}>Search Doctors</Text>
            <Text style={styles.featureSubtitle}>Browse by specialty and location</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard} onPress={handleViewAppointments}>
          <Ionicons name="calendar" size={30} color="#45B7D1" />
          <View style={styles.featureContent}>
            <Text style={styles.featureTitle}>My Appointments</Text>
            <Text style={styles.featureSubtitle}>View and manage bookings</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>

        <View style={styles.statsCard}>
          <Text style={styles.statsTitle}>Marketplace Stats</Text>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>1,250+</Text>
              <Text style={styles.statLabel}>Doctors</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>50+</Text>
              <Text style={styles.statLabel}>Specialties</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>4.8★</Text>
              <Text style={styles.statLabel}>Avg Rating</Text>
            </View>
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
  },
  featureCard: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    flexDirection: 'row',
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  featureContent: {
    flex: 1,
    marginLeft: 15,
  },
  featureTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  featureSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  statsCard: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    marginTop: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFEAA7',
  },
  statLabel: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
});