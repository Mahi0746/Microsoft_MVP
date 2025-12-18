// HealthSync AI - Future Simulator Screen
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

export default function FutureSimulatorScreen() {
  const handleUploadPhoto = () => {
    Alert.alert('Future Simulator', 'Photo upload feature will be implemented here');
  };

  const handleViewHistory = () => {
    Alert.alert('Future Simulator', 'Simulation history will be implemented here');
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#96CEB4', '#FFEAA7']}
        style={styles.header}
      >
        <Ionicons name="time" size={60} color="white" />
        <Text style={styles.headerTitle}>Future-You Simulator</Text>
        <Text style={styles.headerSubtitle}>
          See how your health choices affect your future
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        <TouchableOpacity style={styles.actionCard} onPress={handleUploadPhoto}>
          <LinearGradient
            colors={['#667eea', '#764ba2']}
            style={styles.actionGradient}
          >
            <Ionicons name="camera" size={40} color="white" />
            <Text style={styles.actionTitle}>Upload Photo</Text>
            <Text style={styles.actionSubtitle}>Start your future simulation</Text>
          </LinearGradient>
        </TouchableOpacity>

        <TouchableOpacity style={styles.actionCard} onPress={handleViewHistory}>
          <LinearGradient
            colors={['#f093fb', '#f5576c']}
            style={styles.actionGradient}
          >
            <Ionicons name="library" size={40} color="white" />
            <Text style={styles.actionTitle}>View History</Text>
            <Text style={styles.actionSubtitle}>See past simulations</Text>
          </LinearGradient>
        </TouchableOpacity>

        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>How it works:</Text>
          <View style={styles.infoItem}>
            <Text style={styles.infoNumber}>1</Text>
            <Text style={styles.infoText}>Upload your photo</Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoNumber}>2</Text>
            <Text style={styles.infoText}>AI ages your photo</Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoNumber}>3</Text>
            <Text style={styles.infoText}>Get health projections</Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoNumber}>4</Text>
            <Text style={styles.infoText}>Compare lifestyle scenarios</Text>
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
  actionCard: {
    marginBottom: 20,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  actionGradient: {
    padding: 30,
    alignItems: 'center',
  },
  actionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 15,
  },
  actionSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 5,
  },
  infoCard: {
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
  infoTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  infoNumber: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#96CEB4',
    color: 'white',
    textAlign: 'center',
    lineHeight: 24,
    fontSize: 14,
    fontWeight: 'bold',
    marginRight: 15,
  },
  infoText: {
    fontSize: 16,
    color: '#666',
  },
});