// HealthSync AI - Therapy Game Screen
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

export default function TherapyGameScreen() {
  const handleStartExercise = (exerciseType: string) => {
    Alert.alert('Therapy Game', `${exerciseType} exercise will be implemented here`);
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#45B7D1', '#96CEB4']}
        style={styles.header}
      >
        <Ionicons name="game-controller" size={60} color="white" />
        <Text style={styles.headerTitle}>Pain-to-Game Therapy</Text>
        <Text style={styles.headerSubtitle}>
          Gamified rehabilitation exercises with AI motion tracking
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        <TouchableOpacity 
          style={styles.exerciseCard} 
          onPress={() => handleStartExercise('Arm Raises')}
        >
          <Ionicons name="fitness" size={30} color="#45B7D1" />
          <Text style={styles.exerciseTitle}>Arm Raises</Text>
          <Text style={styles.exerciseSubtitle}>Upper body strength</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.exerciseCard} 
          onPress={() => handleStartExercise('Knee Bends')}
        >
          <Ionicons name="walk" size={30} color="#96CEB4" />
          <Text style={styles.exerciseTitle}>Knee Bends</Text>
          <Text style={styles.exerciseSubtitle}>Lower body mobility</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.exerciseCard} 
          onPress={() => handleStartExercise('Neck Rotations')}
        >
          <Ionicons name="refresh" size={30} color="#FFEAA7" />
          <Text style={styles.exerciseTitle}>Neck Rotations</Text>
          <Text style={styles.exerciseSubtitle}>Neck flexibility</Text>
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
  },
  exerciseCard: {
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
  exerciseTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 15,
    flex: 1,
  },
  exerciseSubtitle: {
    fontSize: 14,
    color: '#666',
    marginLeft: 15,
  },
});