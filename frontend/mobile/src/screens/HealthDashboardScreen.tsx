// HealthSync AI - Health Dashboard Screen
import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useHealthStore } from '../stores/healthStore';

export default function HealthDashboardScreen() {
  const { healthMetrics, predictions } = useHealthStore();

  const getHealthScore = () => {
    if (predictions.length === 0) return 85;
    const totalRisk = predictions.reduce((sum, pred) => sum + pred.probability, 0);
    const avgRisk = totalRisk / predictions.length;
    return Math.max(0, Math.min(100, 100 - (avgRisk * 100)));
  };

  const healthScore = getHealthScore();
  const scoreColor = healthScore >= 80 ? '#4CAF50' : healthScore >= 60 ? '#FF9800' : '#F44336';

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <Ionicons name="heart" size={60} color="white" />
        <Text style={styles.headerTitle}>Health Dashboard</Text>
        <Text style={styles.headerSubtitle}>
          Your comprehensive health overview
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        {/* Health Score */}
        <View style={styles.scoreCard}>
          <Text style={styles.sectionTitle}>Health Score</Text>
          <View style={styles.scoreContainer}>
            <View style={styles.scoreCircle}>
              <Text style={[styles.scoreText, { color: scoreColor }]}>
                {Math.round(healthScore)}
              </Text>
              <Text style={styles.scoreLabel}>/ 100</Text>
            </View>
            <View style={styles.scoreDetails}>
              <View style={styles.scoreItem}>
                <Text style={styles.scoreItemLabel}>Metrics Tracked</Text>
                <Text style={styles.scoreItemValue}>{healthMetrics.length}</Text>
              </View>
              <View style={styles.scoreItem}>
                <Text style={styles.scoreItemLabel}>Risk Factors</Text>
                <Text style={styles.scoreItemValue}>{predictions.length}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* Quick Actions */}
        <View style={styles.actionsContainer}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.actionsGrid}>
            <TouchableOpacity style={styles.actionCard}>
              <Ionicons name="add-circle" size={30} color="#4CAF50" />
              <Text style={styles.actionText}>Add Metric</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionCard}>
              <Ionicons name="analytics" size={30} color="#2196F3" />
              <Text style={styles.actionText}>View Trends</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionCard}>
              <Ionicons name="people" size={30} color="#FF9800" />
              <Text style={styles.actionText}>Family Graph</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionCard}>
              <Ionicons name="warning" size={30} color="#F44336" />
              <Text style={styles.actionText}>Risk Factors</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Recent Metrics */}
        <View style={styles.metricsContainer}>
          <Text style={styles.sectionTitle}>Recent Metrics</Text>
          {healthMetrics.length > 0 ? (
            healthMetrics.slice(0, 3).map((metric, index) => (
              <View key={index} style={styles.metricItem}>
                <Ionicons name="pulse" size={24} color="#667eea" />
                <View style={styles.metricContent}>
                  <Text style={styles.metricName}>{metric.metricType}</Text>
                  <Text style={styles.metricValue}>{metric.value} {metric.unit}</Text>
                </View>
                <Text style={styles.metricDate}>
                  {new Date(metric.measuredAt).toLocaleDateString()}
                </Text>
              </View>
            ))
          ) : (
            <View style={styles.emptyState}>
              <Ionicons name="medical" size={40} color="#ccc" />
              <Text style={styles.emptyText}>No health metrics yet</Text>
              <Text style={styles.emptySubtext}>Start tracking your health data</Text>
            </View>
          )}
        </View>

        {/* Risk Factors */}
        {predictions.length > 0 && (
          <View style={styles.risksContainer}>
            <Text style={styles.sectionTitle}>Health Risks</Text>
            {predictions.slice(0, 3).map((prediction, index) => (
              <View key={index} style={styles.riskItem}>
                <Text style={styles.riskName}>{prediction.disease}</Text>
                <View style={styles.riskBar}>
                  <View 
                    style={[
                      styles.riskProgress, 
                      { 
                        width: `${prediction.probability * 100}%`,
                        backgroundColor: prediction.probability > 0.7 ? '#F44336' : 
                                       prediction.probability > 0.4 ? '#FF9800' : '#4CAF50'
                      }
                    ]} 
                  />
                </View>
                <Text style={styles.riskPercentage}>
                  {Math.round(prediction.probability * 100)}%
                </Text>
              </View>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
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
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  scoreCard: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  scoreCircle: {
    alignItems: 'center',
    marginRight: 20,
  },
  scoreText: {
    fontSize: 36,
    fontWeight: 'bold',
  },
  scoreLabel: {
    fontSize: 16,
    color: '#666',
  },
  scoreDetails: {
    flex: 1,
  },
  scoreItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  scoreItemLabel: {
    fontSize: 14,
    color: '#666',
  },
  scoreItemValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  actionsContainer: {
    marginBottom: 20,
  },
  actionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionCard: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    width: '48%',
    alignItems: 'center',
    marginBottom: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  actionText: {
    fontSize: 14,
    color: '#333',
    marginTop: 10,
    textAlign: 'center',
  },
  metricsContainer: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  metricItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  metricContent: {
    flex: 1,
    marginLeft: 15,
  },
  metricName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    textTransform: 'capitalize',
  },
  metricValue: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  metricDate: {
    fontSize: 12,
    color: '#999',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 30,
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    marginTop: 10,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 5,
  },
  risksContainer: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  riskItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  riskName: {
    fontSize: 14,
    color: '#333',
    width: 100,
    textTransform: 'capitalize',
  },
  riskBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    marginHorizontal: 10,
  },
  riskProgress: {
    height: '100%',
    borderRadius: 4,
  },
  riskPercentage: {
    fontSize: 12,
    color: '#666',
    width: 35,
    textAlign: 'right',
  },
});