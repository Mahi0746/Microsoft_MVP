// HealthSync AI - Home Screen
import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  RefreshControl,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { useAuthStore } from '../stores/authStore';
import { useHealthStore } from '../stores/healthStore';
import { apiService } from '../services/ApiService';

const { width } = Dimensions.get('window');

interface QuickAction {
  id: string;
  title: string;
  subtitle: string;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  screen: string;
}

const quickActions: QuickAction[] = [
  {
    id: 'voice',
    title: 'AI Doctor',
    subtitle: 'Voice consultation',
    icon: 'mic-circle',
    color: '#FF6B6B',
    screen: 'VoiceDoctor',
  },
  {
    id: 'ar',
    title: 'AR Scanner',
    subtitle: 'Scan prescriptions',
    icon: 'camera',
    color: '#4ECDC4',
    screen: 'ARScanner',
  },
  {
    id: 'therapy',
    title: 'Therapy Game',
    subtitle: 'Rehabilitation',
    icon: 'game-controller',
    color: '#45B7D1',
    screen: 'TherapyGame',
  },
  {
    id: 'future',
    title: 'Future-You',
    subtitle: 'Health simulation',
    icon: 'time',
    color: '#96CEB4',
    screen: 'FutureSimulator',
  },
  {
    id: 'doctors',
    title: 'Find Doctors',
    subtitle: 'Marketplace',
    icon: 'people',
    color: '#FFEAA7',
    screen: 'DoctorMarketplace',
  },
  {
    id: 'health',
    title: 'Health Data',
    subtitle: 'Metrics & trends',
    icon: 'analytics',
    color: '#DDA0DD',
    screen: 'Health',
  },
];

export default function HomeScreen() {
  const navigation = useNavigation();
  const { user } = useAuthStore();
  const { predictions, healthMetrics, isLoading, initializeHealth } = useHealthStore();
  const [refreshing, setRefreshing] = useState(false);
  const [healthInsights, setHealthInsights] = useState<any>(null);

  useEffect(() => {
    if (user) {
      loadHealthInsights();
    }
  }, [user]);

  const loadHealthInsights = async () => {
    try {
      const response = await apiService.getHealthInsights();
      if (response.success) {
        setHealthInsights(response.data);
      }
    } catch (error) {
      console.error('Failed to load health insights:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    if (user) {
      await initializeHealth(user.id);
      await loadHealthInsights();
    }
    setRefreshing(false);
  };

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  const getHealthScore = () => {
    if (predictions.length === 0) return 85; // Default score
    
    // Calculate health score based on predictions
    const totalRisk = predictions.reduce((sum, pred) => sum + pred.probability, 0);
    const avgRisk = totalRisk / predictions.length;
    return Math.max(0, Math.min(100, 100 - (avgRisk * 100)));
  };

  const renderQuickAction = (action: QuickAction) => (
    <TouchableOpacity
      key={action.id}
      style={styles.actionCard}
      onPress={() => navigation.navigate(action.screen as never)}
    >
      <LinearGradient
        colors={[action.color, `${action.color}80`]}
        style={styles.actionGradient}
      >
        <Ionicons name={action.icon} size={32} color="white" />
        <Text style={styles.actionTitle}>{action.title}</Text>
        <Text style={styles.actionSubtitle}>{action.subtitle}</Text>
      </LinearGradient>
    </TouchableOpacity>
  );

  const renderHealthSummary = () => {
    const healthScore = getHealthScore();
    const scoreColor = healthScore >= 80 ? '#4CAF50' : healthScore >= 60 ? '#FF9800' : '#F44336';

    return (
      <View style={styles.healthSummary}>
        <Text style={styles.sectionTitle}>Health Overview</Text>
        
        <View style={styles.healthScoreCard}>
          <View style={styles.scoreCircle}>
            <Text style={[styles.scoreText, { color: scoreColor }]}>
              {Math.round(healthScore)}
            </Text>
            <Text style={styles.scoreLabel}>Health Score</Text>
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

        {predictions.length > 0 && (
          <View style={styles.riskFactors}>
            <Text style={styles.riskTitle}>Top Health Risks</Text>
            {predictions.slice(0, 3).map((prediction, index) => (
              <View key={prediction.id} style={styles.riskItem}>
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
    );
  };

  const renderInsights = () => {
    if (!healthInsights || !healthInsights.insights) return null;

    return (
      <View style={styles.insightsSection}>
        <Text style={styles.sectionTitle}>Health Insights</Text>
        {healthInsights.insights.slice(0, 2).map((insight: any, index: number) => (
          <View key={index} style={styles.insightCard}>
            <Ionicons 
              name="bulb" 
              size={24} 
              color="#FF9800" 
              style={styles.insightIcon} 
            />
            <View style={styles.insightContent}>
              <Text style={styles.insightTitle}>{insight.title}</Text>
              <Text style={styles.insightDescription}>{insight.description}</Text>
              {insight.action && (
                <Text style={styles.insightAction}>💡 {insight.action}</Text>
              )}
            </View>
          </View>
        ))}
      </View>
    );
  };

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Header */}
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <Text style={styles.greeting}>
          {getGreeting()}, {user?.firstName || 'there'}!
        </Text>
        <Text style={styles.subtitle}>
          How are you feeling today?
        </Text>
      </LinearGradient>

      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.actionsGrid}>
          {quickActions.map(renderQuickAction)}
        </View>
      </View>

      {/* Health Summary */}
      {renderHealthSummary()}

      {/* Health Insights */}
      {renderInsights()}

      {/* Recent Activity */}
      <View style={styles.recentActivity}>
        <Text style={styles.sectionTitle}>Recent Activity</Text>
        
        <TouchableOpacity 
          style={styles.activityItem}
          onPress={() => navigation.navigate('Health' as never)}
        >
          <Ionicons name="heart" size={24} color="#FF6B6B" />
          <View style={styles.activityContent}>
            <Text style={styles.activityTitle}>Health Metrics</Text>
            <Text style={styles.activitySubtitle}>
              {healthMetrics.length} metrics tracked
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.activityItem}
          onPress={() => navigation.navigate('VoiceDoctor' as never)}
        >
          <Ionicons name="mic" size={24} color="#4ECDC4" />
          <View style={styles.activityContent}>
            <Text style={styles.activityTitle}>AI Consultations</Text>
            <Text style={styles.activitySubtitle}>
              Voice analysis available
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>
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
    padding: 20,
    paddingTop: 60,
    paddingBottom: 30,
  },
  greeting: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  quickActions: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  actionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionCard: {
    width: (width - 60) / 2,
    marginBottom: 15,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  actionGradient: {
    padding: 20,
    alignItems: 'center',
    minHeight: 120,
    justifyContent: 'center',
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 10,
    textAlign: 'center',
  },
  actionSubtitle: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 5,
    textAlign: 'center',
  },
  healthSummary: {
    padding: 20,
    paddingTop: 0,
  },
  healthScoreCard: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
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
    fontSize: 12,
    color: '#666',
    marginTop: 5,
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
  riskFactors: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  riskTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  riskItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  riskName: {
    fontSize: 14,
    color: '#333',
    width: 80,
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
  insightsSection: {
    padding: 20,
    paddingTop: 0,
  },
  insightCard: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  insightIcon: {
    marginRight: 15,
    marginTop: 2,
  },
  insightContent: {
    flex: 1,
  },
  insightTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  insightDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  insightAction: {
    fontSize: 12,
    color: '#FF9800',
    marginTop: 5,
    fontStyle: 'italic',
  },
  recentActivity: {
    padding: 20,
    paddingTop: 0,
  },
  activityItem: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  activityContent: {
    flex: 1,
    marginLeft: 15,
  },
  activityTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  activitySubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
});