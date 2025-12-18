// HealthSync AI - React Native Mobile App Entry Point
import React, { useEffect, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import * as SplashScreen from 'expo-splash-screen';
import * as Font from 'expo-font';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

// Screens
import AuthScreen from './src/screens/AuthScreen';
import HomeScreen from './src/screens/HomeScreen';
import VoiceDoctorScreen from './src/screens/VoiceDoctorScreen';
import ARScannerScreen from './src/screens/ARScannerScreen';
import TherapyGameScreen from './src/screens/TherapyGameScreen';
import FutureSimulatorScreen from './src/screens/FutureSimulatorScreen';
import DoctorMarketplaceScreen from './src/screens/DoctorMarketplaceScreen';
import HealthDashboardScreen from './src/screens/HealthDashboardScreen';
import ProfileScreen from './src/screens/ProfileScreen';

// Services
import { AuthService } from './src/services/AuthService';
import { useAuthStore } from './src/stores/authStore';
import { useHealthStore } from './src/stores/healthStore';

// Components
import LoadingScreen from './src/components/LoadingScreen';
import { ThemeProvider } from './src/contexts/ThemeContext';

// Types
import { RootStackParamList, MainTabParamList } from './src/types/navigation';

// Keep splash screen visible while loading
SplashScreen.preventAutoHideAsync();

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();

// Main Tab Navigator
function MainTabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'Health':
              iconName = focused ? 'heart' : 'heart-outline';
              break;
            case 'VoiceDoctor':
              iconName = focused ? 'mic' : 'mic-outline';
              break;
            case 'ARScanner':
              iconName = focused ? 'camera' : 'camera-outline';
              break;
            case 'Profile':
              iconName = focused ? 'person' : 'person-outline';
              break;
            default:
              iconName = 'help-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      })}
    >
      <Tab.Screen 
        name="Home" 
        component={HomeScreen}
        options={{ title: 'Home' }}
      />
      <Tab.Screen 
        name="Health" 
        component={HealthDashboardScreen}
        options={{ title: 'Health' }}
      />
      <Tab.Screen 
        name="VoiceDoctor" 
        component={VoiceDoctorScreen}
        options={{ title: 'AI Doctor' }}
      />
      <Tab.Screen 
        name="ARScanner" 
        component={ARScannerScreen}
        options={{ title: 'AR Scan' }}
      />
      <Tab.Screen 
        name="Profile" 
        component={ProfileScreen}
        options={{ title: 'Profile' }}
      />
    </Tab.Navigator>
  );
}

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [fontsLoaded, setFontsLoaded] = useState(false);
  const { user, isAuthenticated, initialize } = useAuthStore();
  const { initializeHealth } = useHealthStore();

  useEffect(() => {
    async function loadResourcesAndDataAsync() {
      try {
        // Load fonts
        await Font.loadAsync({
          'Inter-Regular': require('./assets/fonts/Inter-Regular.ttf'),
          'Inter-Medium': require('./assets/fonts/Inter-Medium.ttf'),
          'Inter-Bold': require('./assets/fonts/Inter-Bold.ttf'),
        });
        setFontsLoaded(true);

        // Initialize authentication
        await initialize();

        // Initialize health data if user is authenticated
        if (isAuthenticated && user) {
          await initializeHealth(user.id);
        }

      } catch (error) {
        console.warn('Error loading resources:', error);
      } finally {
        setIsLoading(false);
        await SplashScreen.hideAsync();
      }
    }

    loadResourcesAndDataAsync();
  }, [initialize, initializeHealth, isAuthenticated, user]);

  if (isLoading || !fontsLoaded) {
    return <LoadingScreen />;
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <ThemeProvider>
          <NavigationContainer>
            <Stack.Navigator screenOptions={{ headerShown: false }}>
              {!isAuthenticated ? (
                // Auth Stack
                <Stack.Screen name="Auth" component={AuthScreen} />
              ) : (
                // Main App Stack
                <>
                  <Stack.Screen name="MainTabs" component={MainTabNavigator} />
                  <Stack.Screen 
                    name="TherapyGame" 
                    component={TherapyGameScreen}
                    options={{ 
                      presentation: 'modal',
                      headerShown: true,
                      title: 'Therapy Game'
                    }}
                  />
                  <Stack.Screen 
                    name="FutureSimulator" 
                    component={FutureSimulatorScreen}
                    options={{ 
                      presentation: 'modal',
                      headerShown: true,
                      title: 'Future-You Simulator'
                    }}
                  />
                  <Stack.Screen 
                    name="DoctorMarketplace" 
                    component={DoctorMarketplaceScreen}
                    options={{ 
                      presentation: 'modal',
                      headerShown: true,
                      title: 'Find Doctors'
                    }}
                  />
                </>
              )}
            </Stack.Navigator>
          </NavigationContainer>
          <StatusBar style="auto" />
        </ThemeProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}