# HealthSync AI - Frontend Applications

This folder contains both mobile and web frontend applications for the HealthSync AI healthcare platform.

## 📱 Mobile App (React Native/Expo)

### Overview
Cross-platform mobile application for patients to access all HealthSync AI features including voice consultations, AR scanning, therapy games, and future health simulations.

### Tech Stack
- **Framework**: React Native with Expo 49
- **Language**: TypeScript
- **Navigation**: React Navigation 6
- **State Management**: Zustand
- **Authentication**: Supabase Auth
- **Styling**: StyleSheet with LinearGradient
- **Camera/AR**: Expo Camera + Expo Three
- **Icons**: Expo Vector Icons

### Project Structure
```
mobile/
├── package.json              # Dependencies and scripts
├── app.json                  # Expo configuration
├── tsconfig.json             # TypeScript configuration
├── babel.config.js           # Babel configuration
├── .env.example              # Environment variables template
├── App.tsx                   # Main app entry point
└── src/
    ├── components/           # Reusable components
    │   └── LoadingScreen.tsx
    ├── contexts/             # React contexts
    │   └── ThemeContext.tsx
    ├── screens/              # App screens
    │   ├── AuthScreen.tsx
    │   ├── HomeScreen.tsx
    │   ├── VoiceDoctorScreen.tsx
    │   ├── ARScannerScreen.tsx
    │   ├── TherapyGameScreen.tsx
    │   ├── FutureSimulatorScreen.tsx
    │   ├── DoctorMarketplaceScreen.tsx
    │   ├── HealthDashboardScreen.tsx
    │   └── ProfileScreen.tsx
    ├── services/             # API and auth services
    │   ├── AuthService.ts
    │   └── ApiService.ts
    ├── stores/               # Zustand state stores
    │   ├── authStore.ts
    │   └── healthStore.ts
    └── types/                # TypeScript definitions
        ├── navigation.ts
        └── health.ts
```

### Key Features
- **Authentication**: Supabase Auth with secure token storage
- **Voice AI Doctor**: Real-time voice analysis and consultation
- **AR Medical Scanner**: Camera-based prescription and document scanning
- **Therapy Game**: Gamified rehabilitation exercises with motion tracking
- **Future Simulator**: AI-powered age progression and health projections
- **Doctor Marketplace**: Specialist search and appointment booking
- **Health Dashboard**: Comprehensive health metrics and insights
- **Profile Management**: User settings and data management

### Setup Instructions
1. Install dependencies: `npm install`
2. Copy `.env.example` to `.env` and configure variables
3. Start development server: `npm start`
4. Run on device: `npm run android` or `npm run ios`

## 🌐 Web Dashboard (Next.js)

### Overview
Professional web dashboard for doctors and administrators to manage patients, appointments, and platform analytics.

### Tech Stack
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS + Headless UI
- **State Management**: Zustand
- **Authentication**: Supabase Auth
- **Charts**: Chart.js with React-ChartJS-2
- **Animations**: Framer Motion
- **Forms**: React Hook Form
- **Notifications**: React Hot Toast

### Project Structure
```
web/
├── package.json              # Dependencies and scripts
├── next.config.js            # Next.js configuration
├── tailwind.config.js        # Tailwind CSS configuration
├── tsconfig.json             # TypeScript configuration
├── postcss.config.js         # PostCSS configuration
├── .eslintrc.json            # ESLint configuration
├── .env.example              # Environment variables template
├── public/                   # Static assets
│   └── favicon.ico
└── src/
    ├── components/           # React components
    │   ├── dashboard/
    │   │   └── DashboardOverview.tsx
    │   ├── layout/
    │   │   └── DashboardLayout.tsx
    │   └── ui/
    │       └── LoadingSpinner.tsx
    ├── contexts/             # React contexts
    │   └── AuthContext.tsx
    ├── pages/                # Next.js pages
    │   ├── _app.tsx          # App wrapper
    │   ├── index.tsx         # Dashboard home
    │   └── auth/
    │       └── login.tsx     # Login page
    ├── stores/               # Zustand state stores
    │   └── authStore.ts
    └── styles/               # Global styles
        └── globals.css
```

### Key Features
- **Role-Based Dashboard**: Different interfaces for doctors and admins
- **Patient Management**: View and manage patient records (doctors)
- **Appointment System**: Schedule and manage appointments
- **Analytics Dashboard**: Platform metrics and health insights
- **User Management**: Admin tools for user administration
- **Responsive Design**: Mobile-friendly responsive layout
- **Real-time Updates**: Live data updates and notifications

### Setup Instructions
1. Install dependencies: `npm install`
2. Copy `.env.example` to `.env.local` and configure variables
3. Start development server: `npm run dev`
4. Build for production: `npm run build`
5. Start production server: `npm start`

## 🔧 Shared Configuration

### Environment Variables
Both applications require the following environment variables:

```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

# API Configuration
API_URL=http://localhost:8000

# App Configuration
APP_NAME=HealthSync AI
APP_VERSION=1.0.0
```

### Authentication Flow
1. **Login/Register**: Users authenticate via Supabase Auth
2. **Token Storage**: JWT tokens stored securely (SecureStore on mobile, localStorage on web)
3. **Auto-refresh**: Tokens automatically refreshed on expiration
4. **Role-based Access**: Different features based on user role (patient/doctor/admin)

### API Integration
Both applications use a shared API service pattern:
- Centralized API client with authentication headers
- Consistent error handling and response formatting
- Type-safe request/response interfaces
- Automatic token refresh and retry logic

## 🚀 Deployment

### Mobile App Deployment
1. **Development**: Use Expo Go app for testing
2. **Staging**: Build with `eas build` for internal testing
3. **Production**: Submit to App Store and Google Play Store

### Web Dashboard Deployment
1. **Development**: Local development server
2. **Staging**: Deploy to Vercel preview environment
3. **Production**: Deploy to Vercel production with custom domain

## 🧪 Testing

### Mobile App Testing
- **Unit Tests**: Jest with React Native Testing Library
- **Integration Tests**: Detox for E2E testing
- **Device Testing**: Physical devices and simulators

### Web Dashboard Testing
- **Unit Tests**: Jest with React Testing Library
- **Integration Tests**: Cypress for E2E testing
- **Browser Testing**: Cross-browser compatibility testing

## 📱 Mobile App Features

### Authentication Screen
- Email/password login and registration
- Secure token storage with Expo SecureStore
- Role-based redirection after login

### Home Screen
- Personalized dashboard with health overview
- Quick access to all major features
- Health score calculation and risk factors display
- Recent activity and insights

### Voice Doctor Screen
- Voice recording interface with visual feedback
- Real-time audio analysis and transcription
- AI-powered symptom analysis and recommendations
- Voice stress detection and health assessment

### AR Scanner Screen
- Camera interface for document scanning
- OCR for prescription and lab report analysis
- Real-time text extraction and medication identification
- Safety warnings and drug interaction alerts

### Therapy Game Screen
- Gamified rehabilitation exercises
- Motion tracking with MediaPipe integration
- Pain detection through facial analysis
- Progress tracking and achievement system

### Future Simulator Screen
- Photo upload for age progression
- AI-powered health projections
- Lifestyle scenario comparisons
- Personalized health narratives

### Doctor Marketplace Screen
- AI-powered symptom-to-specialist matching
- Doctor search and filtering
- Appointment booking with bidding system
- Real-time appointment status updates

### Health Dashboard Screen
- Comprehensive health metrics overview
- Interactive charts and trend analysis
- Risk factor visualization
- Family health graph integration

### Profile Screen
- User profile management
- Settings and preferences
- Data export and privacy controls
- Support and help resources

## 🌐 Web Dashboard Features

### Dashboard Overview
- Role-specific dashboards for doctors and admins
- Key performance indicators and metrics
- Recent activity and notifications
- System health monitoring

### Doctor Dashboard
- Patient management and medical records
- Appointment scheduling and management
- Revenue tracking and analytics
- Performance metrics and ratings

### Admin Dashboard
- Platform-wide analytics and insights
- User management and moderation
- System monitoring and health checks
- Marketplace statistics and trends

### Authentication
- Secure login with role-based access
- Password reset and account recovery
- Session management and security

## 🔒 Security Features

### Mobile App Security
- Secure token storage with Expo SecureStore
- Biometric authentication support
- Certificate pinning for API calls
- Data encryption at rest and in transit

### Web Dashboard Security
- JWT token authentication
- CSRF protection
- XSS prevention
- Secure cookie handling

## 🎨 Design System

### Mobile App Design
- Consistent color palette and typography
- Material Design principles
- Accessibility compliance (WCAG 2.1)
- Dark mode support

### Web Dashboard Design
- Professional healthcare-focused design
- Tailwind CSS utility classes
- Responsive grid system
- Consistent component library

## 📊 Performance Optimization

### Mobile App Performance
- Lazy loading of screens and components
- Image optimization and caching
- Efficient state management with Zustand
- Bundle size optimization

### Web Dashboard Performance
- Next.js automatic code splitting
- Image optimization with Next.js Image
- Static generation for improved loading
- CDN integration for assets

## 🔄 State Management

Both applications use Zustand for lightweight, efficient state management:

### Auth Store
- User authentication state
- Login/logout functionality
- Token management
- Role-based permissions

### Health Store
- Health metrics and predictions
- Therapy progress tracking
- AR scan results
- Future simulation history

## 🌍 Internationalization

### Planned Features
- Multi-language support (English, Spanish, French)
- Localized date and number formatting
- Cultural adaptation for health metrics
- Right-to-left language support

## 📈 Analytics Integration

### Mobile App Analytics
- User engagement tracking
- Feature usage analytics
- Performance monitoring
- Crash reporting

### Web Dashboard Analytics
- Admin usage patterns
- Doctor productivity metrics
- Platform performance insights
- Business intelligence dashboards

## 🚀 Future Enhancements

### Mobile App Roadmap
- Offline mode support
- Wearable device integration
- Advanced AR features
- Social health features

### Web Dashboard Roadmap
- Advanced analytics dashboards
- Telemedicine integration
- Automated reporting
- AI-powered insights

## 📞 Support

For technical support and questions:
- **Email**: dev-support@healthsync.ai
- **Documentation**: https://docs.healthsync.ai
- **Issues**: GitHub Issues
- **Community**: Discord Server