import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const FutureSimulatorPage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [currentAge, setCurrentAge] = useState(35);
  const [targetAge, setTargetAge] = useState(65);
  const [lifestyleFactors, setLifestyleFactors] = useState({
    exercise: 'moderate',
    diet: 'good',
    smoking: 'never',
    alcohol: 'moderate'
  });
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulation, setSimulation] = useState<any>(null);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
    }
  }, [user, loading, router]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const createSimulation = async () => {
    setIsSimulating(true);
    setSimulation(null);

    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append('file', selectedFile);
      }
      formData.append('user_id', user?.id || 'demo_user');
      formData.append('current_age', currentAge.toString());
      formData.append('target_age', targetAge.toString());
      formData.append('lifestyle_factors', JSON.stringify(lifestyleFactors));

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/future-simulator/create-simulation`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setSimulation(data);
      }
    } catch (error) {
      console.error('Simulation failed:', error);
    } finally {
      setIsSimulating(false);
    }
  };

  const handleLifestyleChange = (factor: string, value: string) => {
    setLifestyleFactors(prev => ({
      ...prev,
      [factor]: value
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
          <h1 className="text-2xl font-bold text-gradient">Future-You Simulator</h1>
          <p className="text-dark-text-secondary mt-1">
            See how your lifestyle choices will affect your health and appearance in the future
          </p>
        </div>

        {!simulation ? (
          /* Simulation Setup */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Form */}
            <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
              <h2 className="text-lg font-semibold text-dark-text-primary mb-4">Create Your Simulation</h2>

              <div className="mb-6">
                <label className="block text-sm font-medium text-dark-text-primary mb-2">
                  Upload Your Photo (Optional)
                </label>
                <div className="border-2 border-dashed border-dark-border-primary rounded-xl p-6 text-center hover:bg-dark-bg-hover transition-all">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="photo-upload"
                  />
                  <label htmlFor="photo-upload" className="cursor-pointer">
                    <svg className="mx-auto h-12 w-12 text-dark-text-tertiary" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                      <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <div className="mt-2">
                      <span className="text-primary-400 font-medium">Click to upload</span>
                      <span className="text-dark-text-secondary"> or drag and drop</span>
                    </div>
                    <p className="text-xs text-dark-text-tertiary mt-1">PNG, JPG, GIF up to 10MB</p>
                  </label>
                </div>
                {selectedFile && (
                  <p className="mt-2 text-sm text-dark-text-secondary">
                    Selected: {selectedFile.name}
                  </p>
                )}
              </div>

              {/* Age Settings */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-2">
                    Current Age
                  </label>
                  <input
                    type="number"
                    min="18"
                    max="80"
                    value={currentAge}
                    onChange={(e) => setCurrentAge(parseInt(e.target.value))}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-2">
                    Target Age
                  </label>
                  <input
                    type="number"
                    min={currentAge + 1}
                    max="100"
                    value={targetAge}
                    onChange={(e) => setTargetAge(parseInt(e.target.value))}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  />
                </div>
              </div>

              {/* Lifestyle Factors */}
              <div className="space-y-4 mb-6">
                <h3 className="font-medium text-dark-text-primary">Lifestyle Factors</h3>

                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-1">
                    Exercise Level
                  </label>
                  <select
                    value={lifestyleFactors.exercise}
                    onChange={(e) => handleLifestyleChange('exercise', e.target.value)}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  >
                    <option value="poor">Poor (No regular exercise)</option>
                    <option value="moderate">Moderate (2-3 times/week)</option>
                    <option value="good">Good (4-5 times/week)</option>
                    <option value="excellent">Excellent (Daily exercise)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-1">
                    Diet Quality
                  </label>
                  <select
                    value={lifestyleFactors.diet}
                    onChange={(e) => handleLifestyleChange('diet', e.target.value)}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  >
                    <option value="poor">Poor (Fast food, processed)</option>
                    <option value="fair">Fair (Mixed diet)</option>
                    <option value="good">Good (Balanced nutrition)</option>
                    <option value="excellent">Excellent (Optimal nutrition)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-1">
                    Smoking Status
                  </label>
                  <select
                    value={lifestyleFactors.smoking}
                    onChange={(e) => handleLifestyleChange('smoking', e.target.value)}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  >
                    <option value="never">Never smoked</option>
                    <option value="former">Former smoker</option>
                    <option value="light">Light smoker</option>
                    <option value="heavy">Heavy smoker</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-1">
                    Alcohol Consumption
                  </label>
                  <select
                    value={lifestyleFactors.alcohol}
                    onChange={(e) => handleLifestyleChange('alcohol', e.target.value)}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  >
                    <option value="none">None</option>
                    <option value="light">Light (1-2 drinks/week)</option>
                    <option value="moderate">Moderate (3-7 drinks/week)</option>
                    <option value="heavy">Heavy (8+ drinks/week)</option>
                  </select>
                </div>
              </div>

              <button
                onClick={createSimulation}
                disabled={isSimulating}
                className="w-full bg-gradient-primary hover:opacity-90 disabled:opacity-50 text-white py-3 px-4 rounded-xl font-semibold shadow-lg glow-blue transition-all"
              >
                {isSimulating ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Creating Simulation...
                  </div>
                ) : (
                  'Create Future Simulation'
                )}
              </button>
            </div>

            {/* Preview */}
            <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
              <h2 className="text-lg font-semibold text-dark-text-primary mb-4">How It Works</h2>

              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="bg-primary-900 bg-opacity-30 w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-primary-400 font-semibold text-sm">1</span>
                  </div>
                  <div>
                    <h3 className="font-medium text-dark-text-primary">AI Age Progression</h3>
                    <p className="text-sm text-dark-text-secondary">Advanced AI analyzes your photo and creates realistic age progression</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="bg-primary-900 bg-opacity-30 w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-primary-400 font-semibold text-sm">2</span>
                  </div>
                  <div>
                    <h3 className="font-medium text-dark-text-primary">Health Predictions</h3>
                    <p className="text-sm text-dark-text-secondary">ML models predict health risks based on lifestyle and genetics</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="bg-primary-900 bg-opacity-30 w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-primary-400 font-semibold text-sm">3</span>
                  </div>
                  <div>
                    <h3 className="font-medium text-dark-text-primary">Lifestyle Impact</h3>
                    <p className="text-sm text-dark-text-secondary">See how different choices affect your future health and longevity</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="bg-primary-900 bg-opacity-30 w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-primary-400 font-semibold text-sm">4</span>
                  </div>
                  <div>
                    <h3 className="font-medium text-dark-text-primary">Personalized Recommendations</h3>
                    <p className="text-sm text-dark-text-secondary">Get actionable advice to improve your future health outcomes</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 p-4 bg-dark-bg-tertiary rounded-xl border border-dark-border-primary">
                <h4 className="font-medium text-primary-400 mb-2">Preview Simulation</h4>
                <p className="text-sm text-dark-text-secondary">
                  Age progression from {currentAge} to {targetAge} years old with {lifestyleFactors.exercise} exercise level and {lifestyleFactors.diet} diet quality.
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* Simulation Results */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Aged Image */}
            <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
              <h2 className="text-lg font-semibold text-dark-text-primary mb-4">Future You at Age {targetAge}</h2>

              <div className="aspect-square bg-dark-bg-tertiary rounded-xl flex items-center justify-center mb-4 border border-dark-border-primary overflow-hidden">
                <img
                  src={simulation.aged_image_url}
                  alt={`Future you at age ${targetAge}`}
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="text-center">
                <button
                  onClick={() => setSimulation(null)}
                  className="bg-dark-bg-tertiary hover:bg-dark-bg-hover text-dark-text-primary border border-dark-border-primary py-2 px-4 rounded-xl font-medium transition-all"
                >
                  Create New Simulation
                </button>
              </div>
            </div>

            {/* Health Predictions */}
            <div className="space-y-6">
              {/* Health Score */}
              <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                <h3 className="text-lg font-semibold text-dark-text-primary mb-4">Health Predictions</h3>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="text-center p-4 bg-dark-bg-tertiary rounded-xl border border-dark-border-primary">
                    <div className="text-3xl font-bold text-blue-400">
                      {simulation.health_predictions.health_score.toFixed(0)}
                    </div>
                    <div className="text-sm text-dark-text-secondary">Health Score</div>
                  </div>
                  <div className="text-center p-4 bg-dark-bg-tertiary rounded-xl border border-dark-border-primary">
                    <div className="text-3xl font-bold text-green-400">
                      {simulation.health_predictions.life_expectancy}
                    </div>
                    <div className="text-sm text-dark-text-secondary">Life Expectancy</div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-sm">
                      <span className="text-dark-text-secondary">Cardiovascular Risk</span>
                      <span className="text-white font-medium">{simulation.health_predictions.cardiovascular_risk.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-dark-bg-tertiary rounded-full h-2">
                      <div
                        className="bg-red-500 h-2 rounded-full shadow-lg glow-red"
                        style={{ width: `${simulation.health_predictions.cardiovascular_risk}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-sm">
                      <span className="text-dark-text-secondary">Diabetes Risk</span>
                      <span className="text-white font-medium">{simulation.health_predictions.diabetes_risk.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-dark-bg-tertiary rounded-full h-2">
                      <div
                        className="bg-orange-500 h-2 rounded-full shadow-lg glow-orange"
                        style={{ width: `${simulation.health_predictions.diabetes_risk}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-sm">
                      <span className="text-dark-text-secondary">Cancer Risk</span>
                      <span className="text-white font-medium">{simulation.health_predictions.cancer_risk.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-dark-bg-tertiary rounded-full h-2">
                      <div
                        className="bg-purple-500 h-2 rounded-full shadow-lg glow-purple"
                        style={{ width: `${simulation.health_predictions.cancer_risk}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Lifestyle Scenarios */}
              <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                <h3 className="text-lg font-semibold text-dark-text-primary mb-4">Lifestyle Impact</h3>

                <div className="space-y-3">
                  <div className="p-4 bg-green-500/10 rounded-xl border border-green-500/20">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-green-400">Improved Lifestyle</span>
                      <span className="text-green-400 font-bold">
                        {simulation.lifestyle_scenarios.improved.life_expectancy.toFixed(1)} years
                      </span>
                    </div>
                    <p className="text-sm text-green-400/80 mt-1">
                      +{(simulation.lifestyle_scenarios.improved.life_expectancy - simulation.lifestyle_scenarios.current.life_expectancy).toFixed(1)} years with better habits
                    </p>
                  </div>

                  <div className="p-4 bg-yellow-500/10 rounded-xl border border-yellow-500/20">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-yellow-400">Current Lifestyle</span>
                      <span className="text-yellow-400 font-bold">
                        {simulation.lifestyle_scenarios.current.life_expectancy.toFixed(1)} years
                      </span>
                    </div>
                    <p className="text-sm text-yellow-400/80 mt-1">
                      Based on your current habits
                    </p>
                  </div>

                  <div className="p-4 bg-red-500/10 rounded-xl border border-red-500/20">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-red-400">Declined Lifestyle</span>
                      <span className="text-red-400 font-bold">
                        {simulation.lifestyle_scenarios.declined.life_expectancy.toFixed(1)} years
                      </span>
                    </div>
                    <p className="text-sm text-red-400/80 mt-1">
                      {(simulation.lifestyle_scenarios.current.life_expectancy - simulation.lifestyle_scenarios.declined.life_expectancy).toFixed(1)} years lost with poor habits
                    </p>
                  </div>
                </div>
              </div>

              {/* AI Narrative */}
              <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                <h3 className="text-lg font-semibold text-dark-text-primary mb-4">AI Health Insights</h3>
                <p className="text-dark-text-secondary text-sm leading-relaxed">
                  {simulation.ai_narrative}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Recommendations */}
        {simulation && (
          <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
            <h2 className="text-lg font-semibold text-dark-text-primary mb-4">Personalized Recommendations</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {simulation.recommendations.map((recommendation: string, index: number) => (
                <div key={index} className="flex items-start space-x-3 p-4 bg-dark-bg-tertiary rounded-xl border border-dark-border-primary">
                  <div className="bg-primary-900 bg-opacity-30 w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <svg className="w-3 h-3 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <p className="text-sm text-dark-text-secondary">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default FutureSimulatorPage;