import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const TherapyGamePage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [gameType, setGameType] = useState('shoulder_rehabilitation');
  const [currentExercise, setCurrentExercise] = useState<any>(null);
  const [exercises, setExercises] = useState<any[]>([]);
  const [score, setScore] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
    }
  }, [user, loading, router]);

  const startTherapySession = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/therapy-game/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: user?.id || 'demo_user',
          game_type: gameType,
          difficulty: 'beginner'
        }),
      });

      const data = await response.json();
      if (data.success) {
        setSessionId(data.session_id);
        setExercises(data.exercises);
        setCurrentExercise(data.exercises[0]);
        setIsPlaying(true);
      }
    } catch (error) {
      console.error('Failed to start therapy session:', error);
    }
  };

  const completeExercise = async (performanceScore: number) => {
    if (!sessionId || !currentExercise) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/therapy-game/complete-exercise`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          exercise_name: currentExercise.name,
          performance_score: performanceScore
        }),
      });

      const data = await response.json();
      if (data.success) {
        setScore(data.total_score);
        setProgress(data.progress);
        
        // Mark current exercise as completed
        const updatedExercises = exercises.map(ex => 
          ex.name === currentExercise.name 
            ? { ...ex, completed: true, performance_score: performanceScore }
            : ex
        );
        setExercises(updatedExercises);

        // Move to next exercise or end session
        const nextExercise = updatedExercises.find(ex => !ex.completed);
        if (nextExercise) {
          setCurrentExercise(nextExercise);
        } else {
          setIsPlaying(false);
          setCurrentExercise(null);
        }
      }
    } catch (error) {
      console.error('Failed to complete exercise:', error);
    }
  };

  const resetSession = () => {
    setSessionId(null);
    setCurrentExercise(null);
    setExercises([]);
    setScore(0);
    setProgress(0);
    setIsPlaying(false);
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
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gradient">Pain-to-Game Therapy</h1>
              <p className="text-dark-text-secondary mt-2">
                Transform rehabilitation into engaging, gamified exercises
              </p>
            </div>
            {sessionId && (
              <div className="text-right">
                <div className="text-3xl font-bold text-gradient">{score} pts</div>
                <div className="text-sm text-dark-text-secondary">Progress: {progress.toFixed(0)}%</div>
              </div>
            )}
          </div>
        </div>

        {!sessionId ? (
          /* Game Setup */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
              <h2 className="text-xl font-semibold text-dark-text-primary mb-4">Choose Your Therapy</h2>
              
              <div className="space-y-3 mb-6">
                <label className="flex items-center p-4 border border-dark-border-primary rounded-xl cursor-pointer hover:bg-dark-bg-hover transition-all bg-dark-bg-tertiary">
                  <input
                    type="radio"
                    name="gameType"
                    value="shoulder_rehabilitation"
                    checked={gameType === 'shoulder_rehabilitation'}
                    onChange={(e) => setGameType(e.target.value)}
                    className="mr-3 text-primary-500"
                  />
                  <div>
                    <div className="font-semibold text-dark-text-primary">Shoulder Rehabilitation</div>
                    <div className="text-sm text-dark-text-secondary">Arm circles, shoulder shrugs, wall push-ups</div>
                  </div>
                </label>

                <label className="flex items-center p-4 border border-dark-border-primary rounded-xl cursor-pointer hover:bg-dark-bg-hover transition-all bg-dark-bg-tertiary">
                  <input
                    type="radio"
                    name="gameType"
                    value="back_strengthening"
                    checked={gameType === 'back_strengthening'}
                    onChange={(e) => setGameType(e.target.value)}
                    className="mr-3 text-primary-500"
                  />
                  <div>
                    <div className="font-semibold text-dark-text-primary">Back Strengthening</div>
                    <div className="text-sm text-dark-text-secondary">Cat-cow stretch, bird dog, bridge pose</div>
                  </div>
                </label>

                <label className="flex items-center p-4 border border-dark-border-primary rounded-xl cursor-pointer hover:bg-dark-bg-hover transition-all bg-dark-bg-tertiary">
                  <input
                    type="radio"
                    name="gameType"
                    value="knee_recovery"
                    checked={gameType === 'knee_recovery'}
                    onChange={(e) => setGameType(e.target.value)}
                    className="mr-3 text-primary-500"
                  />
                  <div>
                    <div className="font-semibold text-dark-text-primary">Knee Recovery</div>
                    <div className="text-sm text-dark-text-secondary">Leg raises, heel slides, quad sets</div>
                  </div>
                </label>
              </div>

              <button
                onClick={startTherapySession}
                className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:opacity-90 text-white py-3 px-4 rounded-xl font-semibold shadow-lg glow-purple transition-all"
              >
                Start Therapy Session
              </button>
            </div>

            <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
              <h2 className="text-xl font-semibold text-dark-text-primary mb-4">How It Works</h2>
              
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="bg-gradient-to-br from-purple-500 to-purple-600 w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg glow-purple">
                    <span className="text-white font-bold text-sm">1</span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-dark-text-primary">Motion Tracking</h3>
                    <p className="text-sm text-dark-text-secondary">AI tracks your movements in real-time using your device camera</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="bg-gradient-to-br from-purple-500 to-purple-600 w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg glow-purple">
                    <span className="text-white font-bold text-sm">2</span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-dark-text-primary">Performance Analysis</h3>
                    <p className="text-sm text-dark-text-secondary">Get instant feedback on form, speed, and accuracy</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="bg-gradient-to-br from-purple-500 to-purple-600 w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg glow-purple">
                    <span className="text-white font-bold text-sm">3</span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-dark-text-primary">Earn Rewards</h3>
                    <p className="text-sm text-dark-text-secondary">Complete exercises to earn points and unlock achievements</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Active Session */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Exercise Area */}
            <div className="lg:col-span-2 glass-strong rounded-2xl p-6 border border-dark-border-primary">
              {currentExercise ? (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-dark-text-primary">{currentExercise.name}</h2>
                    <div className="text-sm text-dark-text-secondary">
                      {exercises.filter(ex => ex.completed).length} / {exercises.length} completed
                    </div>
                  </div>

                  {/* Exercise Demo Area */}
                  <div className="bg-dark-bg-secondary rounded-xl h-64 flex items-center justify-center mb-4 border border-dark-border-primary">
                    <div className="text-center">
                      <div className="bg-gradient-to-br from-purple-500 to-purple-600 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg glow-purple">
                        <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.5a2.5 2.5 0 110 5H9m4.5-1.206a11.955 11.955 0 01-2.5 2.5M15 6.5a11.955 11.955 0 01-2.5-2.5M9 6.5a11.955 11.955 0 00-2.5-2.5m1.5 2.5h3m-3 0h-.5a2.5 2.5 0 00-2.5 2.5V12a2.5 2.5 0 002.5 2.5H9m-3-6h3m-3 0h-.5a2.5 2.5 0 00-2.5 2.5v3a2.5 2.5 0 002.5 2.5H9" />
                        </svg>
                      </div>
                      <p className="text-dark-text-primary">Camera feed would appear here</p>
                      <p className="text-sm text-dark-text-secondary mt-2">Follow the on-screen instructions</p>
                    </div>
                  </div>

                  {/* Exercise Info */}
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="text-center bg-dark-bg-tertiary p-3 rounded-xl border border-dark-border-primary">
                      <div className="text-lg font-semibold text-dark-text-primary">{currentExercise.duration}s</div>
                      <div className="text-sm text-dark-text-secondary">Duration</div>
                    </div>
                    <div className="text-center bg-dark-bg-tertiary p-3 rounded-xl border border-dark-border-primary">
                      <div className="text-lg font-semibold text-dark-text-primary">{currentExercise.repetitions}</div>
                      <div className="text-sm text-dark-text-secondary">Reps</div>
                    </div>
                    <div className="text-center bg-gradient-to-br from-purple-500 to-purple-600 p-3 rounded-xl shadow-lg glow-purple">
                      <div className="text-lg font-semibold text-white">{currentExercise.points}</div>
                      <div className="text-sm text-purple-100">Points</div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-3">
                    <button
                      onClick={() => completeExercise(85)}
                      className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:opacity-90 text-white py-3 px-4 rounded-xl font-semibold shadow-lg glow-green transition-all"
                    >
                      Complete Exercise (85%)
                    </button>
                    <button
                      onClick={() => completeExercise(95)}
                      className="flex-1 bg-gradient-to-r from-purple-600 to-purple-700 hover:opacity-90 text-white py-3 px-4 rounded-xl font-semibold shadow-lg glow-purple transition-all"
                    >
                      Perfect Form (95%)
                    </button>
                  </div>
                </div>
              ) : (
                /* Session Complete */
                <div className="text-center py-8">
                  <div className="bg-gradient-to-br from-green-500 to-emerald-600 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg glow-green">
                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h2 className="text-2xl font-semibold text-dark-text-primary mb-2">Session Complete!</h2>
                  <p className="text-dark-text-secondary mb-4">Great job! You earned {score} points.</p>
                  <button
                    onClick={resetSession}
                    className="bg-gradient-to-r from-purple-600 to-purple-700 hover:opacity-90 text-white py-3 px-6 rounded-xl font-semibold shadow-lg glow-purple transition-all"
                  >
                    Start New Session
                  </button>
                </div>
              )}
            </div>

            {/* Progress Panel */}
            <div className="space-y-6">
              {/* Progress */}
              <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                <h3 className="font-semibold text-dark-text-primary mb-4">Session Progress</h3>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-dark-text-secondary">Overall Progress</span>
                    <span className="text-dark-text-primary font-semibold">{progress.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-dark-bg-tertiary rounded-full h-3 border border-dark-border-primary">
                    <div 
                      className="bg-gradient-to-r from-purple-600 to-purple-700 h-3 rounded-full transition-all duration-300 shadow-lg"
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Exercise List */}
              <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                <h3 className="font-semibold text-dark-text-primary mb-4">Exercises</h3>
                <div className="space-y-2">
                  {exercises.map((exercise, index) => (
                    <div 
                      key={index}
                      className={`flex items-center justify-between p-3 rounded-xl border transition-all ${
                        exercise.completed 
                          ? 'bg-green-900 bg-opacity-30 border-green-500 text-green-400' 
                          : exercise === currentExercise
                          ? 'bg-gradient-to-r from-purple-600 to-purple-700 border-purple-500 text-white shadow-lg glow-purple'
                          : 'bg-dark-bg-tertiary border-dark-border-primary text-dark-text-secondary'
                      }`}
                    >
                      <span className="text-sm font-medium">{exercise.name}</span>
                      <div className="flex items-center space-x-2">
                        {exercise.completed && (
                          <span className="text-xs bg-green-600 text-white px-2 py-1 rounded-lg font-semibold">
                            {exercise.performance_score}%
                          </span>
                        )}
                        <span className="text-xs font-semibold">{exercise.points}pts</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default TherapyGamePage;