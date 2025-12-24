import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuthStore } from '../stores/authStore'; // Corrected import
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const TherapyGamePage: React.FC = () => {
  const { user, isLoading: loading } = useAuthStore();
  const router = useRouter();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [gameType, setGameType] = useState('shoulder_rehabilitation');
  const [adventureState, setAdventureState] = useState<any>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [score, setScore] = useState(0);
  const [expressMode, setExpressMode] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
    }
  }, [user, loading, router]);

  const startAdventure = async () => {
    setIsGenerating(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/therapy-game/start-adventure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // Send cookies
        body: JSON.stringify({
          user_id: user?.id,
          game_type: gameType,
          difficulty: 3,
          express_mode: expressMode
        }),
      });
      const data = await response.json();
      setAdventureState(data); // StateGraph result
      setSessionId('active');
    } catch (error) {
      console.error('Failed to start adventure:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const nextTurn = async (performanceScore: number) => {
    setIsGenerating(true);
    try {
      setScore(prev => prev + performanceScore);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/therapy-game/next-turn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // Send cookies
        body: JSON.stringify({
          user_id: user?.id,
          game_type: gameType,
          difficulty: adventureState?.difficulty_level || 3,
          last_score: performanceScore,
          express_mode: expressMode
        }),
      });
      const data = await response.json();
      setAdventureState(data);
    } catch (error) {
      console.error('Failed next turn:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  if (loading) return <div className="min-h-screen center"><LoadingSpinner /></div>;
  if (!user) return null;

  // Helper to get story text
  const storyText = adventureState?.narrative_history ? adventureState.narrative_history[0] : "Your adventure awaits...";
  const exercise = adventureState?.current_exercise || {};

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
          <h1 className="text-3xl font-bold text-gradient">AI Adaptive Adventure</h1>
          <p className="text-dark-text-secondary mt-2">
            A unique story generated in real-time by your progress.
          </p>
        </div>

        {!sessionId ? (
          <div className="glass-strong rounded-2xl p-8 border border-dark-border-primary text-center">
            <h2 className="text-2xl font-bold text-dark-text-primary mb-6">Choose Your Quest</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {['shoulder_rehabilitation', 'knee_recovery', 'back_strengthening'].map(type => (
                <button
                  key={type}
                  onClick={() => setGameType(type)}
                  className={`p-6 rounded-xl border-2 transition-all ${gameType === type
                    ? 'border-purple-500 bg-purple-900/20 shadow-glow-purple'
                    : 'border-dark-border-primary bg-dark-bg-tertiary hover:border-purple-400'
                    }`}
                >
                  <div className="text-lg font-semibold capitalize text-dark-text-primary">
                    {type.replace('_', ' ')}
                  </div>
                </button>
              ))}
            </div>

            {/* Express Mode Toggle */}
            <div className="flex justify-center mb-8">
              <button
                onClick={() => setExpressMode(!expressMode)}
                className={`flex items-center space-x-3 px-6 py-3 rounded-full border transition-all ${expressMode
                  ? 'bg-gradient-to-r from-yellow-500 to-orange-500 border-orange-400 text-white'
                  : 'bg-dark-bg-tertiary border-dark-border-primary text-gray-400 hover:border-gray-500'
                  }`}
              >
                <span className="text-2xl">⚡</span>
                <span className="font-bold">
                  {expressMode ? "Express Mode Active" : "Enable Express Mode"}
                </span>
              </button>
            </div>

            <button
              onClick={startAdventure}
              disabled={isGenerating}
              className="bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 px-12 rounded-full font-bold text-lg shadow-lg glow-purple hover:scale-105 transition-transform disabled:opacity-50"
            >
              {isGenerating ? "Summoning Dungeon Master..." : "Start Adventure"}
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Story & Visuals */}
            <div className="lg:col-span-2 space-y-6">
              {/* Narrative Card - Hidden in Express Mode */}
              {!expressMode && (
                <div className="glass-strong rounded-2xl p-6 border border-purple-500/30 relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 to-blue-900/20 opacity-50"></div>
                  <div className="relative z-10">
                    <div className="flex items-center space-x-3 mb-4">
                      <span className="text-2xl">📜</span>
                      <h3 className="text-xl font-bold text-purple-200">The Story So Far...</h3>
                    </div>
                    <p className="text-lg text-gray-100 leading-relaxed font-serif italic">
                      "{storyText}"
                    </p>
                  </div>
                </div>
              )}

              {/* Dynamic Exercise Card */}
              <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                {isGenerating ? (
                  <div className="h-64 flex flex-col items-center justify-center space-y-4">
                    <LoadingSpinner />
                    <p className="text-purple-300 animate-pulse">The Motion Architect is designing your challenge...</p>
                  </div>
                ) : (
                  <>
                    <div className="flex justify-between items-start mb-6">
                      <div>
                        <h2 className="text-2xl font-bold text-white mb-2">{exercise.name || "Unknown Challenge"}</h2>
                        <p className="text-gray-400 max-w-lg">{exercise.description}</p>
                      </div>
                      <div className="bg-dark-bg-tertiary px-4 py-2 rounded-lg border border-dark-border-primary">
                        <span className="text-purple-400 font-bold">{exercise.duration}s</span> / {exercise.reps} reps
                      </div>
                    </div>

                    {/* Camera Feed Placeholder */}
                    <div className="aspect-video bg-black/50 rounded-xl border border-dashed border-gray-600 flex items-center justify-center mb-6">
                      <div className="text-center text-gray-500">
                        <div className="text-4xl mb-2">📷</div>
                        <p>AI Motion Tracking Active</p>
                      </div>
                    </div>

                    <div className="flex gap-4">
                      <button
                        onClick={() => nextTurn(95)}
                        className="flex-1 bg-green-600 hover:bg-green-500 text-white py-3 rounded-xl font-bold transition-colors"
                      >
                        Complete (Excellent Form)
                      </button>
                      <button
                        onClick={() => nextTurn(60)}
                        className="flex-1 bg-yellow-600 hover:bg-yellow-500 text-white py-3 rounded-xl font-bold transition-colors"
                      >
                        Complete (Struggled)
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Stats Panel */}
            <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary h-fit">
              <h3 className="text-xl font-bold text-white mb-4">Hero Stats</h3>
              <div className="space-y-4">
                <div className="bg-dark-bg-tertiary p-4 rounded-xl">
                  <div className="text-gray-400 text-sm">Total Score</div>
                  <div className="text-2xl font-bold text-purple-400">{score} XP</div>
                </div>
                <div className="bg-dark-bg-tertiary p-4 rounded-xl">
                  <div className="text-gray-400 text-sm">Difficulty</div>
                  <div className="text-2xl font-bold text-blue-400">{adventureState?.difficulty_level || 3}/10</div>
                </div>
                <div className="bg-dark-bg-tertiary p-4 rounded-xl">
                  <div className="text-gray-400 text-sm">Fatigue</div>
                  <div className="w-full bg-gray-700 h-2 rounded-full mt-2">
                    <div
                      className="bg-red-500 h-full rounded-full transition-all"
                      style={{ width: `${adventureState?.fatigue || 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
        }
      </div >
    </DashboardLayout >
  );
};

export default TherapyGamePage;