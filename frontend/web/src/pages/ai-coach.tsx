// AI Coach Dashboard - Daily Medical Plan View (Refactored)
// Shows today's plan with medication reminders, exercises, diet, adherence

import { useState, useEffect } from 'react';
import DashboardLayout from '../components/layout/DashboardLayout';
import { useAuthStore } from '../stores/authStore';

export default function AICoachPage() {
    const { user } = useAuthStore();
    const [loading, setLoading] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [dailyPlan, setDailyPlan] = useState<any>(null);
    const [history, setHistory] = useState<any[]>([]);
    const [selectedTab, setSelectedTab] = useState<'today' | 'history'>('today');

    useEffect(() => {
        if (user) {
            fetchDailyPlan();
            fetchHistory();
        }
    }, [user]);

    const fetchDailyPlan = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/api/agents/daily-plan', {
                credentials: 'include',
            });

            if (response.ok) {
                const data = await response.json();
                setDailyPlan(data);
            } else {
                setDailyPlan(null);
            }
        } catch (error) {
            console.error('Failed to fetch daily plan:', error);
            setDailyPlan(null);
        } finally {
            setLoading(false);
        }
    };

    const fetchHistory = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/agents/history?limit=7', {
                credentials: 'include',
            });

            if (response.ok) {
                const data = await response.json();
                setHistory(data.history || []);
            }
        } catch (error) {
            console.error('Failed to fetch history:', error);
        }
    };

    const generateTodaysPlan = async () => {
        setGenerating(true);
        try {
            const response = await fetch('http://localhost:8000/api/agents/trigger-daily-planning', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ force: false }),
            });

            const data = await response.json();

            if (data.success) {
                // Refresh plan
                await fetchDailyPlan();
                await fetchHistory();
            } else {
                alert(data.skip_reason || 'Failed to generate plan');
            }
        } catch (error) {
            console.error('Failed to generate plan:', error);
            alert('Failed to generate today\'s plan');
        } finally {
            setGenerating(false);
        }
    };

    const toggleTask = async (taskId: string, completed: boolean) => {
        // TODO: Implement task completion API
        console.log('Toggle task:', taskId, completed);
    };

    if (!user) return null;

    const plan = dailyPlan?.daily_plan || {};
    const reflection = dailyPlan?.reflection || {};
    const safety = dailyPlan?.safety_approval || {};

    return (
        <DashboardLayout>
            <div className="max-w-7xl mx-auto py-8">
                {/* Header */}
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-4xl font-bold text-gradient">🤖 AI Health Coach</h1>
                        <p className="text-dark-text-secondary mt-2">
                            Your personalized daily health plan
                        </p>
                    </div>

                    <button
                        onClick={generateTodaysPlan}
                        disabled={generating || (dailyPlan && !dailyPlan.reflection)}
                        className={`px-6 py-3 rounded-lg font-medium transition-all ${generating || (dailyPlan && !dailyPlan.reflection)
                                ? 'bg-dark-bg-tertiary text-dark-text-tertiary cursor-not-allowed'
                                : 'bg-gradient-primary text-white hover:shadow-glow'
                            }`}
                    >
                        {generating ? 'Generating...' : dailyPlan ? 'Plan Already Generated Today' : 'Generate Today\'s Plan'}
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex gap-4 mb-6">
                    <button
                        onClick={() => setSelectedTab('today')}
                        className={`px-6 py-3 rounded-lg font-medium transition-all ${selectedTab === 'today'
                                ? 'bg-primary-500 text-white'
                                : 'bg-dark-bg-secondary text-dark-text-secondary hover:bg-dark-bg-hover'
                            }`}
                    >
                        📅 Today's Plan
                    </button>
                    <button
                        onClick={() => setSelectedTab('history')}
                        className={`px-6 py-3 rounded-lg font-medium transition-all ${selectedTab === 'history'
                                ? 'bg-primary-500 text-white'
                                : 'bg-dark-bg-secondary text-dark-text-secondary hover:bg-dark-bg-hover'
                            }`}
                    >
                        📊 History ({history.length} days)
                    </button>
                </div>

                {/* Content */}
                {selectedTab === 'today' && (
                    <div>
                        {loading ? (
                            <div className="text-center py-12">
                                <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-500 border-t-transparent mx-auto mb-4"></div>
                                <p className="text-dark-text-secondary">Loading today's plan...</p>
                            </div>
                        ) : !dailyPlan ? (
                            <div className="glass-strong rounded-2xl p-12 text-center border border-dark-border-primary">
                                <div className="text-6xl mb-4">📋</div>
                                <h2 className="text-2xl font-bold text-dark-text-primary mb-2">
                                    No Plan Generated Yet
                                </h2>
                                <p className="text-dark-text-secondary mb-6">
                                    Click "Generate Today's Plan" to create your personalized daily health plan
                                </p>
                                <button
                                    onClick={generateTodaysPlan}
                                    disabled={generating}
                                    className="px-8 py-3 bg-gradient-primary text-white rounded-lg font-medium hover:shadow-glow transition-all"
                                >
                                    {generating ? 'Generating...' : 'Generate Now'}
                                </button>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* Medication Reminders */}
                                <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                                    <div className="flex items-center gap-3 mb-4">
                                        <span className="text-3xl">💊</span>
                                        <h2 className="text-xl font-bold text-dark-text-primary">Medication Reminders</h2>
                                    </div>

                                    {plan.medication_reminders?.length > 0 ? (
                                        <div className="space-y-3">
                                            {plan.medication_reminders.map((med: any, index: number) => (
                                                <div
                                                    key={index}
                                                    className="bg-dark-bg-secondary rounded-lg p-4 border border-dark-border-secondary"
                                                >
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex-1">
                                                            <div className="font-medium text-dark-text-primary">{med.drug}</div>
                                                            <div className="text-sm text-dark-text-secondary">
                                                                {med.dose} • {med.time}
                                                            </div>
                                                        </div>
                                                        <input
                                                            type="checkbox"
                                                            className="w-5 h-5 rounded border-dark-border-primary"
                                                            onChange={(e) => toggleTask(`med-${index}`, e.target.checked)}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <p className="text-dark-text-tertiary italic">No medications scheduled</p>
                                    )}
                                </div>

                                {/* Exercises */}
                                <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                                    <div className="flex items-center gap-3 mb-4">
                                        <span className="text-3xl">🏋️</span>
                                        <h2 className="text-xl font-bold text-dark-text-primary">Exercise Plan</h2>
                                    </div>

                                    {plan.exercises?.length > 0 ? (
                                        <div className="space-y-3">
                                            {plan.exercises.map((exercise: any, index: number) => (
                                                <div
                                                    key={index}
                                                    className="bg-dark-bg-secondary rounded-lg p-4 border border-dark-border-secondary"
                                                >
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex-1">
                                                            <div className="font-medium text-dark-text-primary">{exercise.name}</div>
                                                            <div className="text-sm text-dark-text-secondary">
                                                                {exercise.reps} reps • {exercise.duration} • {exercise.time}
                                                            </div>
                                                            <div className="text-xs text-dark-text-tertiary mt-1">
                                                                Intensity: {exercise.intensity}
                                                            </div>
                                                        </div>
                                                        <input
                                                            type="checkbox"
                                                            className="w-5 h-5 rounded border-dark-border-primary"
                                                            onChange={(e) => toggleTask(`ex-${index}`, e.target.checked)}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <p className="text-dark-text-tertiary italic">No exercises scheduled</p>
                                    )}
                                </div>

                                {/* Diet Recommendations */}
                                <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                                    <div className="flex items-center gap-3 mb-4">
                                        <span className="text-3xl">🥗</span>
                                        <h2 className="text-xl font-bold text-dark-text-primary">Diet Recommendations</h2>
                                    </div>

                                    {plan.diet_recommendations?.length > 0 ? (
                                        <ul className="space-y-2">
                                            {plan.diet_recommendations.map((diet: string, index: number) => (
                                                <li key={index} className="flex items-start gap-2">
                                                    <span className="text-primary-500 mt-1">✓</span>
                                                    <span className="text-dark-text-secondary">{diet}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <p className="text-dark-text-tertiary italic">No diet recommendations</p>
                                    )}
                                </div>

                                {/* Health Tips */}
                                <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                                    <div className="flex items-center gap-3 mb-4">
                                        <span className="text-3xl">💡</span>
                                        <h2 className="text-xl font-bold text-dark-text-primary">Health Tips</h2>
                                    </div>

                                    {plan.health_tips?.length > 0 ? (
                                        <ul className="space-y-2">
                                            {plan.health_tips.map((tip: string, index: number) => (
                                                <li key={index} className="flex items-start gap-2">
                                                    <span className="text-secondary-500 mt-1">💡</span>
                                                    <span className="text-dark-text-secondary">{tip}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <p className="text-dark-text-tertiary italic">No health tips</p>
                                    )}
                                </div>

                                {/* Safety Approval */}
                                {safety.approved !== undefined && (
                                    <div className={`glass-strong rounded-2xl p-6 border ${safety.approved ? 'border-green-500' : 'border-red-500'
                                        }`}>
                                        <div className="flex items-center gap-3 mb-2">
                                            <span className="text-3xl">{safety.approved ? '✅' : '⚠️'}</span>
                                            <h2 className="text-xl font-bold text-dark-text-primary">Safety Check</h2>
                                        </div>
                                        <p className={`${safety.approved ? 'text-green-400' : 'text-red-400'}`}>
                                            {safety.approved ? 'Plan approved - safe for your conditions' : `Rejected: ${safety.risk_reason}`}
                                        </p>
                                    </div>
                                )}

                                {/* Reflection from Yesterday */}
                                {reflection.adherence_score !== undefined && (
                                    <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                                        <div className="flex items-center gap-3 mb-4">
                                            <span className="text-3xl">📈</span>
                                            <h2 className="text-xl font-bold text-dark-text-primary">Yesterday's Adherence</h2>
                                        </div>

                                        <div className="space-y-4">
                                            <div>
                                                <div className="flex justify-between text-sm text-dark-text-secondary mb-1">
                                                    <span>Overall Score</span>
                                                    <span>{reflection.adherence_score}%</span>
                                                </div>
                                                <div className="h-2 bg-dark-bg-tertiary rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-gradient-primary"
                                                        style={{ width: `${reflection.adherence_score}%` }}
                                                    />
                                                </div>
                                            </div>

                                            {reflection.lessons_learned?.length > 0 && (
                                                <div>
                                                    <div className="text-sm font-medium text-dark-text-secondary mb-2">Lessons Learned:</div>
                                                    <ul className="space-y-1">
                                                        {reflection.lessons_learned.map((lesson: string, index: number) => (
                                                            <li key={index} className="text-sm text-dark-text-tertiary">• {lesson}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* Plan Notes */}
                                {plan.notes && (
                                    <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary lg:col-span-2">
                                        <div className="flex items-center gap-3 mb-2">
                                            <span className="text-3xl">📝</span>
                                            <h2 className="text-xl font-bold text-dark-text-primary">AI Coach Notes</h2>
                                        </div>
                                        <p className="text-dark-text-secondary">{plan.notes}</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {/* History Tab */}
                {selectedTab === 'history' && (
                    <div className="space-y-4">
                        {history.length === 0 ? (
                            <div className="glass-strong rounded-2xl p-12 text-center border border-dark-border-primary">
                                <div className="text-6xl mb-4">📊</div>
                                <h2 className="text-2xl font-bold text-dark-text-primary mb-2">
                                    No History Yet
                                </h2>
                                <p className="text-dark-text-secondary">
                                    Your daily plan history will appear here
                                </p>
                            </div>
                        ) : (
                            history.map((record, index) => (
                                <div key={index} className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
                                    <div className="flex justify-between items-start mb-4">
                                        <div>
                                            <h3 className="text-lg font-bold text-dark-text-primary">
                                                {new Date(record.date).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                                            </h3>
                                            {record.reflection && (
                                                <p className="text-sm text-dark-text-secondary mt-1">
                                                    Adherence: {record.reflection.adherence_score}%
                                                </p>
                                            )}
                                        </div>
                                        <span className={`px-3 py-1 rounded-full text-sm ${record.safety_approval?.approved
                                                ? 'bg-green-500 bg-opacity-20 text-green-400'
                                                : 'bg-red-500 bg-opacity-20 text-red-400'
                                            }`}>
                                            {record.safety_approval?.approved ? 'Approved' : 'Rejected'}
                                        </span>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                                        <div>
                                            <div className="text-dark-text-tertiary mb-1">Medications</div>
                                            <div className="font-medium text-dark-text-primary">
                                                {record.daily_plan?.medication_reminders?.length || 0}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-dark-text-tertiary mb-1">Exercises</div>
                                            <div className="font-medium text-dark-text-primary">
                                                {record.daily_plan?.exercises?.length || 0}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-dark-text-tertiary mb-1">Completion</div>
                                            <div className="font-medium text-dark-text-primary">
                                                {record.reflection?.adherence_score || 0}%
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                )}
            </div>
        </DashboardLayout>
    );
}
