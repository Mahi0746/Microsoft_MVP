// HealthSync AI - Lifestyle Onboarding
// Animated 5-question assessment to determine health profile

import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/layout/DashboardLayout';
import { useAuthStore } from '../stores/authStore';

const questions = [
    {
        id: 'activity',
        question: 'How active are you?',
        icon: '🏃',
        options: [
            { value: 1, label: 'Mostly sitting/desk work', emoji: '💺' },
            { value: 2, label: 'Light activity (some walking)', emoji: '🚶' },
            { value: 3, label: 'Moderately active (regular exercise)', emoji: '🏋️' },
            { value: 4, label: 'Very active (daily intense exercise)', emoji: '⚡' },
        ],
    },
    {
        id: 'exercise',
        question: 'How often do you exercise?',
        icon: '💪',
        options: [
            { value: 1, label: 'Rarely/Never', emoji: '😴' },
            { value: 2, label: '1-2 times per week', emoji: '🌱' },
            { value: 3, label: '3-4 times per week', emoji: '🔥' },
            { value: 4, label: '5+ times per week', emoji: '⚡' },
        ],
    },
    {
        id: 'sleep',
        question: 'How would you rate your sleep?',
        icon: '😴',
        options: [
            { value: 1, label: 'Poor (often tired)', emoji: '😫' },
            { value: 2, label: 'Fair (sometimes tired)', emoji: '😐' },
            { value: 3, label: 'Good (usually rested)', emoji: '😊' },
            { value: 4, label: 'Excellent (always energized)', emoji: '✨' },
        ],
    },
    {
        id: 'stress',
        question: 'Your stress level?',
        icon: '🧘',
        options: [
            { value: 1, label: 'Very high stress', emoji: '😰' },
            { value: 2, label: 'Moderate stress', emoji: '😌' },
            { value: 3, label: 'Low stress', emoji: '😎' },
            { value: 4, label: 'Very relaxed', emoji: '🧘' },
        ],
    },
    {
        id: 'diet',
        question: "How's your diet?",
        icon: '🥗',
        options: [
            { value: 1, label: 'Mostly fast food/processed', emoji: '🍔' },
            { value: 2, label: 'Mixed (some healthy)', emoji: '🥙' },
            { value: 3, label: 'Mostly healthy', emoji: '🥗' },
            { value: 4, label: 'Very clean/nutritious', emoji: '🥦' },
        ],
    },
];

export default function OnboardingPage() {
    const router = useRouter();
    const { user } = useAuthStore();
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState<Record<string, number>>({});
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [showResult, setShowResult] = useState(false);
    const [profile, setProfile] = useState('');

    // Check if already completed
    useEffect(() => {
        checkOnboardingStatus();
    }, []);

    const checkOnboardingStatus = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/onboarding/status', {
                credentials: 'include',
            });
            const data = await response.json();

            if (data.completed) {
                // Already completed, redirect to AI Coach
                router.push('/ai-coach');
            }
        } catch (error) {
            console.error('Failed to check onboarding status:', error);
        }
    };

    const handleAnswer = (questionId: string, value: number) => {
        setAnswers({ ...answers, [questionId]: value });

        // Auto-advance after a short delay
        setTimeout(() => {
            if (currentQuestion < questions.length - 1) {
                setCurrentQuestion(currentQuestion + 1);
            } else {
                submitAnswers({ ...answers, [questionId]: value });
            }
        }, 300);
    };

    const submitAnswers = async (finalAnswers: Record<string, number>) => {
        setIsSubmitting(true);

        try {
            const response = await fetch('http://localhost:8000/api/onboarding/lifestyle-assessment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({
                    activity: finalAnswers.activity,
                    exercise: finalAnswers.exercise,
                    sleep: finalAnswers.sleep,
                    stress: finalAnswers.stress,
                    diet: finalAnswers.diet,
                }),
            });

            if (!response.ok) throw new Error('Failed to submit assessment');

            const data = await response.json();
            setProfile(data.profile);
            setShowResult(true);

            // Redirect to AI Coach after 3 seconds
            setTimeout(() => {
                router.push('/ai-coach');
            }, 3000);
        } catch (error) {
            console.error('Failed to submit assessment:', error);
            alert('Failed to save your answers. Please try again.');
            setIsSubmitting(false);
        }
    };

    const progress = ((currentQuestion + 1) / questions.length) * 100;

    if (!user) return null;

    if (showResult) {
        return (
            <DashboardLayout>
                <div className="min-h-screen flex items-center justify-center">
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="text-center"
                    >
                        <motion.div
                            initial={{ rotate: 0 }}
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1 }}
                            className="text-8xl mb-6"
                        >
                            ✅
                        </motion.div>
                        <h1 className="text-4xl font-bold text-gradient mb-4">
                            Profile Set!
                        </h1>
                        <p className="text-xl text-dark-text-secondary mb-2">
                            Your lifestyle profile: <span className="text-primary-500 font-bold uppercase">{profile}</span>
                        </p>
                        <p className="text-dark-text-tertiary">
                            Redirecting to AI Coach...
                        </p>
                    </motion.div>
                </div>
            </DashboardLayout>
        );
    }

    if (isSubmitting) {
        return (
            <DashboardLayout>
                <div className="min-h-screen flex items-center justify-center">
                    <div className="text-center">
                        <div className="animate-spin rounded-full h-16 w-16 border-4 border-primary-500 border-t-transparent mx-auto mb-4"></div>
                        <p className="text-dark-text-secondary">Processing your answers...</p>
                    </div>
                </div>
            </DashboardLayout>
        );
    }

    const currentQ = questions[currentQuestion];

    return (
        <DashboardLayout>
            <div className="max-w-3xl mx-auto py-12">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-gradient mb-4">
                        🎯 Lifestyle Assessment
                    </h1>
                    <p className="text-dark-text-secondary">
                        Answer 5 quick questions to personalize your health insights
                    </p>
                </div>

                {/* Progress Bar */}
                <div className="mb-8">
                    <div className="flex justify-between text-sm text-dark-text-tertiary mb-2">
                        <span>Question {currentQuestion + 1} of {questions.length}</span>
                        <span>{Math.round(progress)}%</span>
                    </div>
                    <div className="h-2 bg-dark-bg-tertiary rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-gradient-primary"
                            initial={{ width: 0 }}
                            animate={{ width: `${progress}%` }}
                            transition={{ duration: 0.3 }}
                        />
                    </div>
                </div>

                {/* Question Card */}
                <AnimatePresence mode="wait">
                    <motion.div
                        key={currentQuestion}
                        initial={{ x: 300, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: -300, opacity: 0 }}
                        transition={{ type: 'spring', stiffness: 200, damping: 25 }}
                        className="glass-strong rounded-2xl p-8 border border-dark-border-primary"
                    >
                        {/* Question Header */}
                        <div className="text-center mb-8">
                            <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ delay: 0.2, type: 'spring' }}
                                className="text-6xl mb-4"
                            >
                                {currentQ.icon}
                            </motion.div>
                            <h2 className="text-2xl font-bold text-dark-text-primary">
                                {currentQ.question}
                            </h2>
                        </div>

                        {/* Options */}
                        <div className="space-y-3">
                            {currentQ.options.map((option, index) => (
                                <motion.button
                                    key={option.value}
                                    initial={{ x: -50, opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    transition={{ delay: index * 0.1 }}
                                    onClick={() => handleAnswer(currentQ.id, option.value)}
                                    className={`w-full p-4 rounded-xl border-2 transition-all text-left flex items-center gap-4 ${answers[currentQ.id] === option.value
                                            ? 'border-primary-500 bg-primary-500 bg-opacity-10'
                                            : 'border-dark-border-primary bg-dark-bg-secondary hover:border-primary-400 hover:bg-dark-bg-hover'
                                        }`}
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    <span className="text-3xl">{option.emoji}</span>
                                    <span className="text-dark-text-primary font-medium flex-1">
                                        {option.label}
                                    </span>
                                    {answers[currentQ.id] === option.value && (
                                        <motion.span
                                            initial={{ scale: 0 }}
                                            animate={{ scale: 1 }}
                                            className="text-primary-500 text-xl"
                                        >
                                            ✓
                                        </motion.span>
                                    )}
                                </motion.button>
                            ))}
                        </div>

                        {/* Navigation Hint */}
                        {currentQuestion > 0 && (
                            <motion.button
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                onClick={() => setCurrentQuestion(currentQuestion - 1)}
                                className="mt-6 text-dark-text-tertiary hover:text-dark-text-primary transition-colors text-sm"
                            >
                                ← Back
                            </motion.button>
                        )}
                    </motion.div>
                </AnimatePresence>

                {/* Answered Questions Indicator */}
                <div className="flex justify-center gap-2 mt-8">
                    {questions.map((q, index) => (
                        <div
                            key={q.id}
                            className={`h-2 w-8 rounded-full transition-all ${index < currentQuestion
                                    ? 'bg-gradient-primary'
                                    : index === currentQuestion
                                        ? 'bg-primary-400'
                                        : 'bg-dark-bg-tertiary'
                                }`}
                        />
                    ))}
                </div>
            </div>
        </DashboardLayout>
    );
}
