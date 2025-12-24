import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';

interface VoiceSession {
  _id?: string;
  session_id?: string;
  user_id: string;
  status: string;
  conversation: Array<{ role: string, message: string, timestamp: string }>;
  created_at: string;
  symptoms?: string[];
}

const VoiceDoctorPage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [conversation, setConversation] = useState<Array<{ role: string, message: string, timestamp: string }>>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const recognitionRef = useRef<any>(null); // Use any for webkitSpeechRecognition

  // Session history state
  const [previousSessions, setPreviousSessions] = useState<VoiceSession[]>([]);
  const [showHistory, setShowHistory] = useState(true);

  // Fetch previous sessions on load
  const fetchPreviousSessions = async () => {
    if (!user?.id) return;
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/voice/user-sessions/${user.id}`);
      const data = await response.json();
      if (data.success) {
        setPreviousSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('Failed to fetch previous sessions:', error);
    }
  };

  useEffect(() => {
    if (!loading && !user) {
      router.push('/');
    }
    if (user) {
      fetchPreviousSessions();
    }
  }, [user, loading, router]);

  // Initialize Speech Recognition
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false; // Stop after one command for "Siri-like" interaction
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          console.log('🎤 Heard:', transcript);
          handleVoiceCommand(transcript);
        };

        recognition.onerror = (event: any) => {
          console.error('❌ Speech recognition error:', event.error);
          setIsListening(false);
        };

        recognition.onend = () => {
          setIsListening(false);
        };

        recognitionRef.current = recognition;
      } else {
        console.warn('Speech recognition not supported in this browser.');
      }
    }
  }, []);

  const handleVoiceCommand = async (text: string) => {
    setIsListening(false); // UI update

    // Add user message to conversation immediately
    const userMsg = {
      role: 'user',
      message: text,
      timestamp: new Date().toISOString()
    };
    setConversation(prev => [...prev, userMsg]);
    setIsProcessing(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      // Use the new /command endpoint
      const response = await fetch(`${apiUrl}/api/voice/command`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          user_id: user?.id || 'demo_user',
          session_id: sessionId || undefined,
          context: {
            patient_id: user?.id,
            name: user?.firstName || 'User',
            role: user?.role || 'patient'
          }
        }),
      });

      const data = await response.json();

      if (data.success && data.result) {
        const result = data.result;

        // Handle Action (Navigation)
        if (result.type === 'navigate' || (result.action === 'navigate')) {
          // We can assume result.message is the spoken response
          const aiMsg = {
            role: 'ai',
            message: result.message || `Navigating to ${result.target}...`,
            timestamp: new Date().toISOString()
          };
          setConversation(prev => [...prev, aiMsg]);

          // Perform client-side navigation after a short delay to let user read
          setTimeout(() => {
            const target = result.target || 'home';
            if (target === 'consultant') {
              // stay here or maybe open a modal? 
              // for now, we are already on 'voice-doctor' which is the consultant page
            } else {
              router.push(`/${target}`);
            }
          }, 1500);

        } else if (result.action === 'display') {
          // Show data cards
          const aiMsg = {
            role: 'ai',
            message: result.message,
            timestamp: new Date().toISOString()
          };
          setConversation(prev => [...prev, aiMsg]);

          if (result.target === 'medication' && result.data) {
            // Render specialized card in chat
            const medsList = result.data.map((m: any) => `- ${m.name} (${m.dosage})`).join('\n');
            setConversation(prev => [...prev, {
              role: 'system',
              message: `💊 **Medications:**\n${medsList}`,
              timestamp: new Date().toISOString()
            }]);
          } else if (result.target === 'appointments' && result.data) {
            const apptsList = result.data.map((a: any) => `- ${a.date} at ${a.time} with ${a.doctor}`).join('\n');
            setConversation(prev => [...prev, {
              role: 'system',
              message: `📅 **Appointments:**\n${apptsList}`,
              timestamp: new Date().toISOString()
            }]);
          }

        } else {
          // Normal message
          const aiMsg = {
            role: 'ai',
            message: typeof result.message === 'string' ? result.message : JSON.stringify(result.message),
            timestamp: new Date().toISOString()
          };
          setConversation(prev => [...prev, aiMsg]);
        }

      }
    } catch (error) {
      console.error('Failed to process command:', error);
      setConversation(prev => [...prev, {
        role: 'system',
        message: 'Sorry, I encountered an error processing that command.',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const startListening = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (e) {
        console.error("Recognition start error", e);
      }
    } else {
      alert("Speech recognition is not supported in this browser. Try Chrome/Edge.");
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const handleSendText = () => {
    if (currentMessage.trim()) {
      handleVoiceCommand(currentMessage);
      setCurrentMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
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
      <div className="flex gap-6 h-[calc(100vh-100px)]">
        {/* Session History Sidebar - Integrated clean look */}
        <div className={`${showHistory ? 'w-80' : 'w-12'} flex-shrink-0 transition-all duration-300 hidden md:block`}>
          <div className="glass-strong rounded-2xl border border-dark-border-primary h-full flex flex-col">
            <div className="p-4 border-b border-dark-border-primary flex items-center justify-between">
              {showHistory && (
                <h3 className="text-lg font-semibold text-dark-text-primary">Medical History</h3>
              )}
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="p-2 hover:bg-dark-bg-hover rounded-lg transition-all"
              >
                <svg className={`w-5 h-5 text-dark-text-secondary transform transition-transform ${showHistory ? '' : 'rotate-180'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </button>
            </div>

            {showHistory && (
              <div className="p-4 space-y-3 overflow-y-auto flex-1 custom-scrollbar">
                {previousSessions.length > 0 ? (
                  previousSessions.map((session) => (
                    <button
                      key={session.session_id}
                      onClick={() => {
                        setSessionId(session.session_id || session._id || null);
                        setConversation(session.conversation || []);
                      }}
                      className={`w-full text-left p-3 rounded-xl border transition-all hover:bg-dark-bg-hover ${sessionId === session.session_id
                        ? 'border-primary-500 bg-primary-900 bg-opacity-20'
                        : 'border-dark-border-primary bg-dark-bg-tertiary'
                        }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span className="text-sm font-medium text-dark-text-primary truncate">
                          Consultation {session.created_at ? new Date(session.created_at).toLocaleDateString() : 'Draft'}
                        </span>
                      </div>
                    </button>
                  ))
                ) : (
                  <p className="text-sm text-dark-text-secondary text-center py-4">
                    No previous consultations
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Main Interface */}
        <div className="flex-1 flex flex-col h-full max-w-5xl mx-auto">
          {/* Header */}
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gradient">AI Health Assistant</h1>
              <p className="text-dark-text-secondary mt-1">
                Your personal medical expert. Ask about medications, appointments, or symptoms.
              </p>
            </div>
            {!sessionId && conversation.length === 0 && (
              <div className="hidden md:block">
                <span className="bg-primary-500/10 text-primary-400 px-3 py-1 rounded-full text-sm border border-primary-500/20">
                  Powered by Llama 3.1 & Groq
                </span>
              </div>
            )}
          </div>

          {/* Chat / Interaction Area */}
          <div className="flex-1 glass-strong rounded-2xl border border-dark-border-primary flex flex-col overflow-hidden relative">

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
              {conversation.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 opacity-60">
                  <div className="w-24 h-24 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center mb-6 animate-pulse">
                    <svg className="w-12 h-12 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </div>
                  <h3 className="text-2xl font-semibold text-dark-text-primary mb-3">
                    "Hey, how can I help you today?"
                  </h3>
                  <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                    {["Show my medications", "When is my next appointment?", "I have a headache", "Navigate to marketplace"].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => handleVoiceCommand(suggestion)}
                        className="px-4 py-2 bg-dark-bg-tertiary hover:bg-dark-bg-hover border border-dark-border-primary rounded-full text-sm text-dark-text-secondary transition-all"
                      >
                        "{suggestion}"
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {conversation.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`
                        max-w-[80%] rounded-2xl px-5 py-3 shadow-lg
                        ${msg.role === 'user'
                      ? 'bg-gradient-primary text-white rounded-br-none'
                      : msg.role === 'system'
                        ? 'bg-dark-bg-tertiary border border-dashed border-dark-border-primary w-full max-w-full text-dark-text-secondary font-mono text-sm'
                        : 'bg-dark-bg-secondary border border-dark-border-primary text-dark-text-primary rounded-bl-none'
                    }
                    `}>
                    {msg.role === 'system' ? (
                      <pre className="whitespace-pre-wrap font-sans">{msg.message}</pre>
                    ) : (
                      <p className="leading-relaxed">{msg.message}</p>
                    )}
                    <span className="text-[10px] opacity-50 mt-1 block tracking-wider">
                      {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </div>
              ))}

              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-dark-bg-secondary px-5 py-4 rounded-2xl rounded-bl-none border border-dark-border-primary flex items-center space-x-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm text-dark-text-secondary">Thinking...</span>
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-dark-bg-primary/80 backdrop-blur border-t border-dark-border-primary">
              <div className="flex items-center gap-3">
                <button
                  onClick={isListening ? stopListening : startListening}
                  className={`
                           p-4 rounded-full transition-all duration-300 shadow-lg flex-shrink-0
                           ${isListening
                      ? 'bg-red-500 hover:bg-red-600 animate-pulse ring-4 ring-red-500/20'
                      : 'bg-gradient-primary hover:opacity-90 ring-4 ring-primary-500/20'
                    }
                       `}
                >
                  {isListening ? (
                    <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  ) : (
                    <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                      <line x1="12" y1="19" x2="12" y2="23"></line>
                      <line x1="8" y1="23" x2="16" y2="23"></line>
                    </svg>
                  )}
                </button>

                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={isListening ? "Listening..." : "Type a message or press the mic..."}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-5 py-4 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50 transition-all placeholder-dark-text-tertiary"
                    disabled={isProcessing}
                  />
                  <button
                    onClick={handleSendText}
                    disabled={!currentMessage.trim() || isProcessing}
                    className="absolute right-2 top-2 p-2 text-primary-400 hover:text-primary-300 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                  >
                    <svg className="w-6 h-6 transform rotate-90" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409 8.75 8.75 0 1113.322 0 1 1 0 001.17-1.41l-7-14z" />
                    </svg>
                  </button>
                </div>
              </div>
              <p className="text-center text-xs text-dark-text-tertiary mt-2">
                {isListening ? "Speak clearly now..." : "Press the microphone to start speaking"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default VoiceDoctorPage;