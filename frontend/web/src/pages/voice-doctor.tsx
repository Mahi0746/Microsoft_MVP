import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const VoiceDoctorPage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [conversation, setConversation] = useState<Array<{role: string, message: string, timestamp: string}>>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingInterval = useRef<NodeJS.Timeout | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
    }
  }, [user, loading, router]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      console.log('🧹 Cleaning up voice doctor component...');
      
      // Clear timer
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
        recordingInterval.current = null;
      }
      
      // Stop recorder
      if (recorderRef.current && recorderRef.current.state === 'recording') {
        recorderRef.current.stop();
      }
      
      // Stop stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  const startSession = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/voice/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: user?.id || 'demo_user',
          symptoms: ['general consultation']
        }),
      });

      const data = await response.json();
      if (data.success) {
        setSessionId(data.session_id);
        setConversation([{
          role: 'ai',
          message: data.ai_response,
          timestamp: new Date().toISOString()
        }]);
      }
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  };

  const startRecording = async () => {
    try {
      console.log('🎤 Requesting microphone access...');
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      console.log('✅ Microphone access granted');
      streamRef.current = stream;
      
      // Check supported MIME types
      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        mimeType = 'audio/mp4';
      } else if (MediaRecorder.isTypeSupported('audio/wav')) {
        mimeType = 'audio/wav';
      }
      
      console.log('🎵 Using MIME type:', mimeType);
      
      // Create MediaRecorder
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;
      
      // Clear previous chunks
      chunksRef.current = [];
      setAudioChunks([]);
      setRecordingTime(0);
      
      // Setup event handlers
      recorder.ondataavailable = (event) => {
        console.log('📦 Data available:', event.data.size, 'bytes');
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
          console.log('📊 Total chunks:', chunksRef.current.length);
        }
      };
      
      recorder.onstop = () => {
        console.log('⏹️ Recording stopped');
        console.log('📦 Total chunks collected:', chunksRef.current.length);
        
        const totalSize = chunksRef.current.reduce((sum, chunk) => sum + chunk.size, 0);
        console.log('📏 Total audio size:', totalSize, 'bytes');
        
        setAudioChunks([...chunksRef.current]);
        
        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => {
            track.stop();
            console.log('🛑 Stopped track:', track.kind);
          });
          streamRef.current = null;
        }
      };
      
      recorder.onerror = (event: Event) => {
        console.error('❌ MediaRecorder error:', event);
        alert('Recording error occurred. Please try again.');
      };
      
      recorder.onstart = () => {
        console.log('▶️ MediaRecorder started successfully');
      };
      
      // Start recording with timeslice
      console.log('🎬 Starting recording...');
      recorder.start(100); // Request data every 100ms
      
      // Update state
      setMediaRecorder(recorder);
      setIsRecording(true);
      
      // Start timer AFTER setting recording state
      console.log('⏱️ Starting timer...');
      let seconds = 0;
      recordingInterval.current = setInterval(() => {
        seconds++;
        console.log('⏱️ Recording time:', seconds, 'seconds');
        setRecordingTime(seconds);
      }, 1000);
      
      console.log('✅ Recording started successfully');
      
    } catch (error) {
      console.error('❌ Failed to start recording:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Failed to access microphone: ${errorMessage}\n\nPlease:\n1. Allow microphone access\n2. Use Chrome, Firefox, or Edge\n3. Ensure you're on localhost or HTTPS`);
      
      // Cleanup on error
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  };

  const stopRecording = () => {
    console.log('🛑 Stop recording requested...');
    
    // Stop timer first
    if (recordingInterval.current) {
      clearInterval(recordingInterval.current);
      recordingInterval.current = null;
      console.log('⏱️ Timer stopped');
    }
    
    // Stop recorder
    if (recorderRef.current && recorderRef.current.state === 'recording') {
      console.log('⏹️ Stopping MediaRecorder...');
      recorderRef.current.stop();
    }
    
    setIsRecording(false);
    console.log('✅ Recording stopped');
  };

  const sendAudioMessage = async () => {
    if (audioChunks.length === 0 || !sessionId) {
      console.log('No audio chunks or session ID');
      alert('No audio recorded. Please record your message first.');
      return;
    }

    setIsProcessing(true);

    try {
      console.log('Creating audio blob from', audioChunks.length, 'chunks');
      
      // Determine MIME type based on what was recorded
      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      }
      
      // Create audio blob with proper MIME type
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      
      console.log('Audio blob size:', audioBlob.size, 'bytes');
      console.log('Audio blob type:', audioBlob.type);
      
      if (audioBlob.size === 0) {
        throw new Error('Audio recording is empty. Please try recording again.');
      }
      
      if (audioBlob.size < 1000) {
        throw new Error('Audio recording is too short. Please record for at least 1 second.');
      }
      
      // Create form data
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'recording.webm');
      formData.append('session_id', sessionId);
      formData.append('user_id', user?.id || 'demo_user');

      console.log('📤 Sending audio to backend...');
      console.log('FormData contents:');
      console.log('  - audio_file:', audioBlob.size, 'bytes, type:', audioBlob.type);
      console.log('  - session_id:', sessionId);
      console.log('  - user_id:', user?.id || 'demo_user');
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      console.log('API URL:', `${apiUrl}/api/voice/send-audio`);
      
      const response = await fetch(`${apiUrl}/api/voice/send-audio`, {
        method: 'POST',
        body: formData,
      });

      console.log('📥 Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Server error response:', errorText);
        throw new Error(`Server error: ${response.status} ${response.statusText}\n${errorText}`);
      }
      
      const data = await response.json();
      console.log('✅ Response data:', data);
      
      if (data.success && data.conversation) {
        // Add both user and AI messages to conversation
        setConversation(prev => [...prev, ...data.conversation]);
      } else {
        throw new Error(data.message || 'Failed to process audio');
      }
    } catch (error) {
      console.error('Failed to send audio:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Failed to process audio: ${errorMessage}. Please try again.`);
    } finally {
      setIsProcessing(false);
      setAudioChunks([]);
      setRecordingTime(0);
    }
  };

  const sendTextMessage = async () => {
    if (!currentMessage.trim() || !sessionId) return;

    const userMessage = currentMessage.trim();
    setCurrentMessage('');
    setIsProcessing(true);

    // Add user message to conversation
    const newUserMessage = {
      role: 'user',
      message: userMessage,
      timestamp: new Date().toISOString()
    };
    setConversation(prev => [...prev, newUserMessage]);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/voice/send-message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage
        }),
      });

      const data = await response.json();
      if (data.success) {
        const aiMessage = {
          role: 'ai',
          message: data.ai_response,
          timestamp: new Date().toISOString()
        };
        setConversation(prev => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gradient">Voice AI Doctor</h1>
              <p className="text-dark-text-secondary mt-2">
                Real-time medical consultation with AI-powered voice analysis
              </p>
            </div>
            <div className="flex items-center space-x-4">
              {!sessionId ? (
                <button
                  onClick={startSession}
                  className="bg-gradient-primary hover:opacity-90 text-white px-6 py-3 rounded-xl font-semibold shadow-lg glow-blue transition-all"
                >
                  Start Consultation
                </button>
              ) : (
                <div className="flex items-center space-x-3 px-4 py-2 bg-green-900 bg-opacity-30 border border-green-500 rounded-xl">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse glow-green"></div>
                  <span className="text-sm text-green-400 font-medium">Session Active</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Chat Interface */}
        {sessionId && (
          <div className="glass-strong rounded-2xl overflow-hidden border border-dark-border-primary">
            {/* Conversation */}
            <div className="h-[500px] overflow-y-auto p-6 space-y-4 bg-dark-bg-secondary">
              {conversation.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-xl ${
                      message.role === 'user'
                        ? 'bg-gradient-primary text-white shadow-lg glow-blue'
                        : 'bg-dark-bg-tertiary text-dark-text-primary border border-dark-border-primary'
                    }`}
                  >
                    <p className="text-sm leading-relaxed">{message.message}</p>
                    <p className={`text-xs mt-2 ${
                      message.role === 'user' ? 'text-blue-100' : 'text-dark-text-secondary'
                    }`}>
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-dark-bg-tertiary text-dark-text-primary px-4 py-3 rounded-xl border border-dark-border-primary">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500"></div>
                      <span className="text-sm">AI is analyzing...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Voice Recording Controls */}
            <div className="border-t border-dark-border-primary p-6 bg-dark-bg-primary">
              {/* Recording Status */}
              {isRecording && (
                <div className="mb-4 p-4 bg-red-900 bg-opacity-30 border border-red-500 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse glow-green"></div>
                      <span className="text-red-400 font-semibold">Recording...</span>
                    </div>
                    <span className="text-red-300 font-mono text-lg">{formatTime(recordingTime)}</span>
                  </div>
                </div>
              )}

              {/* Audio Ready to Send */}
              {audioChunks.length > 0 && !isRecording && (
                <div className="mb-4 p-4 bg-green-900 bg-opacity-30 border border-green-500 rounded-xl">
                  <div className="flex items-center justify-between">
                    <span className="text-green-400 font-semibold">
                      Audio recorded ({formatTime(recordingTime)})
                    </span>
                    <div className="flex space-x-2">
                      <button
                        onClick={() => {
                          setAudioChunks([]);
                          setRecordingTime(0);
                        }}
                        className="text-dark-text-secondary hover:text-dark-text-primary px-4 py-2 text-sm rounded-lg bg-dark-bg-tertiary border border-dark-border-primary transition-all"
                      >
                        Discard
                      </button>
                      <button
                        onClick={sendAudioMessage}
                        disabled={isProcessing}
                        className="bg-gradient-to-r from-green-600 to-emerald-600 hover:opacity-90 disabled:opacity-50 text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-lg transition-all"
                      >
                        {isProcessing ? 'Processing...' : 'Send Audio'}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Voice Controls */}
              <div className="flex items-center space-x-4 mb-4">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={isProcessing}
                    className="flex items-center space-x-2 bg-gradient-to-r from-red-600 to-pink-600 hover:opacity-90 disabled:opacity-50 text-white px-5 py-3 rounded-xl font-semibold shadow-lg transition-all"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="flex items-center space-x-2 bg-dark-bg-tertiary hover:bg-dark-bg-hover text-dark-text-primary px-5 py-3 rounded-xl font-semibold border border-dark-border-primary transition-all"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                    </svg>
                    <span>Stop Recording</span>
                  </button>
                )}
                
                <div className="text-sm text-dark-text-secondary">
                  or type your message below
                </div>
              </div>

              {/* Text Input */}
              <div className="flex space-x-4">
                <input
                  type="text"
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message or describe your symptoms..."
                  className="flex-1 bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  disabled={isProcessing || isRecording}
                />
                <button
                  onClick={sendTextMessage}
                  disabled={!currentMessage.trim() || isProcessing || isRecording}
                  className="bg-gradient-primary hover:opacity-90 disabled:opacity-50 text-white px-6 py-3 rounded-xl font-semibold shadow-lg glow-blue transition-all"
                >
                  Send Text
                </button>
              </div>
              
              <p className="text-xs text-dark-text-secondary mt-3">
                🎤 Click "Start Recording" to speak your symptoms, or type them below. 
                This uses real speech-to-text with AI analysis.
              </p>
            </div>
          </div>
        )}

        {/* Features */}
        {!sessionId && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="glass-strong p-6 rounded-2xl border border-dark-border-primary hover:border-primary-500 transition-all">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 w-14 h-14 rounded-xl flex items-center justify-center mb-4 shadow-lg glow-blue">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-dark-text-primary mb-2">Voice Analysis</h3>
              <p className="text-dark-text-secondary text-sm leading-relaxed">
                Advanced AI analyzes your voice patterns and symptoms to provide personalized health insights.
              </p>
            </div>

            <div className="glass-strong p-6 rounded-2xl border border-dark-border-primary hover:border-primary-500 transition-all">
              <div className="bg-gradient-to-br from-green-500 to-emerald-600 w-14 h-14 rounded-xl flex items-center justify-center mb-4 shadow-lg glow-green">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-dark-text-primary mb-2">Instant Consultation</h3>
              <p className="text-dark-text-secondary text-sm leading-relaxed">
                Get immediate responses to your health questions with our 24/7 AI medical assistant.
              </p>
            </div>

            <div className="glass-strong p-6 rounded-2xl border border-dark-border-primary hover:border-primary-500 transition-all">
              <div className="bg-gradient-to-br from-purple-500 to-purple-600 w-14 h-14 rounded-xl flex items-center justify-center mb-4 shadow-lg glow-purple">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-dark-text-primary mb-2">Health Tracking</h3>
              <p className="text-dark-text-secondary text-sm leading-relaxed">
                All consultations are saved to your health profile for continuous monitoring and analysis.
              </p>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default VoiceDoctorPage;