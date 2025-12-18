# Voice Recording Fix - Complete Guide

## Problem Summary
The voice recording feature was showing 0 seconds and not capturing audio properly due to:
1. MediaRecorder chunks not being stored correctly (closure issue)
2. Missing error handling for audio capture
3. Need for real Groq API key for speech-to-text

## Fixes Applied

### 1. Frontend Audio Recording Fix (`frontend/web/src/pages/voice-doctor.tsx`)

**Problem**: The `audioChunks` array was being managed incorrectly, causing the MediaRecorder's `ondataavailable` event to lose reference to the chunks.

**Solution**: 
- Added `chunksRef` using `useRef` to maintain reference across MediaRecorder events
- Reduced timeslice from 1000ms to 100ms for more frequent data capture
- Added better error handling and logging
- Added validation for audio blob size before sending

**Key Changes**:
```typescript
// Added ref for chunks
const chunksRef = useRef<Blob[]>([]);

// In startRecording:
chunksRef.current = [];
recorder.ondataavailable = (event) => {
  if (event.data.size > 0) {
    chunksRef.current.push(event.data);
  }
};

// Start with smaller timeslice
recorder.start(100); // Request data every 100ms
```

### 2. Backend Configuration Fix (`backend/config_flexible.py`)

**Problem**: Pydantic was rejecting extra fields from `.env` file.

**Solution**: Added `extra = "ignore"` to Config class to allow extra environment variables.

### 3. Audio Processing Improvements

**Added**:
- Better MIME type detection and handling
- Audio size validation (minimum 1000 bytes)
- Detailed error messages
- Console logging for debugging

## Testing the Fix

### Option 1: Test with HTML Test Page

1. Open `test-voice-recording.html` in your browser
2. Click "Start Recording"
3. Allow microphone access
4. Speak for a few seconds
5. Click "Stop Recording"
6. Check the debug log for:
   - Data available events
   - Chunk count
   - Total audio size
7. Click "Play Recording" to verify audio was captured

### Option 2: Test in HealthSync App

1. **Start Backend**:
   ```bash
   cd backend
   python main_complete.py
   ```

2. **Start Frontend**:
   ```bash
   cd frontend/web
   npm run dev
   ```

3. **Test Voice Recording**:
   - Navigate to http://localhost:3000/voice-doctor
   - Click "Start Consultation"
   - Click "Start Recording"
   - Allow microphone access when prompted
   - Speak clearly for 3-5 seconds
   - Click "Stop Recording"
   - Check console for logs showing:
     - "Data available: X bytes"
     - "Total chunks so far: X"
     - "Recording stopped, total chunks: X"
   - Click "Send Audio"

## Getting Real Groq API Key (Required for Speech-to-Text)

### Why You Need It
The voice recording will work, but without a real Groq API key, the backend will return demo responses instead of actual speech-to-text transcription.

### How to Get Free Groq API Key

1. **Visit Groq Console**: https://console.groq.com
2. **Sign Up**: Create a free account (no credit card required)
3. **Create API Key**:
   - Go to "API Keys" section
   - Click "Create API Key"
   - Name it "healthsync-ai"
   - Copy the key (starts with `gsk_`)
4. **Update .env File**:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```
5. **Restart Backend**: Stop and restart `python main_complete.py`

### Groq Free Tier Limits
- **30 requests per minute** (plenty for testing)
- **Whisper Large V3** for speech-to-text
- **Llama 3.1 70B** for AI responses
- **100% FREE** - no credit card required

## Troubleshooting

### Issue: "Failed to access microphone"

**Solutions**:
1. **Check Browser Permissions**:
   - Chrome: Click lock icon in address bar → Site settings → Microphone → Allow
   - Firefox: Click shield icon → Permissions → Microphone → Allow
   
2. **Use HTTPS or localhost**:
   - MediaRecorder requires secure context
   - localhost is automatically secure
   - For remote access, use HTTPS

3. **Check Microphone Hardware**:
   - Test microphone in other apps
   - Check system sound settings
   - Ensure microphone is not muted

### Issue: "Audio blob is empty"

**Solutions**:
1. **Record for longer**: Speak for at least 2-3 seconds
2. **Check browser compatibility**: Use Chrome, Firefox, or Edge (latest versions)
3. **Check console logs**: Look for "Data available" messages
4. **Try test page**: Use `test-voice-recording.html` to isolate the issue

### Issue: "Invalid API Key" in backend

**Solutions**:
1. **Get real Groq API key**: See instructions above
2. **Check .env file**: Ensure `GROQ_API_KEY=gsk_...` (starts with gsk_)
3. **Restart backend**: Changes to .env require restart
4. **Demo mode**: App works in demo mode but won't do real transcription

### Issue: Recording shows 0 seconds

**Solutions**:
1. **Clear browser cache**: Hard refresh (Ctrl+Shift+R)
2. **Check console**: Look for JavaScript errors
3. **Update code**: Ensure you have the latest voice-doctor.tsx
4. **Test with HTML page**: Use test-voice-recording.html

## Browser Compatibility

### Fully Supported
- ✅ Chrome 49+
- ✅ Firefox 25+
- ✅ Edge 79+
- ✅ Safari 14.1+

### Not Supported
- ❌ Internet Explorer
- ❌ Old mobile browsers

## Audio Format Details

### Preferred Format
- **MIME Type**: `audio/webm;codecs=opus`
- **Codec**: Opus (best compression and quality)
- **Container**: WebM

### Fallback Formats
1. `audio/webm` (generic WebM)
2. `audio/mp4` (MP4 container)
3. `audio/wav` (uncompressed, large files)

### Backend Processing
- Backend accepts all formats
- Groq Whisper handles format conversion automatically
- No manual conversion needed

## Next Steps

1. ✅ **Test Audio Recording**: Use test-voice-recording.html
2. ✅ **Get Groq API Key**: Follow instructions above
3. ✅ **Update .env**: Add real Groq API key
4. ✅ **Restart Backend**: Apply new configuration
5. ✅ **Test Full Flow**: Record → Transcribe → AI Response

## Demo Mode vs Real Mode

### Demo Mode (No API Key)
- ✅ Audio recording works
- ✅ UI fully functional
- ❌ No real speech-to-text
- ❌ Generic AI responses
- ⚠️ Returns placeholder transcriptions

### Real Mode (With API Key)
- ✅ Audio recording works
- ✅ Real speech-to-text with Groq Whisper
- ✅ Intelligent AI responses with Llama 3.1
- ✅ Accurate medical consultation
- ✅ Full functionality

## Files Modified

1. `frontend/web/src/pages/voice-doctor.tsx` - Fixed audio recording
2. `backend/config_flexible.py` - Fixed config validation
3. `test-voice-recording.html` - Created test page
4. `VOICE_RECORDING_FIX.md` - This documentation

## Success Indicators

When everything is working correctly, you should see:

### In Browser Console:
```
Requesting microphone access...
Microphone access granted, creating MediaRecorder...
Using MIME type: audio/webm;codecs=opus
MediaRecorder started successfully
Recording started...
Data available: 4096 bytes
Total chunks so far: 1
Data available: 4096 bytes
Total chunks so far: 2
...
Stopping recording...
Recording stopped, total chunks: 15
Creating audio blob from 15 chunks
Audio blob size: 61440 bytes
Sending audio to backend...
Response status: 200
```

### In Backend Console:
```
INFO:     127.0.0.1:xxxxx - "POST /api/voice/send-audio HTTP/1.1" 200 OK
```

### In UI:
- Recording timer counts up (0:01, 0:02, 0:03...)
- Green "Audio recorded" message appears
- "Send Audio" button is enabled
- After sending, user message and AI response appear in chat

## Support

If you still have issues:
1. Check browser console for errors
2. Check backend logs for errors
3. Try the test-voice-recording.html page
4. Verify microphone permissions
5. Ensure you're using localhost or HTTPS
6. Get a real Groq API key for full functionality
