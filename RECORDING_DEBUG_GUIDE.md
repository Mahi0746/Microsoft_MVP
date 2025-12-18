# Recording Debug Guide

## Issue
Recording timer not progressing (stuck at 0:00) and audio not being captured properly.

## Fixes Applied

### 1. Timer Fix
**Problem**: Timer interval was being set but state wasn't updating properly.

**Solution**:
- Use local variable `seconds` in the interval instead of relying on state
- Log each timer tick to console for debugging
- Clear interval properly on stop

### 2. Audio Capture Fix
**Problem**: Audio chunks not being collected by MediaRecorder.

**Solution**:
- Use refs (`chunksRef`, `streamRef`, `recorderRef`) to maintain references
- Set timeslice to 100ms for frequent data events
- Add comprehensive logging for each step
- Proper cleanup of media stream

### 3. State Management Fix
**Problem**: Async state updates causing timing issues.

**Solution**:
- Use refs for immediate access to values
- Update state after starting recorder
- Proper cleanup in useEffect

## Testing Steps

### Step 1: Test with Simple HTML Page

1. **Open** `test-recording-simple.html` in your browser
2. **Click** "Start Recording"
3. **Allow** microphone access
4. **Watch** the timer - it should count: 0:01, 0:02, 0:03...
5. **Watch** the chunks counter - should increase as you speak
6. **Speak** clearly for 5-10 seconds
7. **Click** "Stop Recording"
8. **Check** the log for:
   - ✅ Microphone access granted
   - ✅ MediaRecorder started
   - ⏱️ Recording: 1 seconds, 2 seconds, etc.
   - 📦 Chunk received messages
   - ✅ Recording complete with size
9. **Click** "Play Recording" to verify audio was captured

**Expected Results**:
- Timer counts up every second
- Chunks counter increases (should have 10+ chunks for 5 seconds)
- Size shows bytes (should be 50,000+ bytes for 5 seconds)
- Audio plays back when you click play

### Step 2: Test in HealthSync App

1. **Start Backend**:
   ```bash
   python backend/main_complete.py
   ```

2. **Start Frontend** (new terminal):
   ```bash
   cd frontend/web
   npm run dev
   ```

3. **Open Browser**: http://localhost:3000/voice-doctor

4. **Test Recording**:
   - Click "Start Consultation"
   - Click "Start Recording"
   - Allow microphone
   - Watch timer count up
   - Speak for 5 seconds
   - Click "Stop Recording"
   - Check browser console for logs
   - Click "Send Audio"

5. **Check Console Logs**:
   ```
   🎤 Requesting microphone access...
   ✅ Microphone access granted
   🎵 Using MIME type: audio/webm;codecs=opus
   🎬 Starting recording...
   ▶️ MediaRecorder started successfully
   ⏱️ Starting timer...
   ⏱️ Recording time: 1 seconds
   📦 Data available: 4096 bytes
   📊 Total chunks: 1
   ⏱️ Recording time: 2 seconds
   📦 Data available: 4096 bytes
   📊 Total chunks: 2
   ...
   🛑 Stop recording requested...
   ⏱️ Timer stopped
   ⏹️ Stopping MediaRecorder...
   ⏹️ Recording stopped
   📦 Total chunks collected: 15
   📏 Total audio size: 61440 bytes
   ```

## Troubleshooting

### Timer Stuck at 0:00

**Possible Causes**:
1. JavaScript error preventing interval from running
2. State not updating
3. Component re-rendering issues

**Debug Steps**:
1. Open browser console (F12)
2. Look for JavaScript errors
3. Check if you see "⏱️ Recording time: X seconds" logs
4. If you see logs but timer doesn't update, it's a React state issue
5. Try the simple HTML test page to isolate the issue

**Fix**:
- The new code uses a local variable in the interval
- Logs every second to console
- Should work now

### No Audio Chunks

**Possible Causes**:
1. Microphone not working
2. Browser doesn't support MediaRecorder
3. Wrong MIME type
4. Microphone permissions denied

**Debug Steps**:
1. Check console for "📦 Data available" messages
2. If no messages, MediaRecorder isn't capturing
3. Try different browser (Chrome recommended)
4. Check microphone in system settings
5. Test with simple HTML page

**Fix**:
- Using 100ms timeslice for frequent data events
- Proper event handlers
- Comprehensive logging

### Microphone Permission Denied

**Solutions**:
1. **Chrome**: Click lock icon in address bar → Site settings → Microphone → Allow
2. **Firefox**: Click shield icon → Permissions → Microphone → Allow
3. **Edge**: Click lock icon → Permissions → Microphone → Allow
4. **System**: Check Windows sound settings → Privacy → Microphone

### Recording Works but Backend Fails

**Check**:
1. Backend is running on http://localhost:8000
2. Check backend console for errors
3. Audio file size > 1000 bytes
4. Network tab shows 200 OK response

## Browser Compatibility

### ✅ Fully Supported
- Chrome 49+
- Firefox 25+
- Edge 79+
- Safari 14.1+

### ❌ Not Supported
- Internet Explorer
- Old mobile browsers

### Check Support
Open console and run:
```javascript
console.log('MediaRecorder:', typeof MediaRecorder !== 'undefined');
console.log('getUserMedia:', !!navigator.mediaDevices?.getUserMedia);
```

## Console Logs Explained

### Good Logs (Working):
```
🎤 Requesting microphone access...
✅ Microphone access granted
🎵 Using MIME type: audio/webm;codecs=opus
🎬 Starting recording...
▶️ MediaRecorder started successfully
⏱️ Starting timer...
⏱️ Recording time: 1 seconds
📦 Data available: 4096 bytes
📊 Total chunks: 1
⏱️ Recording time: 2 seconds
📦 Data available: 4096 bytes
📊 Total chunks: 2
```

### Bad Logs (Not Working):
```
🎤 Requesting microphone access...
❌ Failed to start recording: NotAllowedError: Permission denied
```
OR
```
🎤 Requesting microphone access...
✅ Microphone access granted
🎵 Using MIME type: audio/webm;codecs=opus
🎬 Starting recording...
▶️ MediaRecorder started successfully
⏱️ Starting timer...
(no more logs - timer not working)
```

## Quick Fixes

### If Timer Doesn't Count:
1. Hard refresh browser (Ctrl+Shift+R)
2. Clear browser cache
3. Check console for errors
4. Try simple HTML test page

### If No Audio Chunks:
1. Check microphone permissions
2. Try different browser
3. Test microphone in other apps
4. Use simple HTML test page

### If Everything Looks Good but Still Fails:
1. Check audio blob size in console
2. Should be > 1000 bytes
3. Check network tab for upload
4. Check backend logs

## Success Indicators

### Visual:
- ✅ Timer counts: 0:01, 0:02, 0:03...
- ✅ Red recording indicator visible
- ✅ Green "Audio recorded" message after stop
- ✅ Audio size shown (e.g., "0:05")

### Console:
- ✅ Multiple "⏱️ Recording time" logs
- ✅ Multiple "📦 Data available" logs
- ✅ "📦 Total chunks collected: 15+" log
- ✅ "📏 Total audio size: 50000+" bytes log

### Backend:
- ✅ "POST /api/voice/send-audio HTTP/1.1" 200 OK
- ✅ "Audio content size: XXXXX bytes"
- ✅ "Transcription: [text]"

## Next Steps After Fix

1. ✅ Verify timer counts up
2. ✅ Verify audio chunks are collected
3. ✅ Verify audio can be sent to backend
4. ⚠️ Get Groq API key for real transcription
5. ✅ Test full voice consultation flow

## Support

If still having issues:
1. Run simple HTML test first
2. Check all console logs
3. Verify microphone works in other apps
4. Try different browser
5. Check system microphone permissions
