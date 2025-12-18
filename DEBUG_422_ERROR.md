# Debugging 422 Validation Error

## Current Issue
Backend is returning 422 Unprocessable Content when trying to upload audio.

## What 422 Means
- HTTP 422 = Validation Error
- FastAPI couldn't validate the request parameters
- Either missing required fields or wrong data types

## Fixes Applied

### 1. Added Custom Exception Handler
**File**: `backend/main_complete.py`

Now logs validation errors to console so we can see what's wrong:
```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}")
    logger.error(f"Errors: {errors}")
    # Returns clean error without binary data
```

### 2. Made Parameters More Flexible
**File**: `backend/api/routes/voice.py`

Changed from required to optional with defaults:
```python
session_id: str = Form(default="")  # Was Form(...)
user_id: str = Form(default="demo_user")
```

### 3. Added Test Endpoint
**File**: `backend/api/routes/voice.py`

New endpoint `/api/voice/test-upload` to debug uploads:
```python
@router.post("/test-upload")
async def test_audio_upload(...)
```

### 4. Better Frontend Logging
**File**: `frontend/web/src/pages/voice-doctor.tsx`

Now logs all FormData contents before sending.

## How to Debug

### Step 1: Restart Backend
```bash
# Stop current backend (Ctrl+C)
python backend/main_complete.py
```

### Step 2: Check Backend Logs
When you try to upload, backend should now show:
```
ERROR:    Validation error on /api/voice/send-audio
ERROR:    Errors: [{'type': '...', 'loc': ['...'], 'msg': '...'}]
```

This will tell us exactly what's wrong!

### Step 3: Run Test Script
```bash
python test_upload_debug.py
```

This tests both:
1. Test endpoint (simpler)
2. Real endpoint (full flow)

### Step 4: Check Browser Console
Open browser console (F12) and look for:
```
📤 Sending audio to backend...
FormData contents:
  - audio_file: 61440 bytes, type: audio/webm
  - session_id: <uuid>
  - user_id: demo_user
API URL: http://localhost:8000/api/voice/send-audio
```

## Common Causes of 422

### 1. Empty session_id
**Symptom**: session_id is empty string or null
**Fix**: Make sure session is started before recording
**Check**: Look for "session_id: " in console (empty after colon)

### 2. Missing audio_file
**Symptom**: No file in FormData
**Fix**: Make sure audioChunks has data
**Check**: Look for "audio_file: 0 bytes" in console

### 3. Wrong Content-Type
**Symptom**: FastAPI can't parse multipart/form-data
**Fix**: Don't set Content-Type header (let browser set it)
**Check**: Make sure fetch doesn't have headers: {'Content-Type': ...}

### 4. CORS Issue
**Symptom**: Preflight request fails
**Fix**: Check CORS settings in backend
**Check**: Look for OPTIONS request in Network tab

## What to Look For

### In Backend Console:
```
ERROR:    Validation error on /api/voice/send-audio
ERROR:    Errors: [
  {
    'type': 'missing',
    'loc': ['body', 'session_id'],
    'msg': 'Field required'
  }
]
```

This tells us: `session_id` is missing from the request body!

### In Browser Console:
```
📤 Sending audio to backend...
FormData contents:
  - audio_file: 61440 bytes, type: audio/webm
  - session_id: abc-123-def-456
  - user_id: demo_user
📥 Response status: 422
❌ Server error response: {"detail":[...],"message":"Validation error..."}
```

### In Network Tab (F12 → Network):
1. Find the POST request to `/api/voice/send-audio`
2. Click on it
3. Check "Headers" tab:
   - Request Method: POST
   - Content-Type: multipart/form-data; boundary=...
4. Check "Payload" tab:
   - Should show audio_file, session_id, user_id

## Quick Tests

### Test 1: Simple Upload
```bash
python test_upload_debug.py
```

Expected output:
```
✅ Test upload works!
✅ Session started: <uuid>
✅ Audio upload works!
```

### Test 2: Browser Test
1. Open http://localhost:3000/voice-doctor
2. Open console (F12)
3. Start consultation
4. Start recording
5. Stop recording
6. Click "Send Audio"
7. **Check console for detailed logs**
8. **Check backend console for validation errors**

## Solutions Based on Error

### If "session_id field required":
**Problem**: session_id not being sent
**Solution**: 
1. Check sessionId state is not null
2. Check FormData.append('session_id', sessionId)
3. Make sure session was started first

### If "audio_file field required":
**Problem**: File not being sent
**Solution**:
1. Check audioBlob.size > 0
2. Check FormData.append('audio_file', audioBlob, 'recording.webm')
3. Make sure recording actually captured audio

### If "value is not a valid string":
**Problem**: Sending wrong data type
**Solution**:
1. Make sure session_id is string, not object
2. Make sure user_id is string, not object
3. Check FormData values are correct types

## Next Steps

1. ✅ Restart backend with new logging
2. ✅ Try to upload audio
3. ✅ Check backend console for validation error details
4. ✅ Check browser console for FormData contents
5. ✅ Run test_upload_debug.py
6. ✅ Share the validation error details

Once we see the exact validation error, we can fix it immediately!
