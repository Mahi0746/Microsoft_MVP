# HealthSync AI - Quick Start Guide

## ⚡ Quick Setup (5 Minutes)

This guide will get you up and running quickly with simplified MongoDB-only authentication.

### Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB (local or Atlas)

---

## 🚀 Backend Setup

### 1. Open Terminal in Backend Directory

```powershell
cd c:\Microsoft_Hackathon\new\Microsoft_MVP\backend
```

### 2. Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create .env File

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Edit `.env` with these minimum settings:

```env
# REQUIRED - Change this!
SECRET_KEY=your-super-secret-key-change-this-32-characters-minimum

# MongoDB (use local or Atlas)
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=healthsync

# AI Keys (get from providers - optional for auth testing)
GROQ_API_KEY=your-key-here
REPLICATE_API_TOKEN=your-token-here
HUGGINGFACE_API_KEY=your-key-here
```

### 5. Start Backend

```powershell
python main.py
```

✅ Backend running at: http://localhost:8000  
✅ API Docs: http://localhost:8000/docs

---

## 🎨 Frontend Setup

### 1. Open New Terminal in Frontend Directory

```powershell
cd c:\Microsoft_Hackathon\new\Microsoft_MVP\frontend\web
```

### 2. Install Dependencies

```powershell
npm install
```

### 3. Create .env.local File

```powershell
copy .env.example .env.local
```

The default content should be:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Start Frontend

```powershell
npm run dev
```

✅ Frontend running at: http://localhost:3000

---

## 🧪 Test Authentication

### Using the Web UI

1. Open http://localhost:3000/auth/register
2. Fill in the form:
   - First Name: Test
   - Last Name: User
   - Email: test@example.com
   - Password: Test1234
   - Role: Patient (or Doctor/Admin)
3. Click "Create Account"
4. You'll be automatically logged in!

### Using API Directly

**Signup:**

```powershell
curl -X POST http://localhost:8000/api/auth/signup `
  -H "Content-Type: application/json" `
  -d '{
    \"email\": \"test@example.com\",
    \"password\": \"Test1234\",
    \"first_name\": \"Test\",
    \"last_name\": \"User\",
    \"role\": \"patient\"
  }'
```

**Login:**

```powershell
curl -X POST http://localhost:8000/api/auth/login `
  -H "Content-Type: application/json" `
  -d '{
    \"email\": \"test@example.com\",
    \"password\": \"Test1234\"
  }'
```

You'll get a response with tokens:

```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

## 📦 MongoDB Setup Options

### Option 1: Local MongoDB (Easiest)

1. Download and install MongoDB Community Edition
2. Start MongoDB service
3. Use `mongodb://localhost:27017` in `.env`

### Option 2: MongoDB Atlas (Recommended)

1. Go to https://www.mongodb.com/cloud/atlas
2. Create free account and cluster
3. Create database user
4. Whitelist your IP (or use 0.0.0.0/0 for dev)
5. Get connection string from Atlas
6. Update `.env`:

```env
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

---

## ✅ Verify Everything Works

1. ✅ Backend running (http://localhost:8000/health)
2. ✅ Frontend running (http://localhost:3000)
3. ✅ Can register new user
4. ✅ Can login
5. ✅ No CORS errors in browser console

---

## 🔧 Common Issues

### "Port already in use"

**Backend:**
```powershell
# Change port in .env
PORT=8001

# Or run with different port
uvicorn main:app --port 8001
```

**Frontend:**
```powershell
npm run dev -- -p 3001
# Then update NEXT_PUBLIC_API_URL if needed
```

### "MongoDB connection failed"

- Check MongoDB is running locally
- OR verify MongoDB Atlas connection string
- Check firewall/network settings

### "Module not found" errors

```powershell
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend/web
npm install
```

### CORS Errors

- Ensure backend is running
- Check `ALLOWED_ORIGINS` in backend `.env`
- Development mode allows all origins by default

---

## 🎯 Next Steps

Once authentication works:

1. Explore API documentation at http://localhost:8000/docs
2. Test different user roles (Patient, Doctor, Admin)
3. Configure AI services (Groq, Replicate, etc.)
4. Explore other features (Voice AI, AR Scanner, etc.)

---

## 📚 Full Documentation

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## 🆘 Need Help?

1. Check logs:
   - Backend: `logs/healthsync.log`
   - Frontend: Browser console
2. Verify `.env` configuration
3. Check API docs at `/docs`
4. Ensure MongoDB is accessible

---

Happy coding! 🎉
