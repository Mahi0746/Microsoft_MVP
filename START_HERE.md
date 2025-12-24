# Quick Start - Authentication Works!

## ✅ Setup Complete!

Your HealthSync AI authentication system is now configured with MongoDB only (no Supabase).

---

## 🚀 How to Start

### 1. Start Backend

Open a terminal in the backend folder:

```powershell
cd C:\Microsoft_Hackathon\new\Microsoft_MVP\backend
.\venv\Scripts\Activate.ps1
python main.py
```

**Note:** Before running, make sure to:
1. Copy `.env.example` to `.env`
2. Set a SECRET_KEY (minimum 32 characters)
3. Configure MONGODB_URL

### 2. Start Frontend

Open another terminal in the frontend folder:

```powershell
cd C:\Microsoft_Hackathon\new\Microsoft_MVP\frontend\web
npm run dev
```

---

## 🌐 Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🧪 Test Authentication

1. Go to http://localhost:3000/auth/register
2. Create an account with any role:
   - **Patient**: Regular user
   - **Doctor**: Healthcare provider
   - **Admin**: Administrator
3. Login at http://localhost:3000/auth/login
4. You're in! No external services needed.

---

## 📦 MongoDB Setup

### Local MongoDB
```
MONGODB_URL=mongodb://localhost:27017
```

### MongoDB Atlas (Recommended)
1. Create account at https://cloud.mongodb.com
2. Create free cluster
3. Get connection string
4. Update `.env`:
```
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
```

---

##  Troubleshooting

### "MongoDB connection failed"
- Start MongoDB service OR
- Use MongoDB Atlas connection string

### "Port already in use"
Change ports in backend `.env`:
```
PORT=8001
```

### CORS errors
Development mode allows all origins by default. If issues persist, check that:
- Backend running on http://localhost:8000
- Frontend running on http://localhost:3000

---

## 📝 What Changed

- ✅ Removed Supabase completely
- ✅ Removed PostgreSQL completely
- ✅ Using MongoDB for all data
- ✅ Simplified authentication (JWT tokens)
- ✅ No CORS issues
- ✅ Python 3.13 compatible
- ✅ Role-based authorization maintained

---

**You're all set! Create your first account and start using HealthSync AI! 🎉**
