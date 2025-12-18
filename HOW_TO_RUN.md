# 🚀 How to Run HealthSync AI

## Quick Start Guide

### Step 1: Prerequisites Check

Make sure you have installed:
- ✅ **Python 3.11+** - [Download](https://www.python.org/downloads/)
- ✅ **Node.js 18+** - [Download](https://nodejs.org/)
- ✅ **npm** (comes with Node.js)

**Check versions:**
```bash
python --version    # Should be 3.11 or higher
node --version      # Should be 18 or higher
npm --version       # Should be 9 or higher
```

---

### Step 2: Setup Database (Required)

**Minimum: MongoDB Atlas** (5 minutes)

1. **Create MongoDB Atlas account**: https://mongodb.com/atlas
2. **Follow the guide**: Read `DATABASE_SETUP_GUIDE.md` → MongoDB Atlas section
3. **Get connection string** from MongoDB Atlas dashboard
4. **Save it** - you'll need it in the next step

**Optional: Supabase** (for full features)
- Follow `DATABASE_SETUP_GUIDE.md` → Supabase section

---

### Step 3: Configure Environment

1. **Create `.env` file** in the root directory:
   ```bash
   # Windows (PowerShell)
   Copy-Item .env.example .env
   
   # Or manually create .env file
   ```

2. **Open `.env` file** and add your MongoDB connection string:
   ```env
   # Minimum required - MongoDB Atlas
   MONGODB_URL=mongodb+srv://username:password@cluster.xxxxx.mongodb.net/healthsync?retryWrites=true&w=majority
   MONGODB_DATABASE=healthsync
   
   # Optional - Generate a secret key (any random string, 32+ characters)
   SECRET_KEY=your_secret_key_here_minimum_32_characters_long
   
   # Optional - Supabase (if you set it up)
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_ANON_KEY=your_key_here
   SUPABASE_SERVICE_ROLE_KEY=your_key_here
   DATABASE_URL=postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres
   ```

---

### Step 4: Install Dependencies

#### Backend Dependencies

```bash
# Navigate to backend folder
cd backend

# Install Python packages
pip install -r requirements.txt

# If you get errors, try:
pip install --upgrade pip
pip install -r requirements.txt
```

#### Frontend Dependencies

```bash
# Navigate to frontend/web folder
cd frontend/web

# Install Node.js packages
npm install

# If you get errors, try:
npm install --legacy-peer-deps
```

---

### Step 5: Run the Application

You need **2 terminal windows** (one for backend, one for frontend).

#### Option A: Using Batch Files (Windows - Easiest)

**Terminal 1 - Backend:**
```bash
# Double-click START_BACKEND.bat
# OR run from command line:
START_BACKEND.bat
```

**Terminal 2 - Frontend:**
```bash
# Double-click START_FRONTEND.bat
# OR run from command line:
START_FRONTEND.bat
```

#### Option B: Manual Commands

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend/web
npm run dev
```

---

### Step 6: Access the Application

Once both servers are running:

- 🌐 **Web Dashboard**: http://localhost:3000
- 🔧 **API Documentation**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

---

## ✅ Verification

### Check Backend is Running:
1. Open browser: http://localhost:8000/health
2. Should see: `{"status":"healthy",...}`

### Check Frontend is Running:
1. Open browser: http://localhost:3000
2. Should see: Dark-themed HealthSync AI landing page

### Check Database Connection:
1. Visit: http://localhost:8000/health/detailed
2. Look for: `"mongodb": "healthy"` in the response

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error: "Module not found"**
```bash
cd backend
pip install -r requirements.txt
```

**Error: "Port 8000 already in use"**
```bash
# Find and kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Or change port in .env:
PORT=8001
```

**Error: "Database connection failed"**
- ✅ Check `.env` file has correct `MONGODB_URL`
- ✅ Verify MongoDB Atlas IP whitelist includes `0.0.0.0/0`
- ✅ Check username/password are correct
- ✅ Wait 2-3 minutes after creating MongoDB cluster

### Frontend Won't Start

**Error: "Port 3000 already in use"**
```bash
# Find and kill the process
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID_NUMBER> /F

# Or change port:
# Edit frontend/web/package.json scripts
```

**Error: "Module not found"**
```bash
cd frontend/web
rm -rf node_modules package-lock.json
npm install
```

**Error: "Cannot find module"**
```bash
cd frontend/web
npm install --legacy-peer-deps
```

### Database Connection Issues

**MongoDB Atlas:**
- ✅ Connection string format: `mongodb+srv://username:password@cluster...`
- ✅ Replace `<password>` with actual password (URL encode special characters)
- ✅ Add database name: `...mongodb.net/healthsync?...`
- ✅ Check IP whitelist in MongoDB Atlas dashboard

**Check connection:**
```bash
cd backend
python -c "from services.mongodb_atlas_service import mongodb_service; import asyncio; print(asyncio.run(mongodb_service.connect()))"
```

---

## 📋 Quick Command Reference

### Start Everything (Windows)
```bash
# Terminal 1
START_BACKEND.bat

# Terminal 2
START_FRONTEND.bat
```

### Start Everything (Manual)
```bash
# Terminal 1 - Backend
cd backend
python -m uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend/web
npm run dev
```

### Stop Servers
- Press `Ctrl + C` in each terminal window

### Check if Running
```bash
# Backend
curl http://localhost:8000/health

# Frontend
# Open browser: http://localhost:3000
```

---

## 🎯 What You Should See

### Backend Terminal:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
INFO:     Database connections initialized
```

### Frontend Terminal:
```
▲ Next.js 14.0.4
- Local:        http://localhost:3000
- ready started server on 0.0.0.0:3000
```

### Browser (http://localhost:3000):
- Dark-themed landing page
- "HealthSync AI" logo
- "Get Started" button
- Feature cards with gradients

---

## 🚀 Next Steps After Running

1. **Create Account**: Click "Get Started" → Register
2. **Explore Dashboard**: Login → See dark-themed dashboard
3. **Test Features**: Try Voice AI Doctor, AR Scanner, etc.
4. **Check API Docs**: Visit http://localhost:8000/docs

---

## 💡 Pro Tips

1. **Keep both terminals open** - Backend and Frontend need to run simultaneously
2. **Check `.env` file** - Make sure MongoDB URL is correct
3. **Watch terminal output** - Errors will show there
4. **Use browser console** - Press F12 to see frontend errors
5. **Check logs** - Backend logs in `backend/logs/healthsync.log`

---

## 📚 Need More Help?

- **Database Setup**: Read `DATABASE_SETUP_GUIDE.md`
- **Full Documentation**: Read `README.md`
- **Setup Summary**: Read `SETUP_COMPLETE.md`
- **API Documentation**: http://localhost:8000/docs (when backend is running)

---

**That's it! Your HealthSync AI platform should now be running! 🎉**

