# 🗄️ Database Setup Guide - HealthSync AI

This guide will help you set up **Supabase** (PostgreSQL) and **MongoDB Atlas** for HealthSync AI.

## 📋 Overview

HealthSync AI uses a **hybrid database architecture**:
- **Supabase (PostgreSQL)**: Relational data, user authentication, appointments
- **MongoDB Atlas**: Document storage, AI analysis results, family graphs, complex data
- **Redis**: Caching and session management (optional but recommended)

---

## 🚀 Quick Start (Choose One Database)

### Option 1: MongoDB Atlas Only (Easiest - Recommended for MVP)
If you want to start quickly, you can use **MongoDB Atlas only**:

1. **Setup MongoDB Atlas** (see section below)
2. **Skip Supabase** for now
3. Use MongoDB for all data storage

### Option 2: Full Setup (Supabase + MongoDB Atlas)
For production-ready setup with authentication and relational data.

---

## 🍃 MongoDB Atlas Setup (FREE - 5 minutes)

### Step 1: Create Account
1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Click **"Try Free"** or **"Sign Up"**
3. Sign up with Google/GitHub or email

### Step 2: Create FREE Cluster
1. Click **"Build a Database"**
2. Select **"M0 Sandbox"** (FREE tier - 512MB storage)
3. Choose **AWS** as cloud provider
4. Select region closest to you (e.g., `us-east-1`)
5. Cluster name: `healthsync-cluster`
6. Click **"Create Cluster"** (takes 2-3 minutes)

### Step 3: Create Database User
1. Go to **"Database Access"** in left menu
2. Click **"Add New Database User"**
3. Authentication: **Password**
4. Username: `healthsync_user`
5. Password: Click **"Autogenerate Secure Password"** (SAVE THIS!)
6. Database User Privileges: **"Read and write to any database"**
7. Click **"Add User"**

### Step 4: Configure Network Access
1. Go to **"Network Access"** in left menu
2. Click **"Add IP Address"**
3. Click **"Allow Access from Anywhere"** (for development)
   - This adds `0.0.0.0/0` to whitelist
4. Click **"Confirm"**

### Step 5: Get Connection String
1. Go to **"Database"** in left menu
2. Click **"Connect"** on your cluster
3. Choose **"Connect your application"**
4. Driver: **Python**, Version: **3.12 or later**
5. Copy the connection string:
   ```
   mongodb+srv://healthsync_user:<password>@healthsync-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
6. **Replace `<password>`** with your actual password
7. **Add database name** at the end:
   ```
   mongodb+srv://healthsync_user:your_password@healthsync-cluster.xxxxx.mongodb.net/healthsync?retryWrites=true&w=majority
   ```

### Step 6: Add to .env File
Add this to your `.env` file:
```env
MONGODB_URL=mongodb+srv://healthsync_user:your_password@healthsync-cluster.xxxxx.mongodb.net/healthsync?retryWrites=true&w=majority
MONGODB_DATABASE=healthsync
```

### ✅ Test Connection
Run this command to test:
```bash
cd backend
python -c "from services.mongodb_atlas_service import mongodb_service; import asyncio; asyncio.run(mongodb_service.connect())"
```

---

## 🐘 Supabase Setup (PostgreSQL - Optional but Recommended)

### Step 1: Create Account
1. Go to [supabase.com](https://supabase.com)
2. Click **"Start your project"** or **"Sign Up"**
3. Sign up with GitHub or email

### Step 2: Create Project
1. Click **"New Project"**
2. Choose organization (or create one)
3. Project name: `healthsync-ai`
4. Database password: Generate secure password (SAVE THIS!)
5. Region: Choose closest to you
6. Click **"Create new project"** (takes 2-3 minutes)

### Step 3: Get API Keys
1. Go to **Settings** > **API** in left menu
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon/public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
   - **service_role key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (keep secret!)

### Step 4: Get Database Connection String
1. Go to **Settings** > **Database**
2. Scroll to **Connection string** section
3. Copy **URI** connection string:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
   ```
4. Replace `[YOUR-PASSWORD]` with your database password

### Step 5: Run Database Schema
1. Go to **SQL Editor** in left menu
2. Click **"New query"**
3. Copy contents from `database/postgresql/schema.sql`
4. Paste and click **"Run"**
5. Repeat for `database/postgresql/rls_policies.sql` (Row Level Security)

### Step 6: Add to .env File
Add these to your `.env` file:
```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
DATABASE_URL=postgresql://postgres:your_password@db.xxxxx.supabase.co:5432/postgres
```

---

## 🔴 Redis Setup (Optional - For Caching)

### Option 1: Local Redis (Development)
```bash
# Install Redis
# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Mac: brew install redis
# Linux: sudo apt-get install redis-server

# Start Redis
redis-server

# Add to .env
REDIS_URL=redis://localhost:6379/0
```

### Option 2: Upstash Redis (Free Cloud - Recommended)
1. Go to [upstash.com](https://upstash.com)
2. Sign up (free tier: 10K commands/day)
3. Click **"Create Database"**
4. Name: `healthsync-redis`
5. Region: Choose closest to you
6. Click **"Create"**
7. Copy **Redis URL**:
   ```
   redis://default:xxxxx@xxxxx.upstash.io:6379
   ```
8. Add to `.env`:
   ```env
   REDIS_URL=redis://default:xxxxx@xxxxx.upstash.io:6379
   ```

---

## ✅ Verification Checklist

After setup, verify everything works:

### 1. Check MongoDB Atlas
```bash
cd backend
python -c "from services.mongodb_atlas_service import mongodb_service; import asyncio; print(asyncio.run(mongodb_service.health_check()))"
```

### 2. Check Supabase (if using)
```bash
cd backend
python -c "from services.supabase_service import SupabaseService; import asyncio; print(asyncio.run(SupabaseService.health_check()))"
```

### 3. Check Backend Health
```bash
cd backend
python main.py
# Visit http://localhost:8000/health
```

### 4. Test Database Connection
```bash
cd backend
python -c "from services.db_service import DatabaseService; import asyncio; asyncio.run(DatabaseService.initialize())"
```

---

## 🐛 Troubleshooting

### MongoDB Atlas Connection Failed
- ✅ Check password is correct (no special characters need URL encoding)
- ✅ Check IP whitelist includes `0.0.0.0/0` for development
- ✅ Check connection string format is correct
- ✅ Wait 2-3 minutes after creating cluster

### Supabase Connection Failed
- ✅ Check database password is correct
- ✅ Check connection string format
- ✅ Verify project is fully created (wait 2-3 minutes)
- ✅ Check API keys are correct

### Redis Connection Failed
- ✅ Check Redis server is running (if local)
- ✅ Check connection URL format
- ✅ For Upstash, verify database is active

---

## 📚 Additional Resources

- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [Supabase Documentation](https://supabase.com/docs)
- [Upstash Redis Documentation](https://docs.upstash.com/redis)

---

## 💡 Pro Tips

1. **Start with MongoDB Atlas only** for MVP - it's easier to set up
2. **Use Upstash Redis** for free cloud caching
3. **Save all passwords** in a secure password manager
4. **Use environment variables** - never commit `.env` to git
5. **Test connections** before deploying to production

---

## 🎯 Next Steps

After setting up databases:
1. ✅ Copy `.env.example` to `.env` and fill in values
2. ✅ Test database connections
3. ✅ Run backend: `cd backend && python main.py`
4. ✅ Run frontend: `cd frontend/web && npm run dev`
5. ✅ Visit `http://localhost:3000` to see your app!

---

**Need Help?** Check the logs in `backend/logs/healthsync.log` for detailed error messages.

