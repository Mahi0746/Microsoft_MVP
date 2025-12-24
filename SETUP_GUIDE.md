# HealthSync AI - Setup Guide

## Overview

This guide will help you set up the HealthSync AI project with MongoDB-only authentication (Supabase has been removed).

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **MongoDB** (Local or MongoDB Atlas)
- **Redis** (Optional, for caching)

## Backend Setup

### 1. Navigate to Backend Directory

```bash
cd backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Copy the example environment file:

```bash
copy .env.example .env    # Windows
cp .env.example .env      # macOS/Linux
```

Edit `.env` and update the following key values:

```env
# REQUIRED: Change this secret key!
SECRET_KEY=your-super-secret-key-at-least-32-characters-long

# MongoDB Connection (Local)
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=healthsync

# Or use MongoDB Atlas (Recommended for production)
# MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# AI API Keys (Get from respective services)
GROQ_API_KEY=your-groq-api-key
REPLICATE_API_TOKEN=your-replicate-token
HUGGINGFACE_API_KEY=your-huggingface-key
```

### 6. Start Backend Server

```bash
python main.py
# or
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

---

## Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd frontend/web
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

Copy the example environment file:

```bash
copy .env.example .env.local    # Windows
cp .env.example .env.local      # macOS/Linux
```

Edit `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Start Frontend Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

---

## MongoDB Setup

### Option 1: Local MongoDB

1. Install MongoDB Community Edition from [mongodb.com/download-center/community](https://www.mongodb.com/download-center/community)
2. Start MongoDB service
3. Use connection string: `mongodb://localhost:27017`

### Option 2: MongoDB Atlas (Cloud - Recommended)

1. Create free account at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster (Free tier available)
3. Create a database user with password
4. Whitelist your IP address (or use `0.0.0.0/0` for development)
5. Get connection string from Atlas dashboard
6. Update `.env` with the connection string

---

## Authentication Flow

### Simplified Authentication (No Real Auth Required)

The authentication has been simplified:

1. **Signup**: Users can sign up with any email/password and select their role (Patient, Doctor, Admin)
2. **Login**: Simple email/password login that generates JWT tokens
3. **Authorization**: Role-based access control is maintained - users keep their selected roles

### Available Roles

- **Patient**: Regular users seeking healthcare
- **Doctor**: Healthcare providers  
- **Admin**: System administrators

### Testing Authentication

#### 1. Create a Test User

**Via API (using curl or Postman):**

```bash
POST http://localhost:8000/api/auth/signup
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "Test1234",
  "first_name": "Test",
  "last_name": "User",
  "role": "patient"
}
```

#### 2. Login

```bash
POST http://localhost:8000/api/auth/login
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "Test1234"
}
```

You'll receive:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### 3. Use Token in Requests

Add the token to Authorization header:
```
Authorization: Bearer eyJ...
```

---

## Troubleshooting

### CORS Errors

If you get CORS errors, ensure:
1. Backend is running on `http://localhost:8000`
2. Frontend is running on `http://localhost:3000`
3. The `ALLOWED_ORIGINS` in backend `.env` includes frontend URL

### MongoDB Connection Issues

1. **Local MongoDB**: Ensure MongoDB service is running
2. **MongoDB Atlas**: 
   - Check IP whitelist
   - Verify username/password
   - Ensure connection string format is correct

### Authentication Errors

1. Check `.env` has correct `SECRET_KEY` (min 32 characters)
2. Ensure MongoDB is running and accessible
3. Check backend logs for detailed error messages

### Port Already in Use

If port 8000 or 3000 is already in use:

**Backend:**
```bash
# Change port in .env
PORT=8001

# Or specify when running
uvicorn main:app --port 8001
```

**Frontend:**
```bash
npm run dev -- -p 3001
```

---

## API Endpoints

### Authentication

- `POST /api/auth/signup` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/refresh` - Refresh access token
- `GET /api/auth/me` - Get current user info
- `PUT /api/auth/profile` - Update user profile
- `POST /api/auth/change-password` - Change password
- `POST /api/auth/logout` - Logout user

### Health Check

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed service status

---

## Next Steps

1. ✅ Backend running on port 8000
2. ✅ Frontend running on port 3000
3. ✅ MongoDB connected
4. ✅ Test signup and login

Now you can:
- Access the web dashboard at `http://localhost:3000`
- Create new accounts with different roles
- Test the authentication flow
- Explore the API documentation at `http://localhost:8000/docs`

---

## Support

For issues or questions:
1. Check backend logs: `logs/healthsync.log`
2. Check browser console for frontend errors
3. Review API documentation at `/docs`
4. Ensure all environment variables are set correctly
