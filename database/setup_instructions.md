# HealthSync AI - Database Setup Instructions

## Overview

HealthSync AI uses a hybrid database architecture:
- **PostgreSQL (Supabase)**: Relational data, authentication, real-time subscriptions
- **MongoDB Atlas**: Complex documents, family graphs, ML models, AI analysis results
- **Redis (Upstash)**: Caching, session storage, rate limiting

## Prerequisites

- Supabase account (free tier: 500MB DB, 1GB storage)
- MongoDB Atlas account (free tier: 512MB storage)
- Upstash Redis account (free tier: 10K commands/day)

## 1. Supabase PostgreSQL Setup

### Step 1: Create Supabase Project
```bash
# Visit https://supabase.com/dashboard
# Click "New Project"
# Choose organization and region
# Set database password (save this!)
# Wait for project creation (2-3 minutes)
```

### Step 2: Get Connection Details
```bash
# From Supabase Dashboard > Settings > Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
```

### Step 3: Run Database Schema
```bash
# Option 1: Using Supabase SQL Editor
# 1. Go to Supabase Dashboard > SQL Editor
# 2. Copy contents of database/postgresql/schema.sql
# 3. Run the script

# Option 2: Using psql command line
psql "postgresql://postgres:password@db.your-project.supabase.co:5432/postgres" \
  -f database/postgresql/schema.sql
```

### Step 4: Apply RLS Policies
```bash
# In Supabase SQL Editor or psql
psql "postgresql://postgres:password@db.your-project.supabase.co:5432/postgres" \
  -f database/postgresql/rls_policies.sql
```

### Step 5: Insert Sample Data
```bash
# For development/testing only
psql "postgresql://postgres:password@db.your-project.supabase.co:5432/postgres" \
  -f database/postgresql/seed_data.sql
```

### Step 6: Enable Realtime (Optional)
```sql
-- In Supabase SQL Editor
-- Enable realtime for specific tables
ALTER PUBLICATION supabase_realtime ADD TABLE appointments;
ALTER PUBLICATION supabase_realtime ADD TABLE appointment_bids;
ALTER PUBLICATION supabase_realtime ADD TABLE notifications;
ALTER PUBLICATION supabase_realtime ADD TABLE therapy_sessions;
```

## 2. MongoDB Atlas Setup

### Step 1: Create MongoDB Cluster
```bash
# Visit https://cloud.mongodb.com/
# Click "Build a Database"
# Choose "FREE" shared cluster
# Select cloud provider and region (same as Supabase for latency)
# Choose cluster name: "healthsync-cluster"
# Click "Create Cluster" (takes 3-5 minutes)
```

### Step 2: Configure Network Access
```bash
# In MongoDB Atlas Dashboard
# Go to Network Access > IP Access List
# Click "Add IP Address"
# Choose "Allow Access from Anywhere" (0.0.0.0/0) for development
# For production, add specific IP addresses
```

### Step 3: Create Database User
```bash
# Go to Database Access > Database Users
# Click "Add New Database User"
# Choose "Password" authentication
# Username: healthsync_user
# Password: generate secure password (save this!)
# Database User Privileges: "Read and write to any database"
```

### Step 4: Get Connection String
```bash
# Go to Databases > Connect > Connect your application
# Choose "Node.js" and version "4.1 or later"
# Copy connection string:
MONGODB_URL=mongodb+srv://healthsync_user:password@healthsync-cluster.mongodb.net/healthsync?retryWrites=true&w=majority
```

### Step 5: Create Collections and Indexes
```bash
# Using MongoDB Compass (GUI) or mongosh (CLI)
mongosh "mongodb+srv://healthsync_user:password@healthsync-cluster.mongodb.net/healthsync"

# Run the collections script
load('database/mongodb/collections.js')
```

### Step 6: Insert Sample Data
```bash
# For development/testing only
mongosh "mongodb+srv://healthsync_user:password@healthsync-cluster.mongodb.net/healthsync" \
  --file database/mongodb/seed_data.js
```

## 3. Redis Cache Setup (Upstash)

### Step 1: Create Upstash Redis Database
```bash
# Visit https://console.upstash.com/
# Click "Create Database"
# Choose region (same as other services)
# Database name: "healthsync-cache"
# Type: "Regional" (free tier)
# Click "Create"
```

### Step 2: Get Connection Details
```bash
# From Upstash Console > Database Details
REDIS_URL=redis://default:password@redis-host:6379
# Or use the REST API URL for serverless environments
UPSTASH_REDIS_REST_URL=https://your-db.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-token
```

## 4. Environment Variables Setup

### Backend (.env)
```bash
# Copy .env.example to .env
cp .env.example .env

# Update with your actual values
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/healthsync
MONGODB_DATABASE=healthsync

REDIS_URL=redis://default:password@redis-host:6379
```

### Frontend (.env.local)
```bash
# React Native (mobile-app/.env.local)
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
EXPO_PUBLIC_API_URL=http://localhost:8000

# Next.js Admin (admin-dashboard/.env.local)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 5. Verification & Testing

### Test PostgreSQL Connection
```python
# backend/test_db.py
import asyncpg
import asyncio

async def test_postgres():
    conn = await asyncpg.connect("postgresql://postgres:password@db.your-project.supabase.co:5432/postgres")
    result = await conn.fetch("SELECT COUNT(*) FROM users")
    print(f"Users count: {result[0]['count']}")
    await conn.close()

asyncio.run(test_postgres())
```

### Test MongoDB Connection
```python
# backend/test_mongo.py
from pymongo import MongoClient

client = MongoClient("mongodb+srv://user:pass@cluster.mongodb.net/healthsync")
db = client.healthsync
count = db.family_graph.count_documents({})
print(f"Family graphs count: {count}")
```

### Test Redis Connection
```python
# backend/test_redis.py
import redis

r = redis.from_url("redis://default:password@redis-host:6379")
r.set("test_key", "test_value")
value = r.get("test_key")
print(f"Redis test: {value.decode()}")
```

## 6. Database Migrations (Production)

### PostgreSQL Migrations with Alembic
```bash
# Install alembic
pip install alembic

# Initialize alembic (already done in project)
# alembic init alembic

# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

### MongoDB Schema Versioning
```javascript
// Add schema version to collections
db.schema_versions.insertOne({
  collection: "family_graph",
  version: "1.0.0",
  applied_at: new Date(),
  changes: ["Initial schema creation"]
});
```

## 7. Backup & Recovery

### PostgreSQL Backup (Supabase)
```bash
# Supabase provides automatic backups
# Manual backup using pg_dump
pg_dump "postgresql://postgres:password@db.your-project.supabase.co:5432/postgres" \
  > backup_$(date +%Y%m%d_%H%M%S).sql
```

### MongoDB Backup (Atlas)
```bash
# MongoDB Atlas provides automatic backups
# Manual backup using mongodump
mongodump --uri="mongodb+srv://user:pass@cluster.mongodb.net/healthsync" \
  --out=backup_$(date +%Y%m%d_%H%M%S)
```

## 8. Performance Optimization

### PostgreSQL Indexes
```sql
-- Monitor slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_health_metrics_user_date 
ON health_metrics(user_id, measured_at DESC);
```

### MongoDB Indexes
```javascript
// Monitor query performance
db.family_graph.explain("executionStats").find({user_id: "uuid"});

// Add compound indexes
db.health_events.createIndex(
  {"user_id": 1, "timestamp": -1, "event_type": 1}
);
```

### Redis Optimization
```bash
# Monitor Redis performance
redis-cli --latency-history -i 1

# Set memory policy
CONFIG SET maxmemory-policy allkeys-lru
```

## 9. Security Checklist

- [ ] Enable SSL/TLS for all database connections
- [ ] Use strong passwords (minimum 16 characters)
- [ ] Enable IP whitelisting in production
- [ ] Set up database user roles with minimal permissions
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Monitor for suspicious activity
- [ ] Backup encryption
- [ ] Connection string encryption in environment variables

## 10. Monitoring & Alerts

### Supabase Monitoring
- Database size and growth
- Connection pool usage
- Query performance
- RLS policy effectiveness

### MongoDB Atlas Monitoring
- Cluster performance metrics
- Index usage statistics
- Connection counts
- Storage utilization

### Redis Monitoring
- Memory usage
- Hit/miss ratios
- Connection counts
- Command latency

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check network access settings
   - Verify IP whitelist
   - Test connection string format

2. **Authentication Failed**
   - Verify username/password
   - Check user permissions
   - Ensure database exists

3. **SSL Certificate Issues**
   - Update connection string with SSL parameters
   - Check certificate validity
   - Use proper SSL mode

4. **Performance Issues**
   - Add missing indexes
   - Optimize query patterns
   - Check connection pooling
   - Monitor resource usage

### Support Resources

- Supabase: https://supabase.com/docs
- MongoDB Atlas: https://docs.atlas.mongodb.com/
- Upstash Redis: https://docs.upstash.com/redis
- Community forums and Discord channels