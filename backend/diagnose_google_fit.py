#!/usr/bin/env python3
"""
Google Fit Data Source Checker
Checks what data sources are available in your Google Fit account
"""

import asyncio
import pickle
import os
import requests
from datetime import datetime, timedelta

async def check_data_sources():
    print("\n" + "="*70)
    print("🔍 GOOGLE FIT DATA SOURCE DIAGNOSTIC")
    print("="*70)
    
    # Load credentials
    if not os.path.exists('token.pickle'):
        print("\n❌ No token.pickle found. Run test_google_fit.py first.")
        return
    
    with open('token.pickle', 'rb') as f:
        creds = pickle.load(f)
    
    access_token = creds.token
    headers = {'Authorization': f'Bearer {access_token}'}
    
    print("\n📊 Checking available data sources in your Google Fit account...")
    
    # 1. List all data sources
    print("\n" + "-"*70)
    print("1️⃣ LISTING ALL DATA SOURCES")
    print("-"*70)
    
    url = 'https://www.googleapis.com/fitness/v1/users/me/dataSources'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data_sources = response.json().get('dataSource', [])
        print(f"\n   Found {len(data_sources)} data sources:")
        
        for i, source in enumerate(data_sources, 1):
            data_type = source.get('dataType', {}).get('name', 'Unknown')
            app = source.get('application', {}).get('name', 'Unknown')
            print(f"   {i}. {data_type} (from: {app})")
    else:
        print(f"\n   ❌ Error: {response.status_code} - {response.text}")
    
    # 2. Check step count specifically
    print("\n" + "-"*70)
    print("2️⃣ CHECKING STEP COUNT DATA (All Available Sources)")
    print("-"*70)
    
    # Try multiple step count data sources
    step_sources = [
        "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps",
        "derived:com.google.step_count.delta:com.google.android.gms:merge_step_deltas",
        "raw:com.google.step_count.delta:com.google.android.gms:*",
    ]
    
    end_ns = int(datetime.now().timestamp() * 1e9)
    start_ns = end_ns - 7 * 24 * 60 * 60 * int(1e9)
    
    for source in step_sources:
        print(f"\n   Testing: {source}")
        url = f'https://www.googleapis.com/fitness/v1/users/me/dataSources/{source}/datasets/{start_ns}-{end_ns}'
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                points = response.json().get('point', [])
                total = sum(p['value'][0]['intVal'] for p in points)
                print(f"      ✅ Found {len(points)} data points, Total: {total} steps")
            else:
                print(f"      ❌ Error {response.status_code}")
        except Exception as e:
            print(f"      ❌ Exception: {e}")
    
    # 3. Try aggregate API
    print("\n" + "-"*70)
    print("3️⃣ TRYING AGGREGATE API (Recommended Method)")
    print("-"*70)
    
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = end_ms - 7 * 24 * 60 * 60 * 1000
    
    body = {
        "aggregateBy": [{
            "dataTypeName": "com.google.step_count.delta"
        }],
        "bucketByTime": {"durationMillis": 86400000},  # 1 day buckets
        "startTimeMillis": start_ms,
        "endTimeMillis": end_ms
    }
    
    url = 'https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate'
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        buckets = response.json().get('bucket', [])
        print(f"\n   Found {len(buckets)} daily buckets:")
        
        for bucket in buckets:
            start_time = datetime.fromtimestamp(int(bucket['startTimeMillis']) / 1000)
            dataset = bucket.get('dataset', [])
            
            steps = 0
            for ds in dataset:
                for point in ds.get('point', []):
                    steps += point['value'][0]['intVal']
            
            print(f"      {start_time.strftime('%Y-%m-%d')}: {steps} steps")
    else:
        print(f"\n   ❌ Error: {response.status_code}")
        print(f"   {response.text}")
    
    # 4. Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    print("""
📱 Your Google Fit account appears EMPTY. Here's why and how to fix:

WHY NO DATA?
1. ❌ No fitness tracker connected (no phone step counter, no smartwatch)
2. ❌ Google Fit app not actively tracking
3. ❌ Permissions not granted to Google Fit

HOW TO GET DATA:
1. ✅ Install "Google Fit" app on Android phone
2. ✅ Enable "Track your activities" in app settings
3. ✅ Grant location & activity permissions
4. ✅ Walk around for a few hours with phone in pocket
5. ✅ OR manually log data in the app
6. ✅ OR connect a wearable (Fitbit, Wear OS watch, etc.)

ALTERNATIVE FOR TESTING:
- Use Fitbit API instead (if you have Fitbit device)
- Use Samsung Health API (if Samsung phone)
- Manually create test data in Google Fit app

Once you have data logged, re-run the test and you'll see real numbers!
    """)

if __name__ == "__main__":
    asyncio.run(check_data_sources())
