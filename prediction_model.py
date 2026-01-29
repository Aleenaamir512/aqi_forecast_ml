import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# =========================
# 1. LOAD FEATURE-ENGINEERED DATA
# =========================
feature_file = Path(r"D:\aqi\feature_engineered_historical_with_realtime.pkl")
df = pd.read_pickle(feature_file)

# =========================
# 2. SELECT MODEL FEATURES
# =========================
model_features = [
    'year', 'month', 'day', 'dayofweek', 'is_weekend',
    'pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3',
    'temperature', 'humidity', 'precipitation', 'pm2_5_rolling_7'
]

# =========================
# 3. LOAD TRAINED MODEL
# =========================
model_file = Path(r"D:\aqi\XGBoost_karachi_aqi_model.pkl")
model = joblib.load(model_file)
print(f"✓ Model loaded from {model_file.name}")

# =========================
# 4. PREDICT 3-DAY AQI ITERATIVELY
# =========================
predictions = []
last_row = df.iloc[-1].copy()

for day_ahead in range(1, 4):  # Predict day 1, 2, 3
    # Create new date
    new_date = last_row['date'] + timedelta(days=1)
    
    # Prepare new feature row
    X_new = pd.DataFrame([{
        'year': new_date.year,
        'month': new_date.month,
        'day': new_date.day,
        'dayofweek': new_date.weekday(),
        'is_weekend': new_date.weekday() >= 5,
        'pm2.5': last_row['pm2.5'],          # Use last available PM2.5
        'pm10': last_row['pm10'],
        'no2': last_row['no2'],
        'so2': last_row['so2'],
        'co': last_row['co'],
        'o3': last_row['o3'],
        'temperature': last_row['temperature'],
        'humidity': last_row['humidity'],
        'precipitation': last_row['precipitation'],
        'pm2_5_rolling_7': df['pm2.5'].iloc[-7:].mean()  # rolling 7-day mean
    }])

    # Predict next-day AQI
    next_aqi = model.predict(X_new[model_features])[0]
    predictions.append((new_date.date(), next_aqi))

    # Update last_row PM2.5 with prediction for next iteration
    last_row['pm2.5'] = next_aqi
    df = pd.concat([df, X_new], ignore_index=True)

# =========================
# 5. PRINT 3-DAY PREDICTIONS
# =========================
print("✅ 3-Day AQI Predictions:")
for date, aqi in predictions:
    print(f"{date}: {aqi:.2f}")
