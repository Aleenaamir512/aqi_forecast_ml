# prediction_model.py
from pathlib import Path
import pandas as pd
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt
from pymongo import MongoClient
import sys, os

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# =========================
# 0. BASE DIRECTORY
# =========================
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# =========================
# 1. MONGODB CONNECTION
# =========================
import os
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)

db = client["aqi_database"]
collection_features = db["feature_store"]
collection_forecast = db["aqi_forecast"]

# =========================
# 2. LOAD FEATURE DATA
# =========================
df = pd.DataFrame(list(collection_features.find()))
if df.empty:
    print("❌ No feature data found in MongoDB!")
    sys.exit(1)

df.drop(columns=["_id"], inplace=True, errors="ignore")
print(f"✅ Feature data loaded: {df.shape}")

# =========================
# 3. MODEL FEATURES (must match training)
# =========================
model_features = [
    'year', 'month', 'day', 'dayofweek', 'is_weekend',
    'pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3',
    'temperature', 'humidity', 'precipitation',
    'pm2_5_rolling_7'
]

# Keep only these columns
missing_cols = [col for col in model_features if col not in df.columns]
if missing_cols:
    print(f"❌ Missing columns in data for model: {missing_cols}")
    sys.exit(1)

current_row = df[model_features].iloc[[-1]].copy()

# =========================
# 4. LOAD MODEL
# =========================
model_path = BASE_DIR / "RandomForest_karachi_aqi_model.pkl"
if not model_path.exists():
    print(f"❌ Model file not found at {model_path}")
    sys.exit(1)

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

print("✅ Model loaded")

# =========================
# 5. 3-DAY RECURSIVE PREDICTION
# =========================
predictions = []
forecast_dates = []

last_date = pd.Timestamp(
    year=int(df.iloc[-1]['year']),
    month=int(df.iloc[-1]['month']),
    day=int(df.iloc[-1]['day'])
)

for i in range(1, 4):
    # Predict AQI
    pred = model.predict(current_row)[0]
    predictions.append(round(float(pred), 2))

    # Calculate next day's date
    next_date = last_date + timedelta(days=i)
    forecast_dates.append(next_date.strftime("%d %b %Y"))

    # Update date-related features for next prediction
    current_row['year'] = next_date.year
    current_row['month'] = next_date.month
    current_row['day'] = next_date.day
    current_row['dayofweek'] = next_date.dayofweek
    current_row['is_weekend'] = int(next_date.dayofweek >= 5)

# =========================
# 6. CREATE FORECAST DATAFRAME
# =========================
forecast = pd.DataFrame({
    "Date": forecast_dates,
    "Predicted AQI": predictions
})

def aqi_label(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

forecast["Category"] = forecast["Predicted AQI"].apply(aqi_label)

print("\n✅ 3-Day AQI Forecast:")
print(forecast)

# =========================
# 7. PLOT TREND
# =========================
try:
    plt.figure(figsize=(6, 4))  # smaller chart
    plt.plot(forecast["Date"], forecast["Predicted AQI"], marker='o', linestyle='-', color='blue')
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("3-Day AQI Forecast (Karachi)")
    plt.grid(True)

    for i, val in enumerate(forecast["Predicted AQI"]):
        plt.text(i, val + 1, f"{val:.1f}", ha="center")

    plt.savefig(BASE_DIR / "aqi_3day_forecast.png")
    plt.close()
    print("✅ AQI forecast plot saved")
except Exception as e:
    print(f"❌ Failed to plot forecast: {e}")

# =========================
# 8. UPLOAD FORECAST TO MONGODB
# =========================
try:
    forecast_dict = forecast.to_dict("records")
    collection_forecast.delete_many({})  # clear old forecast
    collection_forecast.insert_many(forecast_dict)
    print("✅ 3-day forecast uploaded to MongoDB")
except Exception as e:
    print(f"❌ Failed to upload forecast: {e}")
