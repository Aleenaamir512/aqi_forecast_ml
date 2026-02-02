# prediction_model.py
from pathlib import Path
import pandas as pd
import joblib
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from pymongo import MongoClient
import os
import sys

# Ensure UTF-8 printing
sys.stdout.reconfigure(encoding='utf-8')

# =========================
# 0. BASE DIRECTORY
# =========================
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# =========================
# 1. CONNECT TO MONGODB USING ENV SECRET
# =========================
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    print("❌ MONGO_URI environment variable not set!")
    sys.exit(1)

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
# 3. MODEL FEATURES (must match trained model)
# =========================
model_features = [
    'year', 'month', 'day', 'dayofweek', 'is_weekend',
    'pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3',
    'temperature', 'humidity', 'precipitation',
    'pm2_5_roll7'  # match model column exactly
]

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
# 5. PREPARE ROW FOR PREDICTION
# =========================
current_row = df.iloc[[-1]].copy()

# Rename column if it came differently from MongoDB
if "pm2_5_rolling_7" in current_row.columns:
    current_row.rename(columns={"pm2_5_rolling_7": "pm2_5_roll7"}, inplace=True)

# Ensure only model features are selected
current_row = current_row[model_features]

# =========================
# 6. 3-DAY FORECAST
# =========================
predictions = []
forecast_dates = []

last_date = pd.Timestamp(year=int(df.iloc[-1]['year']),
                         month=int(df.iloc[-1]['month']),
                         day=int(df.iloc[-1]['day']))

for i in range(1, 4):
    # Predict
    pred = model.predict(current_row)[0]
    predictions.append(pred)

    # Next date
    next_date = last_date + timedelta(days=i)
    forecast_dates.append(next_date.strftime("%d %b %Y"))

    # Update features for next day
    current_row['year'] = next_date.year
    current_row['month'] = next_date.month
    current_row['day'] = next_date.day
    current_row['dayofweek'] = next_date.dayofweek
    current_row['is_weekend'] = int(next_date.dayofweek >= 5)
    # Keep rolling feature same (or update if you have logic)
    current_row['pm2_5_roll7'] = current_row['pm2_5_roll7']

# =========================
# 7. CREATE FORECAST DATAFRAME
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
# 8. PLOT FORECAST
# =========================
try:
    plt.figure(figsize=(6,4))  # smaller figure
    plt.plot(forecast["Date"], forecast["Predicted AQI"], marker='o', linestyle='-', color='blue')
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("3-Day AQI Forecast (Karachi)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "aqi_3day_forecast.png")
    plt.close()
    print("✅ AQI forecast plot saved")
except Exception as e:
    print(f"❌ Failed to plot forecast: {e}")

# =========================
# 9. UPLOAD FORECAST TO MONGODB
# =========================
try:
    forecast_dict = forecast.to_dict("records")
    collection_forecast.delete_many({})  # remove old forecast
    collection_forecast.insert_many(forecast_dict)
    print("✅ 3-day forecast uploaded to MongoDB")
except Exception as e:
    print(f"❌ Failed to upload forecast: {e}")
