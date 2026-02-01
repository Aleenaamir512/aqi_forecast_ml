# prediction_model_mongo.py
from pathlib import Path
import pandas as pd
import joblib
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from pymongo import MongoClient
import sys, os

# Fix stdout encoding to handle unicode symbols
sys.stdout.reconfigure(encoding='utf-8')

# =========================
# 0. BASE DIRECTORY & WORKING DIR
# =========================
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# =========================
# 1. CONNECT TO MONGODB
# =========================
MONGO_URI = "mongodb+srv://aleenaamir02_db_user:zY4PRfUXbOplm3Ae@cluster0.km7h66h.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["aqi_database"]
collection_features = db["feature_store"]
collection_forecast = db["aqi_forecast"]

# =========================
# 2. LOAD FEATURE DATA FROM MONGO
# =========================
df = pd.DataFrame(list(collection_features.find()))
if df.empty:
    print("❌ No feature data found in MongoDB!")
    sys.exit(1)

df.drop(columns=["_id"], inplace=True, errors="ignore")
print("✓ Feature data loaded from MongoDB:", df.shape)

# =========================
# 3. MODEL FEATURES
# =========================
model_features = [
    'year', 'month', 'day', 'dayofweek', 'is_weekend',
    'pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3',
    'temperature', 'humidity', 'precipitation',
    'pm2_5_rolling_7'
]

current_row = df.iloc[[-1]].copy()

# =========================
# 4. LOAD MODEL
# =========================
model_path = BASE_DIR / "XGBoost_karachi_aqi_model.pkl"
if not model_path.exists():
    print(f"❌ Model file not found at {model_path}")
    sys.exit(1)

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

print("✓ Model loaded")

from datetime import datetime

# =========================
# 5. 3-DAY RECURSIVE PREDICTION
# =========================
predictions = []

# Start forecasting from tomorrow
last_date = datetime.today()  # <-- change this line

for day in range(1, 4):
    try:
        X = current_row[model_features]
        aqi_pred = model.predict(X)[0]
        predictions.append(aqi_pred)
    except Exception as e:
        print(f"❌ Prediction failed on day {day}: {e}")
        sys.exit(1)

    # Update row for next day
    current_date = last_date + timedelta(days=day)
    current_row['year'] = current_date.year
    current_row['month'] = current_date.month
    current_row['day'] = current_date.day
    current_row['dayofweek'] = current_date.weekday()
    current_row['is_weekend'] = int(current_date.weekday() >= 5)
    current_row['pm2.5'] = current_row['pm2.5']
    current_row['pm2_5_rolling_7'] = current_row['pm2_5_rolling_7']

# =========================
# 6. CREATE FORECAST DATAFRAME
# =========================
forecast_dates = [(datetime.today() + timedelta(days=i)).strftime("%d %b %Y") for i in range(1, 4)]
forecast = pd.DataFrame({
    "Date": forecast_dates,  # <-- use the next 3 days from today
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

print("\n3-Day AQI Forecast with Categories:")
print(forecast)

# =========================
# 7. PLOT 3-DAY TREND
# =========================
try:
    plt.figure()
    plt.plot(forecast["Date"], predictions, marker='o', linestyle='-', color='blue')
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("3-Day AQI Forecast (Karachi)")
    plt.grid(True)

    # Add labels above points
    for i, val in enumerate(predictions):
        plt.text(i, val + 1, f"{val:.1f}", ha="center")

    plt.savefig(BASE_DIR / "aqi_3day_forecast.png")
    plt.close()
    print("✓ AQI forecast plot saved")
except Exception as e:
    print(f"❌ Failed to plot forecast: {e}")

# =========================
# 8. UPLOAD FORECAST TO MONGODB
# =========================
try:
    forecast_dict = forecast.to_dict("records")
    collection_forecast.delete_many({})  # optional: remove old forecast
    collection_forecast.insert_many(forecast_dict)
    print("✓ 3-day forecast uploaded to MongoDB")
except Exception as e:
    print(f"❌ Failed to upload forecast: {e}")
