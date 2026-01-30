import pandas as pd
import joblib
from pathlib import Path

# =========================
# 1. LOAD FEATURE DATA
# =========================
#feature_file = Path("feature_engineered_historical_with_realtime.pkl")
#df = pd.read_pickle(feature_file)

# =========================
# 2. MODEL FEATURES
# =========================
model_features = [
    'year', 'month', 'day', 'dayofweek', 'is_weekend',
    'pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3',
    'temperature', 'humidity', 'precipitation',
    'pm2_5_rolling_7'
]

# Start from last available row
current_row = df.iloc[[-1]].copy()

# =========================
# 3. LOAD MODEL
# =========================
#model = joblib.load("XGBoost_karachi_aqi_model.pkl")
#print("✓ Model loaded")
BASE_DIR = Path(__file__).resolve().parent

feature_file = BASE_DIR / "feature_engineered_historical_with_realtime.pkl"
model = joblib.load(BASE_DIR / "XGBoost_karachi_aqi_model.pkl")

# =========================
# 4. 3-DAY RECURSIVE PREDICTION
# =========================
predictions = []

for day in range(1, 4):
    X = current_row[model_features]
    aqi_pred = model.predict(X)[0]
    predictions.append(aqi_pred)

    # ---- Update features for next day ----
    # Keep pollution stable for short-term forecast
    current_row['pm2.5'] = current_row['pm2.5']
    current_row['pm2_5_rolling_7'] = current_row['pm2_5_rolling_7']

    from datetime import timedelta
    current_date = pd.to_datetime(
    df[['year','month','day']].iloc[-1]
)

    current_date += timedelta(days=1)
    current_row['year'] = current_date.year
    current_row['month'] = current_date.month
    current_row['day'] = current_date.day
    current_row['dayofweek'] = current_date.dayofweek
    current_row['is_weekend'] = int(current_date.dayofweek >= 5)


# =========================
# 5. OUTPUT
# =========================
for i, pred in enumerate(predictions, start=1):
    print(f"✅ Predicted AQI for Day {i}: {pred:.2f}")

# =========================
# 6. Labels for AQI Categories
# =========================
forecast = pd.DataFrame({
    "day": ["Day 1", "Day 2", "Day 3"],
    "predicted_aqi": predictions
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

forecast["category"] = forecast["predicted_aqi"].apply(aqi_label)

# ✅ Print forecast with categories to console
print("\n3-Day AQI Forecast with Categories:")
print(forecast)

# =========================
# 7. PLOT 3-DAY AQI TREND
# =========================
import matplotlib.pyplot as plt

days = ["Day 1", "Day 2", "Day 3"]

plt.plot(days, predictions, marker='o')  
plt.xlabel("Forecast Day")
plt.ylabel("AQI")
plt.title("3-Day AQI Forecast (Karachi)")
plt.grid(True)
plt.savefig("aqi_3day_forecast.png")
plt.close()


# =========================
# 8. Save predictions to CSV
# =========================
forecast.to_csv("aqi_3day_forecast.csv", index=False)
print("✓ 3-day AQI forecast saved to aqi_3day_forecast.csv")
