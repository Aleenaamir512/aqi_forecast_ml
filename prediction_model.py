import pandas as pd
import joblib
from pathlib import Path

# =========================
# 1. LOAD FEATURE DATA
# =========================
feature_file = Path("feature_engineered_historical_with_realtime.pkl")
df = pd.read_pickle(feature_file)

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
model = joblib.load("XGBoost_karachi_aqi_model.pkl")
print("✓ Model loaded")

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


    current_row['day'] += 1
    current_row['dayofweek'] = (current_row['dayofweek'] + 1) % 7
    current_row['is_weekend'] = current_row['dayofweek'].isin([5, 6]).astype(int)

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
plt.show()

# =========================
# 8. Save predictions to CSV
# =========================
forecast.to_csv("aqi_3day_forecast.csv", index=False)
print("✓ 3-day AQI forecast saved to aqi_3day_forecast.csv")
