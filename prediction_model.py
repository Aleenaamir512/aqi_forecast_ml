from pathlib import Path
import pandas as pd
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

# =========================
# 0. BASE DIRECTORY
# =========================
BASE_DIR = Path(__file__).resolve().parent
print("Current working directory:", BASE_DIR)

# =========================
# 1. LOAD FEATURE DATA
# =========================
feature_file = BASE_DIR / "feature_engineered_historical_with_realtime.pkl"

if not feature_file.exists():
    raise FileNotFoundError(
        f"{feature_file} NOT FOUND. Run feature engineering first."
    )

df = pd.read_pickle(feature_file)
print("✓ Feature data loaded, shape:", df.shape)

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
model_path = BASE_DIR / "XGBoost_karachi_aqi_model.pkl"
model = joblib.load(model_path)
print("✓ Model loaded")

# =========================
# 4. 3-DAY RECURSIVE PREDICTION
# =========================
predictions = []

# ✅ Correct way to get last date
last_date = pd.Timestamp(
    year=int(df.iloc[-1]['year']),
    month=int(df.iloc[-1]['month']),
    day=int(df.iloc[-1]['day'])
)

for day in range(1, 4):
    X = current_row[model_features]
    aqi_pred = model.predict(X)[0]
    predictions.append(aqi_pred)

    # ---- Update date ----
    current_date = last_date + timedelta(days=day)

    current_row['year'] = current_date.year
    current_row['month'] = current_date.month
    current_row['day'] = current_date.day
    current_row['dayofweek'] = current_date.dayofweek
    current_row['is_weekend'] = int(current_date.dayofweek >= 5)

    # ---- Keep pollution stable (short-term assumption) ----
    # No change needed, values stay the same
    current_row['pm2.5'] = current_row['pm2.5']
    current_row['pm2_5_rolling_7'] = current_row['pm2_5_rolling_7']

# =========================
# 5. PRINT OUTPUT
# =========================
for i, pred in enumerate(predictions, start=1):
    print(f"✅ Predicted AQI for Day {i}: {pred:.2f}")

# =========================
# 6. AQI CATEGORIES
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

print("\n3-Day AQI Forecast with Categories:")
print(forecast)

# =========================
# 7. PLOT 3-DAY AQI TREND
# =========================
days = ["Day 1", "Day 2", "Day 3"]

plt.figure()
plt.plot(days, predictions, marker='o')
plt.xlabel("Forecast Day")
plt.ylabel("AQI")
plt.title("3-Day AQI Forecast (Karachi)")
plt.grid(True)
plt.savefig(BASE_DIR / "aqi_3day_forecast.png")
plt.close()

print("✓ AQI forecast plot saved")

# =========================
# 8. SAVE CSV
# =========================
output_csv = BASE_DIR / "aqi_3day_forecast.csv"
forecast.to_csv(output_csv, index=False)
print("✓ 3-day AQI forecast saved to aqi_3day_forecast.csv")
