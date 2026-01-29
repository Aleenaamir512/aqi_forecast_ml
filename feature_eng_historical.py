import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

# =========================
# 1. LOAD HISTORICAL CSV
# =========================
file_path = Path(r"D:\aqi\karachi_aqi.csv")

df = pd.read_csv(file_path, sep=",", engine="python", on_bad_lines="skip")

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("✓ CSV loaded")
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================
# 2. DATE HANDLING
# =========================
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")

# =========================
# 3. TIME FEATURES
# =========================
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek
df["is_weekend"] = df["weekday"] >= 5

# Rolling 7-day PM2.5
df["pm2_5_rolling_7"] = df["pm2.5"].rolling(7).mean().fillna(df["pm2.5"])


# =========================
# 4. TARGET VARIABLE
# =========================
df["pm25"] = df["pm2.5"]

# =========================
# 5. LAG FEATURES
# =========================
df["pm25_lag1"] = df["pm25"].shift(1)
df["pm25_lag3"] = df["pm25"].shift(3)

# =========================
# 6. ROLLING FEATURES
# =========================
df["pm25_roll3"] = df["pm25"].rolling(window=3).mean()
df["pm25_roll7"] = df["pm25"].rolling(window=7).mean()

# =========================
# 7. DROP NA VALUES
# =========================
required_cols = ["pm25", "pm25_lag1", "pm25_lag3", "pm25_roll3", "pm25_roll7"]
before = df.shape[0]
df = df.dropna(subset=required_cols)
after = df.shape[0]

print(f"Dropped {before - after} rows due to NaNs")
print("Final dataset shape:", df.shape)

# =========================
# 8. FINAL FEATURE SET
# =========================
final_features = [
    "day", "month", "weekday", "is_weekend",
    "pm25_lag1", "pm25_lag3", "pm25_roll3", "pm25_roll7"
]

X = df[final_features]
y = df["pm25"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# =========================
# 9. FETCH REAL-TIME WAQI DATA
# =========================
WAQI_TOKEN = "f1b6e52a84f2193998bc5a0c04f130531583a24b"
STATION_ID = "A544966"
api_url = f"https://api.waqi.info/feed/{STATION_ID}/?token={WAQI_TOKEN}"

try:
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()
    
    if data["status"] == "ok":
        aqi = data["data"]["aqi"]
        pollutants = data["data"]["iaqi"]
        pm25_now = pollutants.get("pm25", {}).get("v", None)
        pm10_now = pollutants.get("pm10", {}).get("v", None)
        no2_now = pollutants.get("no2", {}).get("v", None)
        so2_now = pollutants.get("so2", {}).get("v", None)
        o3_now = pollutants.get("o3", {}).get("v", None)
        co_now = pollutants.get("co", {}).get("v", None)

        print("✓ WAQI data fetched")
        print(f"AQI: {aqi}, PM2.5: {pm25_now}, PM10: {pm10_now}")
    else:
        print("⚠️ WAQI API returned error:", data.get("data"))
except Exception as e:
    print("⚠️ Could not fetch WAQI API data:", e)
    pm25_now = pm10_now = no2_now = so2_now = o3_now = co_now = None

# =========================
# 10. ADD REAL-TIME DATA TO FEATURES
# =========================
# You can append the real-time PM2.5 as the latest row for prediction
from datetime import datetime

latest_row = {
    "year": datetime.now().year,
    "month": datetime.now().month,
    "day": datetime.now().day,
    "dayofweek": datetime.now().weekday(),
    "is_weekend": datetime.now().weekday() >= 5,
    "pm2.5": pm25_now,  # from WAQI API
    "pm10": pm10_now,   # from WAQI API
    "no2": no2_now,     # from WAQI API
    "so2": so2_now,
    "co": co_now,
    "o3": o3_now,
    "temperature": df["temperature"].iloc[-1],
    "humidity": df["humidity"].iloc[-1],
    "precipitation": df["precipitation"].iloc[-1],
    "pm2_5_rolling_7": df["pm2.5"].iloc[-7:].mean()
}





X_real_time = pd.DataFrame([latest_row])
print("Real-time feature row ready:")
print(X_real_time)


df.rename(columns={"weekday": "dayofweek"}, inplace=True)

# =========================
# 11. SAVE FEATURE ENGINEERED DATA
# =========================
output_path = Path(r"D:\aqi\feature_engineered_historical_with_realtime.pkl")
df.to_pickle(output_path)
print(f"✓ Feature engineered data saved to {output_path}")
