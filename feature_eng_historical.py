import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient

file_path = Path(r"D:\aqi\karachi_aqi.csv")
df = pd.read_csv(file_path, sep=",", engine="python", on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("✓ CSV loaded")
print("Initial shape:", df.shape)

#date 
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")

#time features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = df["dayofweek"] >= 5

df["pm2_5_rolling_7"] = df["pm2.5"].rolling(7).mean().fillna(df["pm2.5"])
df["pm25"] = df["pm2.5"]

#lag and rolling
df["pm25_lag1"] = df["pm25"].shift(1)
df["pm25_lag3"] = df["pm25"].shift(3)
df["pm25_roll3"] = df["pm25"].rolling(3).mean()
df["pm25_roll7"] = df["pm25"].rolling(7).mean()

required_cols = ["pm25", "pm25_lag1", "pm25_lag3", "pm25_roll3", "pm25_roll7"]
before = df.shape[0]
df = df.dropna(subset=required_cols)
after = df.shape[0]
print(f"Dropped {before - after} rows due to NaNs")
print("Final dataset shape:", df.shape)

#realtime data
WAQI_TOKEN = "f1b6e52a84f2193998bc5a0c04f130531583a24b"
STATION_ID = "A544966"
api_url = f"https://api.waqi.info/feed/{STATION_ID}/?token={WAQI_TOKEN}"

try:
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] == "ok":
        pollutants = data["data"]["iaqi"]
        pm25_now = pollutants.get("pm25", {}).get("v")
        pm10_now = pollutants.get("pm10", {}).get("v")
        no2_now = pollutants.get("no2", {}).get("v")
        so2_now = pollutants.get("so2", {}).get("v")
        o3_now = pollutants.get("o3", {}).get("v")
        co_now = pollutants.get("co", {}).get("v")
        print("WAQI data fetched")
    else:
        print("WAQI API returned error")
        pm25_now = pm10_now = no2_now = so2_now = o3_now = co_now = None
except Exception as e:
    print("Could not fetch WAQI API data:", e)
    pm25_now = pm10_now = no2_now = so2_now = o3_now = co_now = None

latest_row = {
    "date": datetime.now(),
    "year": datetime.now().year,
    "month": datetime.now().month,
    "day": datetime.now().day,
    "dayofweek": datetime.now().weekday(),
    "is_weekend": datetime.now().weekday() >= 5,
    "pm2.5": pm25_now,
    "pm10": pm10_now,
    "no2": no2_now,
    "so2": so2_now,
    "co": co_now,
    "o3": o3_now,
    "temperature": df["temperature"].iloc[-1] if "temperature" in df.columns else None,
    "humidity": df["humidity"].iloc[-1] if "humidity" in df.columns else None,
    "precipitation": df["precipitation"].iloc[-1] if "precipitation" in df.columns else None,
    "pm2_5_rolling_7": df["pm2.5"].iloc[-7:].mean(),
    "pm25": pm25_now,
    "pm25_lag1": df["pm25"].iloc[-1] if len(df) >= 1 else None,
    "pm25_lag3": df["pm25"].iloc[-3] if len(df) >= 3 else None,
    "pm25_roll3": df["pm25_roll3"].iloc[-1] if len(df) >= 1 else None,
    "pm25_roll7": df["pm25_roll7"].iloc[-1] if len(df) >= 1 else None,
}

df = pd.concat([df, pd.DataFrame([latest_row])], ignore_index=True)

#clean NaNs
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = df[col].apply(lambda x: x.to_pydatetime() if pd.notnull(x) else None)
    else:
        df[col] = df[col].where(pd.notnull(df[col]), None)

print("✓ Latest real-time row appended and cleaned")

#save feature eng data
output_path = Path(r"D:\aqi\feature_engineered_historical_with_realtime.pkl")
df.to_pickle(output_path)
print(f"✓ Feature engineered data saved to {output_path}")

#upload to mongodb
MONGO_URI = "mongodb+srv://aleenaamir02_db_user:zY4PRfUXbOplm3Ae@cluster0.km7h66h.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["aqi_database"]
collection = db["feature_store"]

collection.delete_many({})
collection.insert_many(df.to_dict("records"))

print("Feature data uploaded to MongoDB:", collection.count_documents({}), "records")
