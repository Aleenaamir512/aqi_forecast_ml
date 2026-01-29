import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Optional: if using xgboost
from xgboost import XGBRegressor

# =========================
# 1. LOAD CSV
# =========================
CSV_PATH = r"D:\aqi\karachi_aqi.csv"  # Change this to your CSV path

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")  # normalize column names

print("✓ CSV loaded")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================
# 2. DATE PARSING
# =========================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna()
df = df.sort_values("date")

# =========================
# 3. FEATURE ENGINEERING
# =========================
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = df["dayofweek"] >= 5
df["pm2_5_roll7"] = df["pm2.5"].rolling(7).mean().fillna(df["pm2.5"])

# =========================
# 4. SELECT FEATURES & TARGET
# =========================
FEATURES = ["year", "month", "day", "dayofweek", "is_weekend",
            "pm2.5", "pm10", "no2", "so2", "co", "o3",
            "temperature", "humidity", "precipitation", "pm2_5_roll7"]

TARGET = "next_day_aqi"

X = df[FEATURES]
y = df[TARGET]

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. DEFINE MODELS
# =========================
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
}

# =========================
# 7. TRAIN & EVALUATE MODELS
# =========================
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "R2": r2}
    
    # Save each model
    joblib.dump(model, f"{name}_karachi_aqi_model.pkl")
    
    print(f"✓ {name} trained and saved")
    print(f"  MAE: {mae:.2f}, R²: {r2:.2f}")

# =========================
# 8. SUMMARY
# =========================
print("\n=== Model Comparison ===")
for name, metrics in results.items():
    print(f"{name}: MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.2f}")
