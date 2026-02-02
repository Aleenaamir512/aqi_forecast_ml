import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="🌤️",
    layout="centered"
)

# =========================
# STYLING
# =========================
st.markdown("""
<style>
.main .block-container {
    background-color: #FFF0F5;
    padding: 2rem;
    border-radius: 10px;
}
.stButton>button {
    background-color: #0288D1;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR: MODEL INFO
# =========================
st.sidebar.title("Model Information")
st.sidebar.markdown("""
**Best Model:** Random Forest Regressor  
**R² Score:** 0.87  
**RMSE:** 12.4   
**MAE:** 7.60  
**Prediction Type:** Recursive 3-Day Forecast  
**Data Source:** MongoDB Feature Store  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🌫 AQI Category Reference")
st.sidebar.markdown("""
🟢 **Good**  
🟡 **Moderate**  
🟠 **Unhealthy (Sensitive Groups)**  
🔴 **Unhealthy**  
⚫ **Very Unhealthy**
""")
st.sidebar.markdown("---")
st.sidebar.info("⚠️ AQI values are predictive estimates, not official measurements.")

# =========================
# HEADER (MAIN PAGE)
# =========================
st.title("Karachi AQI Predictor")
st.write("3-day Air Quality Index (AQI) forecast using Machine Learning")

# =========================
# FUNCTION TO RUN MODEL
# =========================
def run_prediction():
    # Make sure current working directory is the script directory
    BASE_DIR = Path(__file__).resolve().parent
    os.chdir(BASE_DIR)
    
    # Import prediction_model here so it runs in same Python process
    import prediction_model  # your script should define forecast dataframe after running

    # Connect to MongoDB
    MONGO_URI = "mongodb+srv://aleenaamir02_db_user:zY4PRfUXbOplm3Ae@cluster0.km7h66h.mongodb.net/?appName=Cluster0"
    client = MongoClient(MONGO_URI)
    db = client["aqi_database"]
    collection = db["aqi_forecast"]

    forecast = pd.DataFrame(list(collection.find()))
    forecast.drop(columns=["_id"], inplace=True, errors="ignore")

    if forecast.empty:
        st.error("No forecast found in MongoDB.")
        return None

    # Standardize column names
    forecast.columns = [col.lower() for col in forecast.columns]
    if 'category' not in forecast.columns:
        for col in forecast.columns:
            if 'cat' in col:
                forecast.rename(columns={col: 'category'}, inplace=True)

    # Next 3 days dates
    today = datetime.now()
    next_3_dates = [(today + timedelta(days=i+1)).strftime("%d %b %Y") for i in range(3)]
    forecast['days'] = next_3_dates

    # AQI category icons
    def aqi_icon(cat):
        return {
            "Good": "🟢 Good",
            "Moderate": "🟡 Moderate",
            "Unhealthy for Sensitive Groups": "🟠 Unhealthy (Sensitive)",
            "Unhealthy": "🔴 Unhealthy",
            "Very Unhealthy": "⚫ Very Unhealthy"
        }.get(cat, cat)

    forecast["AQI Category"] = forecast["category"].apply(aqi_icon)
    return forecast

# =========================
# RUN PREDICTION BUTTON
# =========================
if st.button("🔮 Generate 3-Day AQI Forecast"):
    with st.spinner("Running model & fetching latest forecast..."):
        try:
            forecast = run_prediction()
            if forecast is not None:
                # Display Table
                st.subheader("📋 3-Day AQI Forecast")
                st.table(forecast[["days", "predicted aqi", "AQI Category"]])

                # Display Chart
                st.subheader("📈 AQI Trend")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(
                    forecast["days"],
                    forecast["predicted aqi"],
                    marker="o",
                    linestyle="-",
                    color="blue"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("AQI")
                ax.set_title("3-Day AQI Forecast (Karachi)")
                ax.grid(True)
                for i, val in enumerate(forecast["predicted aqi"]):
                    ax.text(i, val + 1, f"{val:.1f}", ha="center")
                st.pyplot(fig)

        except Exception as e:
            st.error("Prediction failed!")
            st.code(str(e))

# =========================
# FOOTER
# =========================
st.markdown("""
---
📡 Forecast updates when the model is executed.  
""")
