import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pathlib import Path
from datetime import date, datetime, timedelta
import sys
import os


st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="🌤️",
    layout="centered"
)


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

#header
st.title("Karachi AQI Predictor")
st.write("3-day Air Quality Index (AQI) forecast using Machine Learning")

#function
def run_prediction():
    
    BASE_DIR = Path(__file__).resolve().parent
    os.chdir(BASE_DIR)
    
    import prediction_model  

    import streamlit as st
    from pymongo import MongoClient

    MONGO_URI = st.secrets["MONGO_URI"]
    client = MongoClient(MONGO_URI)
    db = client["aqi_database"]
    collection = db["aqi_forecast"]


    db = client["aqi_database"]
    collection = db["aqi_forecast"]

    forecast = pd.DataFrame(list(collection.find()))
    forecast.drop(columns=["_id"], inplace=True, errors="ignore")

    if forecast.empty:
        st.error("No forecast found in MongoDB.")
        return None

    forecast.columns = [col.lower() for col in forecast.columns]
    if 'category' not in forecast.columns:
        for col in forecast.columns:
            if 'cat' in col:
                forecast.rename(columns={col: 'category'}, inplace=True)

    #for next 3 days
    from datetime import date, timedelta

    num_days = len(forecast)
    today = date.today()
    next_3_dates = [(today + timedelta(days=i+1)).strftime("%d %b %Y") for i in range(num_days)]
    forecast['days'] = next_3_dates


    #aqi category
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

#button
if st.button("Generate Now"):
    with st.spinner("Running model & fetching latest forecast..."):
        try:
            forecast = run_prediction()
            if forecast is not None:
                display_forecast = forecast.rename(columns={
                    "days": "Date",
                    "predicted aqi": "Predicted AQI",
                    "AQI Category": "AQI Category"
                })
                #table
                st.subheader("📋 3-Day AQI Forecast")
                st.dataframe(
                    display_forecast[["Date", "Predicted AQI", "AQI Category"]]) 

                #chart
                st.subheader(" AQI Trend")
                fig, ax = plt.subplots(figsize=(8, 4))
                #plotting aqi values
                ax.plot(
                    forecast["days"],
                    forecast["predicted aqi"],
                    marker="o",
                    linestyle="-",
                    color="blue",
                    linewidth=2
                )

                #labels and title 
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted AQI")
                ax.set_title("3-Day AQI Forecast (Karachi)")
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.xticks(rotation=30)

                #annotations
                offset = max(forecast["predicted aqi"]) * 0.02
                for x, val in zip(forecast["days"], forecast["predicted aqi"]):
                 pass 

                
                st.pyplot(fig)  

        except Exception as e:

                st.error("Prediction failed!")
                st.code(str(e))

st.markdown("""
---
📡 Forecast updates when the model is executed.  
""")
