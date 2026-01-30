import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Karachi 3-Day AQI Forecast",
    layout="centered"
)

# =========================
# 2. TITLE & DESCRIPTION
# =========================
st.title("🌆 Karachi 3-Day AQI Forecast")
st.markdown(
    "This app shows the predicted Air Quality Index (AQI) for the next 3 days in Karachi. "
    "The AQI categories are color-coded for easy reference."
)

# =========================
# 3. LOAD FORECAST DATA
# =========================
try:
    forecast = pd.read_csv("aqi_3day_forecast.csv")
except FileNotFoundError:
    st.error("Forecast CSV file not found. Make sure `aqi_3day_forecast.csv` is in the same folder.")
    st.stop()

# =========================
# 4. ADD AQI CATEGORY LABELS
# =========================
def aqi_label(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 150: return "🟠 Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "🔴 Unhealthy"
    else: return "⚫ Very Unhealthy"

forecast["AQI Category"] = forecast["predicted_aqi"].apply(aqi_label)

# =========================
# 5. DISPLAY TABLES
# =========================
st.subheader("3-Day Forecast Table")
st.table(forecast[["day", "predicted_aqi", "AQI Category"]])

# =========================
# 6. PLOT LINE CHART
# =========================
st.subheader("3-Day AQI Trend")
fig, ax = plt.subplots()
ax.plot(forecast["day"], forecast["predicted_aqi"], marker='o', linestyle='-', color='blue')
ax.set_xlabel("Day")
ax.set_ylabel("Predicted AQI")
ax.set_title("AQI Forecast Trend")
ax.grid(True)

# Add category markers on chart
for i, aqi in enumerate(forecast["predicted_aqi"]):
    ax.text(i, aqi + 1, f"{int(aqi)}", ha='center')

st.pyplot(fig)

# =========================
# 7. FOOTER
# =========================
st.markdown(
    "---\n"
    "⚠️ AQI values are **predicted estimates** and may differ from actual measurements.\n"
    "📊 Data updated daily via GitHub Actions."
)
