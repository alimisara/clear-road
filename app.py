import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import streamlit as st

# =========================
# LOAD MODEL
# =========================
model = joblib.load("traffic_model.pkl")

st.title("🚗 Prédiction du trafic (Smart City)")
st.write("Application de prédiction du trafic routier en temps réel")

# =========================
# INPUTS
# =========================
hour = st.slider("Heure", 0, 23, 12)
month = st.slider("Mois", 1, 12, 6)
dayofweek = st.slider("Jour de la semaine (0=Lundi)", 0, 6, 2)

is_weekend = 1 if dayofweek in [5, 6] else 0
is_holiday = st.selectbox("Jour férié ?", [0, 1])

temp = st.number_input("Température (K)", value=290.0)
rain_1h = st.number_input("Pluie (mm)", value=0.0)
snow_1h = st.number_input("Neige", value=0.0)
clouds_all = st.slider("Nuages (%)", 0, 100, 50)

weather = st.selectbox("Météo", [
    "Clouds", "Drizzle", "Fog", "Haze", "Mist",
    "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
])

# =========================
# ENCODING WEATHER
# =========================
weather_cols = [
    'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog',
    'weather_main_Haze', 'weather_main_Mist', 'weather_main_Rain',
    'weather_main_Smoke', 'weather_main_Snow', 'weather_main_Squall',
    'weather_main_Thunderstorm'
]

weather_values = [1 if weather == col.split('_')[-1] else 0 for col in weather_cols]

# =========================
# FINAL INPUT (EXACT ORDER X_train)
# =========================
input_data = np.array([[
    temp,
    rain_1h,
    snow_1h,
    clouds_all,
    hour,
    month,
    dayofweek,
    is_weekend,
    is_holiday,
    *weather_values
]])

# =========================
# PREDICTION
# =========================
if st.button("🚦 Prédire le trafic"):

    prediction = model.predict(input_data)[0]

    st.success(f"🚗 Trafic estimé: {int(prediction)} véhicules")

    # =========================
    # GRAPH 1: TRAFFIC LEVEL
    # =========================
    st.subheader("📊 Niveau de trafic")

    categories = ["Faible", "Moyen", "Élevé"]
    level = [0, 0, 0]

    if prediction < 2000:
        level = [1, 0, 0]
    elif prediction < 4500:
        level = [0, 1, 0]
    else:
        level = [0, 0, 1]

    fig, ax = plt.subplots()
    ax.bar(categories, level)
    ax.set_ylabel("Niveau")
    ax.set_title("Classification du trafic")

    st.pyplot(fig)

    # =========================
    # GRAPH 2: HOURLY IMPACT
    # =========================
    st.subheader("📈 Impact de l'heure sur le trafic")

    hours = list(range(24))
    predictions_hour = []

    for h in hours:
        temp_input = input_data.copy()
        temp_input[0][4] = h  # modifier hour

        pred = model.predict(temp_input)[0]
        predictions_hour.append(pred)

    fig2, ax2 = plt.subplots()
    ax2.plot(hours, predictions_hour)
    ax2.set_xlabel("Heure")
    ax2.set_ylabel("Trafic")
    ax2.set_title("Variation du trafic selon l'heure")

    st.pyplot(fig2)