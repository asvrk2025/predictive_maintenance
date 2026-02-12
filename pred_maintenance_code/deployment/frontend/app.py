import streamlit as st
import numpy as np
import pandas as pd
import requests
# -----------------------------
# App Config (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    layout="wide"
)

# -----------------------------
# Reduce HF default padding
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title & Description
# -----------------------------
st.markdown("""
<h2 style="margin-top: 0.5rem;">
🔧 Engine Predictive Maintenance System
</h2>
""", unsafe_allow_html=True)

st.markdown(
    "Predict potential engine failures using real-time sensor inputs. "
    "The model is optimized for **high fault recall** to minimize missed failures."
)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Prediction Settings")

threshold = st.sidebar.slider(
    "Fault Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Lower values increase fault detection (recall), higher values reduce false alarms"
)

# -----------------------------
# Layout (Side-by-side)
# -----------------------------
input_col, result_col = st.columns([2.2, 1.2])

# -----------------------------
# Engine Sensor Inputs
# -----------------------------
with input_col:
    st.subheader("🧮 Engine Sensor Inputs")

    c1, c2 = st.columns(2)

    with c1:
        engine_rpm = st.number_input("Engine RPM", value=1500.0)
        lub_oil_pressure = st.number_input("Lub Oil Pressure", value=4.0)
        fuel_pressure = st.number_input("Fuel Pressure", value=3.5)

    with c2:
        coolant_pressure = st.number_input("Coolant Pressure", value=2.5)
        lub_oil_temp = st.number_input("Lub Oil Temp (°C)", value=85.0)
        coolant_temp = st.number_input("Coolant Temp (°C)", value=90.0)

    predict_btn = st.button("🔍 Predict Engine Health", use_container_width=False)

# -----------------------------
# Prediction Result
# -----------------------------
with result_col:
    st.subheader("📊 Prediction Result")

    with st.container(border=True):
        if predict_btn:
            input_df = pd.DataFrame([{
                "engine_rpm": engine_rpm,
                "lub_oil_pressure": lub_oil_pressure,
                "fuel_pressure": fuel_pressure,
                "coolant_pressure": coolant_pressure,
                "lub_oil_temp": lub_oil_temp,
                "coolant_temp": coolant_temp
            }])

            
            #Call backend API and get the fault_prob of class 1 (Backend API is designed to give the fault probability of class 1)
            payload = input_df.iloc[0].to_dict()
            response = requests.post (
                "https://asvravi-asv-predictive-maintenance-backend.hf.space/v1/PredictiveMaintenance",
                json=payload
                )
            if response.status_code == 200:
                result = response.json ()
                fault_prob = result.get ("fault_probability")  # Extract only the value
            else:
                st.error (f"Error in API request - {response.status_code}")

            #get the final prediction based on fault probability and the user defined threshold
            prediction = 1 if fault_prob >= threshold else 0

            if prediction == 1:
                st.error(
                    f"⚠️ **FAULT LIKELY**\n\n"
                    f"Estimated Fault Probability: **{fault_prob:.2f}**\n\n"
                    "Immediate inspection or preventive maintenance is recommended."
                )
            else:
                st.success(
                    f"✅ **ENGINE HEALTHY**\n\n"
                    f"Estimated Fault Probability: **{fault_prob:.2f}**\n\n"
                    "No immediate maintenance action required."
                )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Model: Tuned XGBoost | Objective: Predictive Maintenance | "
    "Optimized for high fault recall to reduce missed failures"
)
