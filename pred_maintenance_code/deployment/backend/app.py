import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify

# -----------------------------
# Load Model 
# -----------------------------
def load_model():
    model_path = hf_hub_download(
        repo_id="asvravi/asv-preventive-maintenance",
        filename="preventive_maintenance_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Initialize Flask App
# -----------------------------
predictive_maintenance_api = Flask("Predictive Maintenance")

# -----------------------------
# Health Check Route
# -----------------------------
@predictive_maintenance_api.get("/")
def home():
    return jsonify({
        "message": "Engine Predictive Maintenance API is running."
    })

# -----------------------------
# Prediction Endpoint
# -----------------------------
@predictive_maintenance_api.post("/v1/PredictiveMaintenance")
def predict_engine_condition():

    try:
        # Get JSON data
        sensor_data = request.get_json()

        if not sensor_data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract required features
        data_info = {
            "engine_rpm": float(sensor_data.get("engine_rpm")),
            "lub_oil_pressure": float(sensor_data.get("lub_oil_pressure")),
            "fuel_pressure": float(sensor_data.get("fuel_pressure")),
            "coolant_pressure": float(sensor_data.get("coolant_pressure")),
            "lub_oil_temp": float(sensor_data.get("lub_oil_temp")),
            "coolant_temp": float(sensor_data.get("coolant_temp"))
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data_info])

        # Predict probability
        fault_prob = model.predict_proba(input_df)[0][1]
        
        result = {
            "fault_probability": round(float(fault_prob), 4)
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    """
    We can expose another API to predict for a batch of inputs, if required
    """
