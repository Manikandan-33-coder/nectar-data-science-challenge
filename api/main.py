from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import sys

app = FastAPI(
    title="Nectar Predictive Maintenance API",
    description="REST API for predicting IoT asset failure probability within 24 hours.",
    version="1.0.0"
)

# Load trained predictive maintenance model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "predictive_maintenance_model.joblib"))

model_artifact = None
if os.path.exists(MODEL_PATH):
    model_artifact = joblib.load(MODEL_PATH)
    print("FastAPI: Predictive Maintenance model loaded successfully.")

class TelemetryPayload(BaseModel):
    asset_id: str = Field(..., example="Chiller_Bldg_A1_01")
    temperature: float = Field(..., example=35.5)
    vibration: float = Field(..., example=6.2)
    pressure: float = Field(..., example=280.0)
    power_consumption: float = Field(..., example=180.0)
    occupancy_count: int = Field(..., example=200)
    capacity: float = Field(..., example=450.0)
    operating_mode: str = Field(..., example="Cooling")
    asset_type: str = Field(..., example="Chiller")
    
    # Optional rolling features (will default to estimates if omitted)
    vibration_roll_max_1h: float = Field(default=None)
    vibration_roll_std_6h: float = Field(default=None)

class PredictionResponse(BaseModel):
    asset_id: str
    failure_probability: float
    risk_level: str
    recommended_action: str

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model_artifact is not None}

@app.post("/predict_failure", response_model=PredictionResponse)
def predict_failure(payload: TelemetryPayload):
    if not model_artifact:
        raise HTTPException(status_code=500, detail="Model artifact not found. Please run the training pipeline first.")
        
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    
    # Prepare input feature dictionary with defaults
    vib_max_1h = payload.vibration_roll_max_1h if payload.vibration_roll_max_1h is not None else payload.vibration * 1.1
    vib_std_6h = payload.vibration_roll_std_6h if payload.vibration_roll_std_6h is not None else 0.5
    
    input_data = {
        "temperature": payload.temperature,
        "vibration": payload.vibration,
        "pressure": payload.pressure,
        "power_consumption": payload.power_consumption,
        "occupancy_count": payload.occupancy_count,
        "capacity": payload.capacity,
        "power_capacity_ratio": payload.power_consumption / (payload.capacity + 1e-5),
        "vibration_roll_max_1h": vib_max_1h,
        "vibration_roll_mean_1h": payload.vibration,
        "vibration_roll_mean_6h": payload.vibration,
        "vibration_roll_std_6h": vib_std_6h,
        "vibration_roll_max_24h": vib_max_1h * 1.2,
        "vibration_delta_24h": payload.vibration * 0.2,
        "temperature_roll_max_1h": payload.temperature,
        "temperature_roll_mean_1h": payload.temperature,
        "temperature_roll_mean_6h": payload.temperature,
        "temperature_roll_std_6h": 0.5,
        "temperature_roll_max_24h": payload.temperature + 2.0,
        "temperature_delta_24h": 1.0,
        "power_consumption_roll_max_1h": payload.power_consumption,
        "power_consumption_roll_mean_1h": payload.power_consumption,
        "power_consumption_roll_mean_6h": payload.power_consumption,
        "power_consumption_roll_std_6h": 5.0,
        "power_consumption_roll_max_24h": payload.power_consumption * 1.1,
        "power_consumption_delta_24h": 5.0,
        "pressure_roll_max_1h": payload.pressure,
        "pressure_roll_mean_1h": payload.pressure,
        "pressure_roll_mean_6h": payload.pressure,
        "pressure_roll_std_6h": 1.0,
        "pressure_roll_max_24h": payload.pressure * 1.05,
        "pressure_delta_24h": 2.0
    }
    
    # Add dummy columns for categories
    df_row = pd.DataFrame([input_data])
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0.0
            
    # Ensure correct column order
    df_row = df_row[feature_cols]
    
    # Predict
    proba = float(model.predict_proba(df_row)[0][1])
    
    if proba >= 0.7:
        risk = "HIGH"
        action = "CRITICAL: Schedule emergency maintenance inspection within 4 hours. Inspect bearings & thermal cooling."
    elif proba >= 0.3:
        risk = "MEDIUM"
        action = "WARNING: Elevate monitoring frequency. Schedule routine maintenance check within 24 hours."
    else:
        risk = "LOW"
        action = "NORMAL: Asset operating within normal thermal and vibration parameters."
        
    return PredictionResponse(
        asset_id=payload.asset_id,
        failure_probability=round(proba, 4),
        risk_level=risk,
        recommended_action=action
    )
