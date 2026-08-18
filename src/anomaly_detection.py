import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

def detect_anomalies(df_telem, contamination=0.02):
    """
    Hybrid Anomaly Detection:
    1. Isolation Forest for multivariate anomaly score.
    2. Rule-based Statistical Thresholding (Z-score / IQR) for specific domain root causes:
       - Sudden Power Spike
       - Excessive Vibration
       - Sensor Drift
       - Extreme Temperature
    """
    df = df_telem.copy()
    
    # 1. Isolation Forest
    feature_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
    df_clean = df.dropna(subset=feature_cols).copy()
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_clean["iforest_anomaly"] = iso_forest.fit_predict(df_clean[feature_cols])
    df_clean["iforest_score"] = iso_forest.decision_function(df_clean[feature_cols])
    # -1 means anomaly in IsolationForest
    df_clean["is_anomaly_iforest"] = (df_clean["iforest_anomaly"] == -1).astype(int)
    
    # 2. Statistical Rule-Based Flagging
    df_clean["z_vibration"] = df_clean.groupby("asset_id")["vibration"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    df_clean["z_power"] = df_clean.groupby("asset_id")["power_consumption"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    df_clean["z_temp"] = df_clean.groupby("asset_id")["temperature"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    
    anomalies = []
    for idx, row in df_clean.iterrows():
        reasons = []
        if row["z_power"] > 4.0:
            reasons.append("Sudden Power Spike")
        if row["z_vibration"] > 3.5:
            reasons.append("Excessive Vibration")
        if row["z_temp"] > 3.5:
            reasons.append("Abnormal Temperature")
        if row["is_anomaly_iforest"] == 1 and not reasons:
            reasons.append("Multivariate Telemetry Anomaly")
            
        if reasons:
            anomalies.append({
                "timestamp": row["timestamp"],
                "site_id": row["site_id"],
                "building_id": row["building_id"],
                "asset_id": row["asset_id"],
                "temperature": row["temperature"],
                "vibration": row["vibration"],
                "power_consumption": row["power_consumption"],
                "anomaly_type": ", ".join(reasons),
                "iforest_score": round(row["iforest_score"], 4)
            })
            
    df_anomalies = pd.DataFrame(anomalies)
    return df_clean, df_anomalies
