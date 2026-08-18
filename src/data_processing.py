import pandas as pd
import numpy as np
import os

def load_data(data_dir="data"):
    telem_path = os.path.join(data_dir, "telemetry.csv")
    meta_path = os.path.join(data_dir, "asset_metadata.csv")
    conn_path = os.path.join(data_dir, "asset_connectivity.csv")
    
    df_telem = pd.read_csv(telem_path)
    df_telem["timestamp"] = pd.to_datetime(df_telem["timestamp"])
    
    df_meta = pd.read_csv(meta_path)
    df_conn = pd.read_csv(conn_path)
    
    return df_telem, df_meta, df_conn

def clean_telemetry(df_telem):
    """
    Impute missing values using forward fill per asset and median backfill
    """
    df = df_telem.sort_values(["asset_id", "timestamp"]).copy()
    
    numeric_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
    for col in numeric_cols:
        df[col] = df.groupby("asset_id")[col].transform(lambda x: x.ffill().bfill())
        
    return df

def aggregate_building_energy(df_telem):
    """
    Aggregate power consumption to building hourly level for energy forecasting
    """
    df_clean = clean_telemetry(df_telem)
    df_clean["hourly_timestamp"] = df_clean["timestamp"].dt.floor("h")
    
    bldg_energy = df_clean.groupby(["site_id", "building_id", "hourly_timestamp"]).agg({
        "power_consumption": "sum",
        "temperature": "mean",
        "humidity": "mean",
        "occupancy_count": "max"
    }).reset_index()
    
    bldg_energy = bldg_energy.rename(columns={"power_consumption": "total_power_kwh"})
    return bldg_energy
