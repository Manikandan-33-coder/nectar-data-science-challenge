import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_nectar_dataset(output_dir="data", days=60, random_seed=42):
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic Nectar IoT dataset ({days} days)...")

    # 1. Generate Metadata
    sites = ["Site_A", "Site_B", "Site_C"]
    buildings = {
        "Site_A": ["Bldg_A1", "Bldg_A2"],
        "Site_B": ["Bldg_B1", "Bldg_B2"],
        "Site_C": ["Bldg_C1", "Bldg_C2"]
    }
    
    manufacturers = ["Carrier", "Trane", "Daikin", "Honeywell", "Johnson Controls", "Schneider Electric"]
    
    assets_meta = []
    connectivity = []
    
    asset_id_counter = 1
    
    for site_id, bldg_list in buildings.items():
        for bldg_id in bldg_list:
            # Building level energy meter
            meter_id = f"Meter_{bldg_id}_01"
            assets_meta.append({
                "asset_id": meter_id,
                "site_id": site_id,
                "building_id": bldg_id,
                "asset_name": f"Main Energy Meter - {bldg_id}",
                "asset_type": "Energy Meter",
                "manufacturer": "Schneider Electric",
                "installation_date": "2020-03-15",
                "capacity": 1000.0,
                "parent_asset_id": None
            })
            
            # Chillers (2 per building)
            for c_idx in [1, 2]:
                chiller_id = f"Chiller_{bldg_id}_{c_idx:02d}"
                assets_meta.append({
                    "asset_id": chiller_id,
                    "site_id": site_id,
                    "building_id": bldg_id,
                    "asset_name": f"Chiller Unit {c_idx} - {bldg_id}",
                    "asset_type": "Chiller",
                    "manufacturer": np.random.choice(["Carrier", "Trane", "Daikin"]),
                    "installation_date": f"2019-0{c_idx}-10",
                    "capacity": float(np.random.choice([350, 450, 500])),
                    "parent_asset_id": meter_id
                })
                
                connectivity.append({
                    "source_asset_id": meter_id,
                    "target_asset_id": chiller_id,
                    "connection_type": "Monitors",
                    "relationship_strength": 0.95
                })
                
                # Pumps connected to Chiller (1 per chiller)
                pump_id = f"Pump_{bldg_id}_{c_idx:02d}"
                assets_meta.append({
                    "asset_id": pump_id,
                    "site_id": site_id,
                    "building_id": bldg_id,
                    "asset_name": f"Chilled Water Pump {c_idx} - {bldg_id}",
                    "asset_type": "Pump",
                    "manufacturer": "Grundfos",
                    "installation_date": "2020-06-20",
                    "capacity": 75.0,
                    "parent_asset_id": chiller_id
                })
                
                connectivity.append({
                    "source_asset_id": chiller_id,
                    "target_asset_id": pump_id,
                    "connection_type": "Supplies",
                    "relationship_strength": 0.90
                })
                
                # AHUs connected to Chiller (2 per chiller)
                for a_idx in [1, 2]:
                    ahu_num = (c_idx - 1) * 2 + a_idx
                    ahu_id = f"AHU_{bldg_id}_{ahu_num:02d}"
                    assets_meta.append({
                        "asset_id": ahu_id,
                        "site_id": site_id,
                        "building_id": bldg_id,
                        "asset_name": f"Air Handling Unit {ahu_num} - {bldg_id}",
                        "asset_type": "AHU",
                        "manufacturer": np.random.choice(["Honeywell", "Trane", "Daikin"]),
                        "installation_date": "2021-01-15",
                        "capacity": 150.0,
                        "parent_asset_id": chiller_id
                    })
                    
                    connectivity.append({
                        "source_asset_id": chiller_id,
                        "target_asset_id": ahu_id,
                        "connection_type": "Supplies",
                        "relationship_strength": 0.85
                    })
                    
                    # Environmental sensors connected to AHU
                    env_id = f"EnvSens_{bldg_id}_{ahu_num:02d}"
                    assets_meta.append({
                        "asset_id": env_id,
                        "site_id": site_id,
                        "building_id": bldg_id,
                        "asset_name": f"Zone Environmental Sensor {ahu_num} - {bldg_id}",
                        "asset_type": "Environmental Sensor",
                        "manufacturer": "Honeywell",
                        "installation_date": "2021-05-10",
                        "capacity": 10.0,
                        "parent_asset_id": ahu_id
                    })
                    
                    connectivity.append({
                        "source_asset_id": ahu_id,
                        "target_asset_id": env_id,
                        "connection_type": "Controls",
                        "relationship_strength": 0.88
                    })

    # Add Data Quality anomalies to Metadata & Connectivity as required by Task 5:
    # 1. Orphan asset (no parent, not connected)
    orphan_id = "Sensor_Orphan_99"
    assets_meta.append({
        "asset_id": orphan_id,
        "site_id": "Site_A",
        "building_id": "Bldg_A1",
        "asset_name": "Orphaned Temperature Sensor",
        "asset_type": "Environmental Sensor",
        "manufacturer": "Unknown",
        "installation_date": "2018-11-11",
        "capacity": 5.0,
        "parent_asset_id": None
    })
    
    # 2. Duplicate connection entry
    if len(connectivity) > 0:
        connectivity.append(connectivity[0].copy())
        
    # 3. Invalid parent-child mapping (Sensor claiming to parent a Chiller)
    connectivity.append({
        "source_asset_id": "EnvSens_Bldg_A1_01",
        "target_asset_id": "Chiller_Bldg_A1_01",
        "connection_type": "Controls",
        "relationship_strength": 0.10
    })

    df_meta = pd.DataFrame(assets_meta)
    df_conn = pd.DataFrame(connectivity)
    
    meta_path = os.path.join(output_dir, "asset_metadata.csv")
    conn_path = os.path.join(output_dir, "asset_connectivity.csv")
    
    df_meta.to_csv(meta_path, index=False)
    df_conn.to_csv(conn_path, index=False)
    print(f"Saved Metadata ({len(df_meta)} rows) to {meta_path}")
    print(f"Saved Connectivity ({len(df_conn)} rows) to {conn_path}")

    # 2. Generate Telemetry Time Series Data
    start_time = datetime(2026, 6, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(days * 24 * 4)]
    
    telemetry_records = []
    
    # Select subset of assets for full telemetry generation to keep file manageable while highly detailed
    telemetry_assets = df_meta[df_meta["asset_id"] != orphan_id].copy()
    
    print(f"Generating telemetry for {len(telemetry_assets)} assets across {len(timestamps)} timestamps ({len(telemetry_assets) * len(timestamps)} rows)...")

    # Define failure events for predictive maintenance & anomaly testing
    # Inject 4 distinct pre-failure degradation profiles across different assets
    failure_schedule = [
        {"asset_id": "Chiller_Bldg_A1_01", "fail_start": start_time + timedelta(days=7), "fail_end": start_time + timedelta(days=8)},
        {"asset_id": "Pump_Bldg_B1_02", "fail_start": start_time + timedelta(days=14), "fail_end": start_time + timedelta(days=15)},
        {"asset_id": "AHU_Bldg_C2_01", "fail_start": start_time + timedelta(days=21), "fail_end": start_time + timedelta(days=22)},
        {"asset_id": "Chiller_Bldg_B2_02", "fail_start": start_time + timedelta(days=27), "fail_end": start_time + timedelta(days=28)}
    ]
    
    # Sensor drift asset
    drift_asset_id = "EnvSens_Bldg_A2_02"
    
    # Power spike asset
    spike_asset_id = "Pump_Bldg_A1_01"

    for _, asset in telemetry_assets.iterrows():
        a_id = asset["asset_id"]
        a_type = asset["asset_type"]
        s_id = asset["site_id"]
        b_id = asset["building_id"]
        cap = asset["capacity"]
        
        # Base asset noise
        base_temp = 7.0 if a_type == "Chiller" else (22.0 if a_type in ["AHU", "Environmental Sensor"] else 35.0)
        base_vib = 0.8 if a_type in ["Chiller", "Pump"] else 0.2
        base_power = cap * 0.45
        
        asset_failures = [f for f in failure_schedule if f["asset_id"] == a_id]
        
        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()
            is_weekend = day_of_week >= 5
            
            # Diurnal & Occupancy pattern
            if 8 <= hour <= 18 and not is_weekend:
                occupancy = int(np.random.normal(loc=250, scale=40))
                occupancy = max(10, min(500, occupancy))
                load_factor = 0.8 + 0.2 * np.sin((hour - 8) / 10.0 * np.pi)
            else:
                occupancy = int(np.random.normal(loc=15, scale=5))
                occupancy = max(0, occupancy)
                load_factor = 0.3
                
            op_mode = "Idle" if load_factor < 0.35 else ("Cooling" if hour % 2 == 0 else "Heating")
            
            # Baseline metrics with thermal noise
            ambient_effect = 4.0 * np.sin((hour - 6) / 24.0 * 2 * np.pi)
            temp = base_temp + ambient_effect * 0.5 + (occupancy / 500.0) * 3.0 + np.random.normal(0, 0.5)
            humidity = 50.0 - ambient_effect * 1.2 + np.random.normal(0, 1.5)
            pressure = (300.0 if a_type in ["Chiller", "Pump"] else 2.5) + np.random.normal(0, 5.0 if a_type in ["Chiller", "Pump"] else 0.1)
            vibration = base_vib + (load_factor * 0.4) + np.random.normal(0, 0.05)
            power = base_power * load_factor * (1.0 + (temp - base_temp) * 0.01) + np.random.normal(0, 2.0)
            
            fault_flag = 0
            
            # Check degradation / failure degradation phase (24-48 hrs leading to failure)
            for f_info in asset_failures:
                f_start = f_info["fail_start"]
                f_end = f_info["fail_end"]
                deg_start = f_start - timedelta(hours=36)
                
                if f_start <= ts <= f_end:
                    fault_flag = 1
                    vibration *= np.random.uniform(4.0, 7.0)
                    temp += np.random.uniform(15.0, 25.0)
                    power *= np.random.uniform(1.5, 2.5)
                    pressure *= np.random.uniform(0.3, 0.6)
                elif deg_start <= ts < f_start:
                    # Exponential degradation phase
                    hours_to_fail = (f_start - ts).total_seconds() / 3600.0
                    deg_severity = (36.0 - hours_to_fail) / 36.0
                    vibration += deg_severity * 4.5 + np.random.normal(0, 0.2)
                    temp += deg_severity * 10.0
                    power += deg_severity * (base_power * 0.4)
            
            # Inject sensor drift for specific asset
            if a_id == drift_asset_id and ts > start_time + timedelta(days=20):
                days_elapsed = (ts - (start_time + timedelta(days=20))).total_seconds() / 86400.0
                temp += days_elapsed * 0.3 # steady sensor drift
                
            # Inject sudden power spike anomaly
            if a_id == spike_asset_id and ts.day == 25 and ts.hour == 14 and ts.minute == 30:
                power *= 5.2
                vibration *= 3.8
                
            # Inject small percentage of random missing values for EDA data cleaning exercise
            if np.random.rand() < 0.002:
                temp = np.nan
            if np.random.rand() < 0.002:
                vibration = np.nan
            if np.random.rand() < 0.002:
                power = np.nan

            telemetry_records.append({
                "timestamp": ts,
                "site_id": s_id,
                "building_id": b_id,
                "asset_id": a_id,
                "temperature": round(temp, 2) if not np.isnan(temp) else np.nan,
                "humidity": round(humidity, 2) if not np.isnan(humidity) else np.nan,
                "pressure": round(pressure, 2) if not np.isnan(pressure) else np.nan,
                "vibration": round(vibration, 3) if not np.isnan(vibration) else np.nan,
                "power_consumption": round(max(0.0, power), 2) if not np.isnan(power) else np.nan,
                "occupancy_count": occupancy,
                "operating_mode": op_mode,
                "fault_flag": fault_flag
            })

    df_telemetry = pd.DataFrame(telemetry_records)
    telem_path = os.path.join(output_dir, "telemetry.csv")
    df_telemetry.to_csv(telem_path, index=False)
    print(f"Saved Telemetry ({len(df_telemetry)} rows) to {telem_path}")
    print("Dataset generation complete!")

if __name__ == "__main__":
    generate_nectar_dataset("data", days=30)
