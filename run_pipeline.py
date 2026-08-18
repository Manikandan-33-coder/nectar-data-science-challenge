import os
import pandas as pd
import numpy as np
import json

from src.data_processing import load_data, clean_telemetry, aggregate_building_energy
from src.predictive_maintenance import engineer_predictive_maintenance_features, train_predictive_maintenance_model
from src.energy_forecasting import engineer_energy_forecasting_features, train_energy_forecaster
from src.anomaly_detection import detect_anomalies
from src.connectivity_analysis import AssetGraphAnalyzer

def run_full_pipeline(data_dir="data", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("=== Step 1: Loading Data ===")
    df_telem, df_meta, df_conn = load_data(data_dir)
    print(f"Loaded Telemetry: {df_telem.shape}, Metadata: {df_meta.shape}, Connectivity: {df_conn.shape}")
    
    print("\n=== Step 2: Predictive Maintenance Modeling (Task 2) ===")
    df_clean_telem = clean_telemetry(df_telem)
    df_pm_feats, pm_feature_cols = engineer_predictive_maintenance_features(df_clean_telem, df_meta)
    
    pm_model_path = os.path.join("models", "predictive_maintenance_model.joblib")
    pm_model, pm_metrics, pm_importance, _ = train_predictive_maintenance_model(
        df_pm_feats, pm_feature_cols, model_path=pm_model_path
    )
    print("Predictive Maintenance Evaluation Metrics:")
    for k, v in pm_metrics.items():
        print(f"  - {k}: {v:.4f}")
    print("\nTop 5 Predictive Features:")
    print(pm_importance.head(5).to_string(index=False))
    
    print("\n=== Step 3: Energy Consumption Forecasting (Task 3) ===")
    df_bldg_energy = aggregate_building_energy(df_telem)
    df_energy_feats, energy_feature_cols = engineer_energy_forecasting_features(df_bldg_energy)
    
    energy_model_path = os.path.join("models", "energy_forecasting_model.joblib")
    energy_model, energy_metrics, energy_results = train_energy_forecaster(
        df_energy_feats, energy_feature_cols, model_path=energy_model_path
    )
    print("Energy Forecasting Evaluation Metrics:")
    for k, v in energy_metrics.items():
        print(f"  - {k}: {v}")
        
    print("\n=== Step 4: Anomaly Detection Framework (Task 4) ===")
    df_anom_clean, df_anomalies = detect_anomalies(df_clean_telem, contamination=0.015)
    print(f"Total Anomalies Flagged: {len(df_anomalies)}")
    print("Anomaly Distribution by Type:")
    print(df_anomalies["anomaly_type"].value_counts().to_string())
    df_anomalies.to_csv(os.path.join(output_dir, "detected_anomalies.csv"), index=False)
    
    print("\n=== Step 5: Multi-Asset Connectivity Analysis (Task 5) ===")
    analyzer = AssetGraphAnalyzer(df_meta, df_conn)
    dq_findings = analyzer.audit_data_quality()
    print("Data Quality Assessment Findings:")
    print(f"  - Duplicate Connections: {len(dq_findings['duplicate_connections'])}")
    print(f"  - Orphan Assets: {dq_findings['orphan_assets']}")
    print(f"  - Invalid Mappings: {dq_findings['invalid_mappings']}")
    print(f"  - Missing Relationships: {dq_findings['missing_relationships']}")
    
    # Simulate Failure Propagation for Chiller_Bldg_A1_01
    fail_sim = analyzer.simulate_failure_propagation("Chiller_Bldg_A1_01")
    print(f"\nFailure Propagation Simulation for {fail_sim['failed_asset']} ({fail_sim['failed_asset_type']}):")
    print(f"  - Downstream Impacted Assets Count: {fail_sim['impact_count']}")
    for asset_info in fail_sim['impacted_assets']:
        print(f"    -> {asset_info['asset_id']} ({asset_info['asset_type']}), Distance: {asset_info['distance']}")
        
    summary_results = {
        "predictive_maintenance_metrics": pm_metrics,
        "energy_forecasting_metrics": energy_metrics,
        "anomalies_detected_count": len(df_anomalies),
        "data_quality_findings": dq_findings,
        "failure_propagation_example": fail_sim
    }
    
    with open(os.path.join(output_dir, "summary_results.json"), "w") as f:
        json.dump(summary_results, f, indent=2)
        
    print(f"\nPipeline Execution Successfully Completed! Summary saved to {output_dir}/summary_results.json")

if __name__ == "__main__":
    run_full_pipeline("data", "output")
