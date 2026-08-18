import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }

def build_all_notebooks(nb_dir="notebooks"):
    os.makedirs(nb_dir, exist_ok=True)
    
    # 01_EDA.ipynb
    nb1 = make_notebook([
        md_cell("# Task 1: Exploratory Data Analysis (EDA)\n## Nectar Intelligent Facilities Platform\nThis notebook analyzes sensor telemetry distributions, missing values, temporal patterns, asset behavior across sites, and key drivers of energy consumption."),
        code_cell("""import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_data, clean_telemetry

df_telem, df_meta, df_conn = load_data('../data')
print(f'Telemetry shape: {df_telem.shape}')
print(f'Metadata shape: {df_meta.shape}')"""),
        md_cell("### 1. Data Cleaning & Missing Values Summary"),
        code_cell("""missing_summary = df_telem.isnull().sum()
print('Missing values count per column:')
print(missing_summary[missing_summary > 0])

df_clean = clean_telemetry(df_telem)
print('\\nMissing values after imputation:', df_clean.isnull().sum().sum())"""),
        md_cell("### 2. Telemetry Statistical Summary"),
        code_cell("""df_clean.describe().round(2)"""),
        md_cell("### 3. Energy & Temperature Patterns Across Operating Modes"),
        code_cell("""df_clean.groupby('operating_mode')[['power_consumption', 'temperature', 'vibration']].mean().round(2)""")
    ])
    
    # 02_Predictive_Maintenance.ipynb
    nb2 = make_notebook([
        md_cell("# Task 2: Predictive Maintenance Modeling\nPredicting equipment failures within the next 24 hours using XGBoost classifier and rolling sensor statistics."),
        code_cell("""import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from src.data_processing import load_data, clean_telemetry
from src.predictive_maintenance import engineer_predictive_maintenance_features, train_predictive_maintenance_model

df_telem, df_meta, _ = load_data('../data')
df_clean = clean_telemetry(df_telem)
df_features, feature_cols = engineer_predictive_maintenance_features(df_clean, df_meta)
print(f'Engineered {len(feature_cols)} features across {len(df_features)} rows.')"""),
        md_cell("### Model Training & Evaluation"),
        code_cell("""model, metrics, importance, (X_test, y_test, y_proba) = train_predictive_maintenance_model(df_features, feature_cols)
print('Evaluation Metrics:', metrics)"""),
        md_cell("### Feature Importance"),
        code_cell("""importance.head(10)""")
    ])
    
    # 03_Energy_Forecasting.ipynb
    nb3 = make_notebook([
        md_cell("# Task 3: Energy Consumption Forecasting\nForecasting 24-hour building energy consumption using XGBoost Regressor and time-series lag features."),
        code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.data_processing import load_data, aggregate_building_energy
from src.energy_forecasting import engineer_energy_forecasting_features, train_energy_forecaster

df_telem, _, _ = load_data('../data')
bldg_energy = aggregate_building_energy(df_telem)
df_features, feature_cols = engineer_energy_forecasting_features(bldg_energy)
print(f'Building energy time-series shape: {df_features.shape}')"""),
        md_cell("### Model Training & Evaluation"),
        code_cell("""model, metrics, test_results = train_energy_forecaster(df_features, feature_cols)
print('Forecasting Evaluation Metrics:', metrics)""")
    ])
    
    # 04_Anomaly_Detection.ipynb
    nb4 = make_notebook([
        md_cell("# Task 4: Anomaly Detection Framework\nDetecting abnormal equipment behavior (power spikes, vibration anomalies, sensor drift) using Isolation Forest and Statistical Z-Score Thresholding."),
        code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.data_processing import load_data, clean_telemetry
from src.anomaly_detection import detect_anomalies

df_telem, _, _ = load_data('../data')
df_clean = clean_telemetry(df_telem)
df_anom_clean, df_anomalies = detect_anomalies(df_clean, contamination=0.015)
print(f'Total anomalies detected: {len(df_anomalies)}')"""),
        md_cell("### Anomaly Breakdown"),
        code_cell("""df_anomalies['anomaly_type'].value_counts()""")
    ])
    
    # 05_Asset_Connectivity.ipynb
    nb5 = make_notebook([
        md_cell("# Task 5: Multi-Asset Connectivity Analysis\nBuilding asset graph, auditing data quality (orphans, duplicate links, invalid parent-child links), and simulating failure propagation."),
        code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.data_processing import load_data
from src.connectivity_analysis import AssetGraphAnalyzer

df_telem, df_meta, df_conn = load_data('../data')
analyzer = AssetGraphAnalyzer(df_meta, df_conn)"""),
        md_cell("### 1. Data Quality Audit Findings"),
        code_cell("""dq = analyzer.audit_data_quality()
print('Orphan Assets:', dq['orphan_assets'])
print('Duplicate Connections Count:', len(dq['duplicate_connections']))
print('Invalid Mappings Count:', len(dq['invalid_mappings']))"""),
        md_cell("### 2. Failure Propagation Simulation (Chiller Failure)"),
        code_cell("""fail_sim = analyzer.simulate_failure_propagation('Chiller_Bldg_A1_01')
print(f'Failure of {fail_sim["failed_asset"]} impacts {fail_sim["impact_count"]} downstream assets:')
for a in fail_sim['impacted_assets']:
    print(f'  -> {a["asset_id"]} ({a["asset_type"]}), distance={a["distance"]}')""")
    ])

    notebooks = {
        "01_EDA.ipynb": nb1,
        "02_Predictive_Maintenance.ipynb": nb2,
        "03_Energy_Forecasting.ipynb": nb3,
        "04_Anomaly_Detection.ipynb": nb4,
        "05_Asset_Connectivity.ipynb": nb5
    }
    
    for filename, nb_obj in notebooks.items():
        filepath = os.path.join(nb_dir, filename)
        with open(filepath, "w") as f:
            json.dump(nb_obj, f, indent=2)
        print(f"Created notebook: {filepath}")

if __name__ == "__main__":
    build_all_notebooks("notebooks")
