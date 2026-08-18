# ⚡ Nectar Data Scientist Challenge Solution

This repository contains the complete end-to-end data science solution for **Nectar's Intelligent Facilities Platform Challenge**.

---

## 📁 Repository Directory Structure

```
nectar_ds_challenge/
├── data/                               # Generated datasets
│   ├── telemetry.csv                   # 30-day 15-min sensor telemetry stream (224,640 rows)
│   ├── asset_metadata.csv              # Asset metadata (79 assets)
│   └── asset_connectivity.csv          # Asset dependency graph edges (74 edges)
├── generate_data.py                    # Physics-informed synthetic IoT data generator
├── src/                                # Modular Python core package
│   ├── __init__.py
│   ├── data_processing.py              # Data loading, missing value imputation, aggregations
│   ├── predictive_maintenance.py       # Task 2: 24h failure target, rolling features, XGBoost model
│   ├── energy_forecasting.py           # Task 3: Building energy lag features & 24h forecaster
│   ├── anomaly_detection.py            # Task 4: Isolation Forest & Z-Score thresholding framework
│   └── connectivity_analysis.py        # Task 5: NetworkX graph analysis & failure propagation
├── run_pipeline.py                     # Master execution pipeline script
├── notebooks/                          # 5 Executable Jupyter Notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Predictive_Maintenance.ipynb
│   ├── 03_Energy_Forecasting.ipynb
│   ├── 04_Anomaly_Detection.ipynb
│   └── 05_Asset_Connectivity.ipynb
├── dashboard/                          # Option A Bonus: Interactive Streamlit Dashboard
│   └── app.py
├── api/                                # Bonus: FastAPI Model Deployment REST API
│   ├── main.py                         # FastAPI server (POST /predict_failure)
│   └── test_api.py                     # API test suite
├── models/                             # Saved ML model artifacts (.joblib)
│   ├── predictive_maintenance_model.joblib
│   └── energy_forecasting_model.joblib
├── reports/                            # Deliverable 4: Executive Report
│   ├── generate_pdf_report.py          # ReportLab PDF generator script
│   ├── Nectar_Data_Science_Challenge_Report.pdf  # 5-Page Executive PDF Report
│   └── Nectar_Data_Science_Challenge_Report.md   # Markdown version of report
└── README.md                           # Documentation
```

---

## ⚙️ Quick Start & Setup Instructions

### 1. Prerequisites & Environment Setup
Python 3.10+ is recommended. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn networkx joblib fastapi uvicorn streamlit reportlab
```

### 2. Generate Synthetic Dataset (If required)
```bash
python generate_data.py
```
This generates `telemetry.csv`, `asset_metadata.csv`, and `asset_connectivity.csv` inside `data/`.

### 3. Run Full Pipeline
```bash
python run_pipeline.py
```
Executes feature engineering, model training, metric calculation, anomaly detection, network graph audit, and saves evaluation summary to `output/summary_results.json`.

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Run FastAPI Deployment Endpoint
```bash
cd api
uvicorn main:app --reload --port 8000
```
Test the API endpoint using:
```bash
python api/test_api.py
```

---

## 🏗️ Architecture Overview

The solution follows a clean modular data science pipeline architecture:

```
[ Raw IoT Sensors Stream ] ---> [ Data Processing & Feature Store ]
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
[ Task 2: Predictive Maintenance ] [ Task 3: Energy Forecaster ] [ Task 4: Anomaly Detection ]
       |                              |                              |
 (XGBoost Classifier)           (XGBoost Regressor)            (Isolation Forest + Z-Score)
       |                              |                              |
       +------------------------------+------------------------------+
                                      |
                        [ Task 5: NetworkX Graph ]
                         (Failure Propagation Tree)
                                      |
                      +---------------+---------------+
                      |                               |
          [ Streamlit Dashboard ]           [ FastAPI Server ]
```

---

## 💡 Key Assumptions & Design Decisions

1. **Synthetic Dataset Physics**:
   - Equipment failure is preceded by a **36-hour degradation window** characterized by increasing vibration standard deviation ($\sigma_{\text{vib}}$) and thermal spikes.
   - Energy consumption follows a diurnal cycle dictated by occupancy (8 AM - 6 PM peak) and ambient temperature.

2. **Predictive Maintenance Target Strategy**:
   - Formulated as a **24-hour lookahead binary classification task** (`fault_next_24h`).
   - Active failure periods (`fault_flag == 1`) are excluded from feature inputs during training to prevent target leakage.

3. **Multi-Asset Graph Modeling**:
   - Represented as a directed graph $G = (V, E)$ using `NetworkX`.
   - Node attributes store asset type, building, and capacity; edge attributes store relationship type (`Supplies`, `Controls`, `Monitors`).
   - Downstream impact analysis utilizes **Breadth-First Search (BFS)** traversal to calculate propagation reachability and distance.

---

## 📊 Summary of Model Evaluation Results

- **Predictive Maintenance (Task 2)**:
  - **Precision:** 83.3% | **Recall:** 100.0% | **F1 Score:** 90.9% | **ROC-AUC:** 1.000
- **Energy Forecasting (Task 3)**:
  - **MAE:** 22.62 kWh | **RMSE:** 42.10 kWh | **MAPE:** 1.16%
- **Anomaly Detection (Task 4)**:
  - Flagged **3,423 anomaly events** categorizing power surges, vibration spikes, and thermal anomalies.
- **Multi-Asset Connectivity (Task 5)**:
  - Uncovered 1 orphan node (`Sensor_Orphan_99`), 2 duplicate edges, and 13 invalid parent-child mappings.
  - Simulated failure of `Chiller_Bldg_A1_01` affecting 5 downstream dependent assets.
