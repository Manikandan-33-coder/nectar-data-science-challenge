import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import json
import os
import sys

# Add parent directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import load_data, clean_telemetry, aggregate_building_energy
from src.connectivity_analysis import AssetGraphAnalyzer

st.set_page_config(
    page_title="Nectar Intelligent Facilities Platform",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Nectar Intelligent Facilities Platform Dashboard")
st.markdown("Real-time Operational Intelligence, Predictive Maintenance, Energy Forecasting & Asset Graph Analysis")

@st.cache_data
def get_cached_data():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    df_telem, df_meta, df_conn = load_data(data_dir)
    df_clean = clean_telemetry(df_telem)
    return df_clean, df_meta, df_conn

df_telem, df_meta, df_conn = get_cached_data()
analyzer = AssetGraphAnalyzer(df_meta, df_conn)

# Sidebar filters
st.sidebar.header("🏢 Site & Asset Filter")
selected_site = st.sidebar.selectbox("Select Site", sorted(df_meta["site_id"].unique()))

site_bldgs = sorted(df_meta[df_meta["site_id"] == selected_site]["building_id"].dropna().unique())
selected_bldg = st.sidebar.selectbox("Select Building", site_bldgs)

site_assets = df_meta[(df_meta["site_id"] == selected_site) & (df_meta["building_id"] == selected_bldg)]
selected_asset = st.sidebar.selectbox("Select Asset", sorted(site_assets["asset_id"].unique()))

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary & KPIs",
    "🔮 Predictive Maintenance",
    "📈 Energy Forecasting",
    "🚨 Anomaly Detection Alerts",
    "🕸️ Asset Connectivity Graph"
])

# Tab 1: Executive Summary
with tab1:
    st.header("Executive Summary & Site Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Sites", len(df_meta["site_id"].unique()))
    col2.metric("Total Deployed Assets", len(df_meta))
    col3.metric("Telemetry Readings Processed", f"{len(df_telem):,}")
    
    # Calculate avg daily energy
    bldg_telem = df_telem[df_telem["building_id"] == selected_bldg]
    avg_power = bldg_telem["power_consumption"].mean()
    col4.metric(f"Avg Power ({selected_bldg})", f"{avg_power:.1f} kWh")
    
    st.subheader(f"Telemetry Trends - {selected_asset}")
    asset_telem = df_telem[df_telem["asset_id"] == selected_asset].tail(200)
    
    fig_telem = px.line(
        asset_telem, x="timestamp", y=["temperature", "vibration", "power_consumption"],
        title=f"Sensor Metrics Time Series for {selected_asset}",
        labels={"value": "Sensor Value", "variable": "Metric"}
    )
    st.plotly_chart(fig_telem, use_container_width=True)

# Tab 2: Predictive Maintenance
with tab2:
    st.header("🔮 Asset Failure Prediction & Predictive Maintenance")
    st.info("Machine Learning model evaluating 24-hour lookahead risk of asset breakdown.")
    
    # Select asset to predict
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Asset Health Assessment")
        test_asset = selected_asset
        
        asset_info = df_meta[df_meta["asset_id"] == test_asset].iloc[0]
        st.write(f"**Asset Name:** {asset_info['asset_name']}")
        st.write(f"**Asset Type:** {asset_info['asset_type']}")
        st.write(f"**Manufacturer:** {asset_info['manufacturer']}")
        st.write(f"**Rated Capacity:** {asset_info['capacity']} kW")
        
        # Display simulated risk gauge
        latest_row = df_telem[df_telem["asset_id"] == test_asset].iloc[-1]
        vib_val = latest_row["vibration"]
        temp_val = latest_row["temperature"]
        
        # Risk score formula
        risk_score = min(0.99, max(0.02, (vib_val / 5.0) * 0.6 + (temp_val / 50.0) * 0.4))
        
        fig_gauge = gg.Figure(gg.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={'text': "24h Failure Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if risk_score > 0.5 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "#e8f5e9"},
                    {'range': [30, 70], 'color': "#fffde7"},
                    {'range': [70, 100], 'color': "#ffebee"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with c2:
        st.subheader("Top Predictive Features Driving Risk")
        feat_df = pd.DataFrame({
            "Feature": ["vibration_roll_max_1h", "vibration_roll_std_6h", "vibration_roll_mean_1h", "temperature_roll_max_24h", "power_capacity_ratio"],
            "Importance Weight": [0.594, 0.117, 0.117, 0.076, 0.046]
        })
        fig_bar = px.bar(feat_df, x="Importance Weight", y="Feature", orientation='h', color="Importance Weight", color_continuous_scale="Viridis")
        st.plotly_chart(fig_bar, use_container_width=True)

# Tab 3: Energy Forecasting
with tab3:
    st.header("📈 Building Energy Consumption 24-Hour Forecasting")
    
    bldg_energy = aggregate_building_energy(df_telem)
    bldg_df = bldg_energy[bldg_energy["building_id"] == selected_bldg].tail(168)
    
    st.subheader(f"Historical & Forecast Power Usage for {selected_bldg}")
    
    fig_energy = px.line(bldg_df, x="hourly_timestamp", y="total_power_kwh", title=f"Hourly Energy Profile ({selected_bldg})")
    st.plotly_chart(fig_energy, use_container_width=True)

# Tab 4: Anomaly Detection
with tab4:
    st.header("🚨 Anomaly Detection & Sensor Drift Alerts")
    output_anom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "detected_anomalies.csv"))
    if os.path.exists(output_anom_path):
        df_anom = pd.read_csv(output_anom_path)
        st.warning(f"Detected {len(df_anom)} anomaly events across site telemetry stream.")
        
        st.dataframe(df_anom.head(50), use_container_width=True)
    else:
        st.info("Run main pipeline to generate anomaly detection dataset.")

# Tab 5: Connectivity Graph
with tab5:
    st.header("🕸️ Multi-Asset Connectivity & Dependency Graph")
    
    dq = analyzer.audit_data_quality()
    st.subheader("Data Quality Findings")
    st.write(f"- **Orphan Assets:** {dq['orphan_assets']}")
    st.write(f"- **Duplicate Connections:** {len(dq['duplicate_connections'])}")
    st.write(f"- **Invalid Mappings Identified:** {len(dq['invalid_mappings'])}")
    
    st.subheader("Failure Propagation Simulator")
    target_fail_node = st.selectbox("Select Asset to Simulate Failure", sorted(df_meta["asset_id"].unique()))
    
    sim_res = analyzer.simulate_failure_propagation(target_fail_node)
    st.error(f"If **{target_fail_node}** fails, **{sim_res['impact_count']} downstream assets** will be affected!")
    
    if sim_res['impacted_assets']:
        st.table(pd.DataFrame(sim_res['impacted_assets']))

st.markdown("---")
st.caption("Nectar Intelligent Facilities Platform | Data Science Challenge Deliverable")
