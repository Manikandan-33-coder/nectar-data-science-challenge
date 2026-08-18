import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import os
import joblib

def engineer_energy_forecasting_features(df_bldg_energy):
    """
    Build time-series lag and calendar features for building energy forecasting.
    """
    df = df_bldg_energy.sort_values(["building_id", "hourly_timestamp"]).copy()
    
    # Calendar features
    df["hour"] = df["hourly_timestamp"].dt.hour
    df["dayofweek"] = df["hourly_timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Lag features
    df["lag_1h"] = df.groupby("building_id")["total_power_kwh"].shift(1)
    df["lag_24h"] = df.groupby("building_id")["total_power_kwh"].shift(24)
    df["lag_168h"] = df.groupby("building_id")["total_power_kwh"].shift(168).fillna(df["lag_24h"])
    
    # Rolling stats
    df["roll_mean_24h"] = df.groupby("building_id")["total_power_kwh"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df["roll_std_24h"] = df.groupby("building_id")["total_power_kwh"].transform(lambda x: x.rolling(24, min_periods=1).std()).fillna(0)
    
    # Target: total_power_kwh
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = [
        "hour", "dayofweek", "is_weekend", "temperature", "humidity", "occupancy_count",
        "lag_1h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_std_24h"
    ]
    
    return df, feature_cols

def train_energy_forecaster(df_features, feature_cols, target_col="total_power_kwh", model_path=None):
    """
    Train and evaluate XGBoost Energy Forecaster.
    """
    # Temporal split: last 24*7 = 168 hours (1 week) for test set
    split_time = df_features["hourly_timestamp"].max() - pd.Timedelta(hours=168)
    
    train_df = df_features[df_features["hourly_timestamp"] <= split_time]
    test_df = df_features[df_features["hourly_timestamp"] > split_time]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100)
    
    metrics = {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE": round(mape, 2)
    }
    
    test_results = test_df.copy()
    test_results["predicted_power_kwh"] = y_pred
    
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        print(f"Saved Energy Forecaster model to {model_path}")
        
    return model, metrics, test_results
