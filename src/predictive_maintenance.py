import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import joblib
import os

def engineer_predictive_maintenance_features(df_telem, df_meta):
    """
    Build predictive maintenance dataset with 24-hour lookahead failure target and rolling sensor features.
    """
    df = df_telem.sort_values(["asset_id", "timestamp"]).copy()
    
    # Merge asset metadata
    df = df.merge(df_meta[["asset_id", "asset_type", "capacity"]], on="asset_id", how="left")
    
    # Create 24h lookahead target: Will fault_flag = 1 in next 24 hours (96 periods of 15 min)?
    # We shift rolling max of fault_flag backward by 96 steps
    df["fault_next_24h"] = df.groupby("asset_id")["fault_flag"].transform(
        lambda x: x.iloc[::-1].rolling(window=96, min_periods=1).max().iloc[::-1]
    )
    
    # Fill target NaN with 0
    df["fault_next_24h"] = df["fault_next_24h"].fillna(0).astype(int)
    
    # Rolling features for key telemetry metrics
    feature_cols = []
    sensor_cols = ["temperature", "vibration", "power_consumption", "pressure"]
    
    for col in sensor_cols:
        # 1-hour (4 periods) rolling stats
        df[f"{col}_roll_mean_1h"] = df.groupby("asset_id")[col].transform(lambda x: x.rolling(4, min_periods=1).mean())
        df[f"{col}_roll_max_1h"] = df.groupby("asset_id")[col].transform(lambda x: x.rolling(4, min_periods=1).max())
        
        # 6-hour (24 periods) rolling stats
        df[f"{col}_roll_mean_6h"] = df.groupby("asset_id")[col].transform(lambda x: x.rolling(24, min_periods=1).mean())
        df[f"{col}_roll_std_6h"] = df.groupby("asset_id")[col].transform(lambda x: x.rolling(24, min_periods=1).std()).fillna(0)
        
        # 24-hour (96 periods) rolling stats & deltas
        df[f"{col}_roll_max_24h"] = df.groupby("asset_id")[col].transform(lambda x: x.rolling(96, min_periods=1).max())
        df[f"{col}_delta_24h"] = df[col] - df.groupby("asset_id")[col].shift(96).fillna(df[col])
        
        feature_cols.extend([
            f"{col}_roll_mean_1h", f"{col}_roll_max_1h",
            f"{col}_roll_mean_6h", f"{col}_roll_std_6h",
            f"{col}_roll_max_24h", f"{col}_delta_24h"
        ])
        
    # Additional domain features
    df["power_capacity_ratio"] = df["power_consumption"] / (df["capacity"] + 1e-5)
    feature_cols.extend(sensor_cols + ["occupancy_count", "power_capacity_ratio"])
    
    # One-hot encode operating mode & asset_type
    df_encoded = pd.get_dummies(df, columns=["operating_mode", "asset_type"], drop_first=False)
    encoded_cols = [c for c in df_encoded.columns if c.startswith("operating_mode_") or c.startswith("asset_type_")]
    feature_cols.extend(encoded_cols)
    
    return df_encoded, feature_cols

def train_predictive_maintenance_model(df_features, feature_cols, model_path=None):
    """
    Train and evaluate Predictive Maintenance XGBoost Classifier using temporal split.
    """
    # Exclude active fault period from training features to avoid leakage, focus on predicting before fault
    df_clean = df_features[df_features["fault_flag"] == 0].copy()
    
    # Temporal train/test split (80% train, 20% test)
    split_idx = int(len(df_clean) * 0.8)
    df_clean = df_clean.sort_values("timestamp")
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df["fault_next_24h"]
    X_test, y_test = test_df[feature_cols], test_df["fault_next_24h"]
    
    # Handle class imbalance using scale_pos_weight
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        print(f"Saved Predictive Maintenance model to {model_path}")
        
    return model, metrics, feature_importance, (X_test, y_test, y_proba)
