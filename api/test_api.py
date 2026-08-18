from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_prediction_endpoint():
    print("=== Testing FastAPI /predict_failure Endpoint ===")
    
    # Test Normal Asset
    normal_payload = {
        "asset_id": "Chiller_Bldg_A1_01",
        "temperature": 8.5,
        "vibration": 0.95,
        "pressure": 300.0,
        "power_consumption": 180.0,
        "occupancy_count": 150,
        "capacity": 450.0,
        "operating_mode": "Cooling",
        "asset_type": "Chiller"
    }
    res_normal = client.post("/predict_failure", json=normal_payload)
    print(f"Normal Asset Test Status: {res_normal.status_code}")
    print("Normal Response:", res_normal.json())
    
    # Test Degraded / Pre-Failure Asset
    degraded_payload = {
        "asset_id": "Chiller_Bldg_A1_01",
        "temperature": 32.0,
        "vibration": 7.8,
        "pressure": 150.0,
        "power_consumption": 420.0,
        "occupancy_count": 250,
        "capacity": 450.0,
        "operating_mode": "Cooling",
        "asset_type": "Chiller",
        "vibration_roll_max_1h": 8.5,
        "vibration_roll_std_6h": 2.4
    }
    res_deg = client.post("/predict_failure", json=degraded_payload)
    print(f"\nDegraded Asset Test Status: {res_deg.status_code}")
    print("Degraded Response:", res_deg.json())

if __name__ == "__main__":
    test_prediction_endpoint()
