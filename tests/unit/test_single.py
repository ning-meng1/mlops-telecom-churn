import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unittest.mock import patch
from src.api_app import app


@patch("src.api_app.predict_single")
@patch("src.service.is_model_loaded")
@patch("src.api_app.is_model_loaded")
def test_single_predict(mock_api_is_loaded, mock_svc_is_loaded, mock_predict_single):
    mock_api_is_loaded.return_value = True
    mock_svc_is_loaded.return_value = True
    # 模拟service.predict_single返回结果
    mock_predict_single.return_value = {"churn_prob": 0.88}

    payload = {
        "features":{
            "gender":"Male",
            "SeniorCitizen":1,
            "Partner":"No",
            "Dependents":"No",
            "tenure":3,
            "PhoneService":"Yes",
            "MultipleLines":"Yes",
            "InternetService":"Fiber optic",
            "OnlineSecurity":"No",
            "OnlineBackup":"No",
            "DeviceProtection":"No",
            "TechSupport":"No",
            "StreamingTV":"Yes",
            "StreamingMovies":"Yes",
            "Contract":"Month-to-month",
            "PaperlessBilling":"Yes",
            "PaymentMethod":"Electronic check",
            "MonthlyCharges":92.5,
            "TotalCharges":277.5
        }
    }
    client = app.test_client()
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    res_data = resp.get_json()
    assert res_data["code"] == 0
    assert "churn_prob" in res_data["data"]
