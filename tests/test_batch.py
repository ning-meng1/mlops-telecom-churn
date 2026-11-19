import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unittest.mock import patch
from src.api_app import app


@patch("src.api_app.predict_batch_service")
@patch("src.service.is_model_loaded")
@patch("src.api_app.is_model_loaded")
def test_batch_predict(mock_api_is_loaded, mock_svc_is_loaded, mock_predict_batch):
    mock_api_is_loaded.return_value = True
    mock_svc_is_loaded.return_value = True
    # 模拟批量service返回，2条样本
    mock_predict_batch.return_value = [
        {"churn_prob":0.11},
        {"churn_prob":0.88}
    ]

    payload = {
        "features_list":[
            {
                "gender":"Female",
                "SeniorCitizen":0,
                "Partner":"Yes",
                "Dependents":"No",
                "tenure":24,
                "PhoneService":"Yes",
                "MultipleLines":"No",
                "InternetService":"DSL",
                "OnlineSecurity":"No",
                "OnlineBackup":"Yes",
                "DeviceProtection":"No",
                "TechSupport":"No",
                "StreamingTV":"Yes",
                "StreamingMovies":"Yes",
                "Contract":"Month-to-month",
                "PaperlessBilling":"Yes",
                "PaymentMethod":"Electronic check",
                "MonthlyCharges":75.2,
                "TotalCharges":1804.8
            },
            {
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
        ]
    }
    client = app.test_client()
    resp = client.post("/batch_predict", json=payload)

    assert resp.status_code == 200
    res_data = resp.get_json()
    assert res_data["code"] == 0
    assert len(res_data["data"]) == len(payload["features_list"])
