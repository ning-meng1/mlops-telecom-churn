import pytest
from unittest.mock import patch
from src.api_app import app

client = app.test_client()

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["code"] == 0
    assert "service ok" in data["msg"]

@patch("src.api_app.is_model_loaded", return_value=True)
def test_predict_normal(mock_is_loaded):
    payload = {
        "features": {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 92.5,
            "TotalCharges": 277.5
        }
    }
    with patch("src.api_app.predict_single", return_value={"pred_class": 0, "confidence": 0.8, "model_version": "v2"}):
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        res = resp.get_json()
        assert res["code"] == 0
        assert "success" in res["msg"]
        assert "pred_class" in res["data"]

@patch("src.api_app.is_model_loaded", return_value=True)
def test_batch_predict(mock_is_loaded):
    payload = {
        "features_list": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.2,
                "TotalCharges": 1804.8
            }
        ]
    }
    mock_ret = {"pred_label": [0], "pred_proba": [0.7]}
    with patch("src.api_app.predict_batch_service", return_value=mock_ret):
        resp = client.post("/batch_predict", json=payload)
        assert resp.status_code == 200
        res = resp.get_json()
        assert res["code"] == 0
        assert len(res["data"]["pred_label"]) == 1

# ========== 修复：这个测试是【缺少顶层features_list】，并且加patch隔离模型状态 ==========
@patch("src.api_app.is_model_loaded", return_value=True)
def test_missing_field(mock_is_loaded):
    # 重点：payload里面根本没有 features_list
    payload = {}
    resp = client.post("/batch_predict", json=payload)
    res = resp.get_json()
    assert res["code"] == 400
    assert "缺少features_list字段" in res["msg"]

# ========== 这个测试：有features_list，但是里面字段非法，同样加patch ==========
@patch("src.api_app.is_model_loaded", return_value=True)
def test_wrong_type(mock_is_loaded):
    payload = {
        "features_list": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": "abc",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.2,
                "TotalCharges": 1804.8
            }
        ]
    }
    resp = client.post("/batch_predict", json=payload)
    res = resp.get_json()
    assert res["code"] == 400

def test_internal_exception():
    resp = client.get("/test_error")
    res = resp.get_json()
    assert res["code"] == 500
    assert "服务器内部异常" in res["msg"]

def test_health_after_exception():
    resp = client.get("/health")
    res = resp.get_json()
    assert res["code"] == 0
