import pandas as pd
import numpy as np
from unittest.mock import patch
from src.predict import predict

@patch("src.predict.raw_data_transform")
@patch("src.predict.load_model_assets")
def test_prediction(mock_load_assets, mock_raw_transform):
    print("======开始模型预测功能测试======")

    fake_raw_sample = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 1687.25
    }])

    mock_model = type("MockModel", (), {
        "predict_proba": lambda x, *args, **kwargs: np.array([[0.72, 0.28]]),
        "predict": lambda x, *args, **kwargs: np.array([1])
    })()

    mock_scaler = type("MockScaler", (), {
        "transform": lambda x, *args, **kwargs: x
    })()

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges']
    mock_feature_cols = numeric_cols
    mock_median = {}

    mock_load_assets.return_value = (mock_model, mock_scaler, mock_feature_cols, None, mock_median)

    transformed_df = pd.DataFrame({
        "tenure": [24],
        "MonthlyCharges": [70.35],
        "TotalCharges": [1687.25],
        "avg_monthly_charges": [70.3]
    })
    mock_raw_transform.return_value = transformed_df

    result = predict(fake_raw_sample)
    print("预测结果：", result)

    # ✅ 修改成和真实返回key一致
    assert "pred_class" in result
    assert "confidence" in result
    assert isinstance(result["pred_class"], (int, np.integer))
    assert 0.0 <= result["confidence"] <= 1.0

    print("✅ 模型预测测试全部通过！")

if __name__ == "__main__":
    test_prediction()
