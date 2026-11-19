
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.api_app import app


def test_single_predict():

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
    assert "churn_prob" in res_data
