import requests

url = "http://127.0.0.1:5000/batch_predict"

# 批量接口 key 是 features_list，值是样本字典组成的列表
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
            "Contract":"Month‑to‑month",
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
            "Contract":"Month‑to‑month",
            "PaperlessBilling":"Yes",
            "PaymentMethod":"Electronic check",
            "MonthlyCharges":92.5,
            "TotalCharges":277.5
        }
    ]
}

resp = requests.post(url, json=payload)
print(resp.status_code)
print(resp.json())

# ---------------------------------------边缘情况检测------------------------------------------------------------------------
# # Flask 服务地址
# url = "http://127.0.0.1:5000/batch_predict"
#
#
# # ============================================================
# # 测试1：features_list 传空列表
# # 预期：HTTP 200，但业务 code 应该是 400
# # ============================================================
#
# print("=" * 60)
# print("测试1：features_list 为空列表")
# print("=" * 60)
#
# payload_empty = {
#     "features_list": []
# }
#
# resp = requests.post(url, json=payload_empty)
#
# print("HTTP状态码：", resp.status_code)
# print("响应结果：", resp.json())
#
#
# # ============================================================
# # 测试2：features_list 类型错误
# # 把列表改成字典
# # 预期：HTTP 200，但业务 code 应该是 400
# # ============================================================
#
# print("=" * 60)
# print("测试2：features_list 类型错误")
# print("=" * 60)
#
# payload_wrong_type = {
#     "features_list": {
#         "gender": "Male",
#         "SeniorCitizen": 1
#     }
# }
#
# resp = requests.post(url, json=payload_wrong_type)
#
# print("HTTP状态码：", resp.status_code)
# print("响应结果：", resp.json())
#
#
# # ============================================================
# # 测试3：features_list 中的元素不是字典
# # 例如传入字符串、数字
# # 预期：应该返回 HTTP 200 + code 400
# # 而不是 500
# # ============================================================
#
# print("=" * 60)
# print("测试3：features_list 元素不是字典")
# print("=" * 60)
#
# payload_invalid_items = {
#     "features_list": [
#         "hello",
#         123,
#         None
#     ]
# }
#
# resp = requests.post(url, json=payload_invalid_items)
#
# print("HTTP状态码：", resp.status_code)
# print("响应结果：", resp.json())

# ---------------------------------------------------------健康接口检测------------------------------------------------------------------------------------------------
url="http://127.0.0.1:5000/health"


resp=requests.get(url)


print(resp.status_code)
print(resp.json())

# ============================================================
# Day24：故意触发预测异常
# ============================================================
#
# print("=" * 60)
# print("测试4：故意触发预测异常")
# print("=" * 60)
#
# url = "http://127.0.0.1:5000/predict"
#
# payload_error = {
#     "features": {
#         "gender": "Female",
#         "SeniorCitizen": 0,
#         "Partner": "Yes",
#         "Dependents": "No",
#         "tenure": 24,
#         "PhoneService": "Yes",
#         "MultipleLines": "No",
#         "InternetService": "DSL",
#         "OnlineSecurity": "No",
#         "OnlineBackup": "Yes",
#         "DeviceProtection": "No",
#         "TechSupport": "No",
#         "StreamingTV": "Yes",
#         "StreamingMovies": "Yes",
#         "Contract": "Month-to-month",
#         "PaperlessBilling": "Yes",
#         "PaymentMethod": "Electronic check",
#         "MonthlyCharges": 75.2,
#         "TotalCharges": 1804.8
#     }
# }
#
# resp = requests.post(
#     url,
#     json=payload_error
# )
#
# print("HTTP状态码：", resp.status_code)
# print("响应结果：", resp.json())
#
# url="http://127.0.0.1:5000/health"
#
#
# resp=requests.get(url)
#
#
# print(resp.status_code)
# print(resp.json())


