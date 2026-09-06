import json, requests
url = "http://127.0.0.1:5000/predict"
# 完整绝对路径
path = r"D:\mlops-lite\docs\single_test.json"
with open(path,"r",encoding="utf-8") as f:
    payload = json.load(f)

resp = requests.post(url, json=payload)
print(resp.status_code)
print(resp.json())