import json
import requests


url = "http://127.0.0.1:5000/batch_predict"

# 批量测试数据路径
path = r"D:\mlops-lite\docs\batch_test.json"


# 读取json
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)


# 发送POST请求
resp = requests.post(
    url,
    json=payload
)


# 打印结果
print("HTTP状态码:", resp.status_code)

print("响应结果:")
print(resp.json())