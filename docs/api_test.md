# API接口测试记录

## 1. 健康检查接口

请求：
GET
curl http://127.0.0.1:5000/health

返回：

{
"code":0,
"msg":"service ok",
"data":{
    "service_status":"running",
    "model_version":"v2",
    "model_loaded":true
}
}
## 2. 单条预测接口

请求：
POST

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d @docs/single_test.json
返回：
{
    "code":0,
    "msg":"success",
    "data":{
        "pred_class":0,
        "confidence":0.1835,
        "label_desc":"0=不流失，1=流失",
        "model_version":"v2"
    }
}

## 3. 批量预测接口
请求：
POST

curl -X POST http://127.0.0.1:5000/batch_predict \
-H "Content-Type: application/json" \
-d @docs/batch_test.json
返回：
{
    "code":0,
    "msg":"success",
    "data":{
        "pred_label":[0,1],
        "pred_proba":[0.1835,0.7501]
    }
}

---
