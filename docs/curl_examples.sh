
echo "1. 健康检查"

curl http://127.0.0.1:5000/health


echo "\n2. 单条预测"

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d @docs/single_test.json


echo "\n3. 批量预测"

curl -X POST http://127.0.0.1:5000/batch_predict \
-H "Content-Type: application/json" \
-d @docs/batch_test.json