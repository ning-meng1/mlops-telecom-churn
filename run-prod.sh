#!/bin/bash

echo "启动生产环境"

docker rm -f mlops-prod-api 2>/dev/null

docker run -d \
-p 5000:5000 \
-v "$(pwd)/models:/app/models" \
-v "$(pwd)/logs:/app/logs" \
--name mlops-prod-api \
--restart always \
-e ENV=prod \
mlops-lite:day31

echo "生产服务启动完成"