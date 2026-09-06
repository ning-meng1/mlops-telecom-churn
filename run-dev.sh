#!/bin/bash

echo "启动开发环境"

docker rm -f mlops-dev-api 2>/dev/null

docker run -d \
-p 5001:5000 \
-v "$(pwd)/models:/app/models" \
-v "$(pwd)/logs:/app/logs" \
--name mlops-dev-api \
-e ENV=dev \
mlops-lite:day31

echo "开发服务启动完成"