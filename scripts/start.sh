#!/bin/bash
set -e

echo "当前运行环境: ${ENV}"
export PYTHONPATH=/app

if [ "${ENV}" = "prod" ]; then
    echo "使用Gunicorn生产模式启动"
    exec gunicorn -c scripts/gunicorn.conf.py src.api_app:app
else
    echo "使用Flask开发模式启动"
    exec python src/api_app.py
fi