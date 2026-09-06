#!/bin/bash

MODE=$1

if [ "$MODE" = "dev" ]; then
    echo "启动开发环境"
    bash ./run-dev.sh

elif [ "$MODE" = "prod" ]; then
    echo "启动生产环境"
    bash ./run-prod.sh

else
    echo "用法：bash scripts/run.sh [dev|prod]"
    exit 1
fi