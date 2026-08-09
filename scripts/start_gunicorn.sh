#!/bin/bash
# 电信用户流失预测API Gunicorn生产启动脚本

# 获取脚本所在上级 = 项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_ROOT}" || exit 1

echo "==================== 电信流失预测API服务启动 ===================="
echo "项目根目录：${PROJECT_ROOT}"

# 激活虚拟环境（Linux/WSL/GitBash适用）
source venv/bin/activate
echo "虚拟环境激活完成"

# 启动gunicorn，读取配置文件，入口src.api_app:app
gunicorn -c scripts/gunicorn.conf.py src.api_app:app
