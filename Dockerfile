FROM python:3.13-slim

WORKDIR /app

# 设置python环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 复制依赖清单
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
# 业务代码、脚本
COPY config ./config
COPY scripts ./scripts
COPY src ./src
COPY assets ./assets
COPY data ./data
COPY models ./models

# 赋予启动脚本执行权限
RUN chmod +x ./scripts/*.sh

EXPOSE 5000
# 默认启动脚本
ENTRYPOINT ["./scripts/start.sh"]