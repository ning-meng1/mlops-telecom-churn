import os
from config.settings import BASE_DIR

# 绑定监听地址、端口
bind = f"0.0.0.0:{5000}"
# 工作进程数：通用经验公式 CPU核心数*2+1
workers = 3
# 同步工作模式，兼容性最好，无需额外安装协程库
worker_class = "sync"
# 单进程最大并发连接数
worker_connections = 1000

# 超时管控
timeout = 60
graceful_timeout = 30

# 日志路径（统一读取settings全局日志目录）
log_dir = os.path.join(BASE_DIR, "logs")
accesslog = os.path.join(log_dir, "gunicorn_access.log")
errorlog = os.path.join(log_dir, "gunicorn_error.log")
loglevel = "info"

# PID文件，用于停止、优雅重启服务
pidfile = os.path.join(log_dir, "gunicorn.pid")
