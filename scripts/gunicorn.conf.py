import os
from config.settings import BASE_DIR

# 绑定监听地址、端口
bind = "0.0.0.0:5000"
# 工作进程数：通用经验公式 CPU核心数*2+1；本地调试可3，服务器根据CPU调整
workers = 1
# 同步工作模式，兼容性最好，无需额外安装协程库
worker_class = "sync"
# 单进程最大并发连接数
worker_connections = 1000

# ⭐关键优化：预加载应用，主进程先加载模型，子进程共享，避免多进程重复加载XGB模型，大幅降低内存占用
preload_app = False

# 超时管控；模型推理存在耗时，适当放宽超时
timeout = 120
graceful_timeout = 30
keepalive = 5

# 日志路径（统一读取settings全局日志目录）
log_dir = os.path.join(BASE_DIR, "logs")
# 自动创建logs文件夹，防止首次启动目录不存在报错
os.makedirs(log_dir, exist_ok=True)

accesslog = os.path.join(log_dir, "gunicorn_access.log")
errorlog = os.path.join(log_dir, "gunicorn_error.log")
loglevel = "info"

# PID文件，用于停止、优雅重启服务
pidfile = os.path.join(log_dir, "gunicorn.pid")

# 限制单个工作进程最大处理请求数，防止内存缓慢泄漏
max_requests = 1000
max_requests_jitter = 100