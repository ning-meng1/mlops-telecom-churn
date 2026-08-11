import logging
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志文件路径
LOG_FILE = LOG_DIR / "mlops_service.log"

def get_logger(name="mlops"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 文件输出
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    # 控制台输出
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = get_logger()