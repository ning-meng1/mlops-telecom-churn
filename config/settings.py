import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基础路径配置（按你现有项目结构补充完整）
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")  # 指向具体数据集文件
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/model_v1.pkl")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")  # 补充标准化器路径
LOG_PATH = os.path.join(BASE_DIR, "logs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")  # Day6 可视化图片目录
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# 接口服务端口（第3周Flask使用）
SERVICE_PORT = 5000

# --- 【新增】按 Day5 任务要求补充 ---
# 1. 全局固定随机种子（Day5核心：保证训练可复现）
SEED = 42

# 2. 日志配置
LOG_CONFIG = {
    "log_file": os.path.join(LOG_PATH, "app.log"),
    "log_level": "INFO"
}

# 3. 模型训练配置（消除硬编码，Day5可复现+调参基础）
TRAIN_CONFIG = {
    "model_type": "xgboost",
    "random_state": SEED,  # 直接引用全局SEED，避免写死数字
    "test_size": 0.2,
    "eval_metric": "logloss",
    "stratify": True  # 补充分层抽样，分类任务更稳定
}

# 4. Day5 超参数搜索范围（XGBoost 核心调参参数）
PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],#（学习率）
    "max_depth": [3, 5, 7, 9],#（树的深度）：可以决定模型考虑问题有多细，太细了会钻牛角尖（过拟合），太粗抓不住重点（欠拟合）
    "n_estimators": [50, 100, 200, 300]#（树的数量）：太多了效率低，太少能力不够

}

# 5. API配置
API_CONFIG = {
    "version": "v1",
    "prefix": "/api"
}
