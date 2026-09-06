import os
import warnings

# 开发阶段临时屏蔽FutureWarning，上线后建议移至对应脚本局部生效
warnings.filterwarnings("ignore", category=FutureWarning)

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 全局默认模型版本
DEFAULT_MODEL_VERSION = "v2"

# 基础文件夹路径
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_PATH = os.path.join(BASE_DIR, "logs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# 自动创建项目目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# 数据集文件路径
DATA_PATH = os.path.join(DATA_DIR, "telco_churn.csv")

# 接口服务端口（Flask部署使用）
SERVICE_PORT = 5000

# 全局固定随机种子，保障实验可复现
DEFAULT_SEED = 42
TEST_SIZE = 0.2
# 日志配置
LOG_CONFIG = {
    "log_file": os.path.join(LOG_PATH, "app.log"),
    "log_level": "INFO"
}

# XGBoost基础训练配置
TRAIN_CONFIG = {
    "model_type": "xgboost",
    "random_state": DEFAULT_SEED,
    "test_size": 0.2,
    "eval_metric": "logloss",
    "stratify": True  # 分层抽样，保障分类任务训练/测试集分布一致
}

# XGBoost网格搜索超参范围
PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "n_estimators": [50, 100, 200, 300]
}

# API接口基础配置
API_CONFIG = {
    "version": "v1",
    "prefix": "/api"
}

# ===================== data_process 数据预处理专用配置 =====================
# 原始数值特征
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges']
# 二分类类别特征
BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
# 需要独热编码的多分类特征
MULTI_CAT_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
# 建模前需要剔除的字段
DROP_COLS = ['customerID']
# 预测目标标签
TARGET_COL = "Churn"

# ===================== 版本归档存储配置 =====================
MODEL_ROOT_DIR = MODEL_DIR

def get_version_storage_dir(version_tag: str) -> str:
    """
    根据版本标签生成独立归档文件夹
    :param version_tag: 版本标识，例如 v1 / v2
    :return: 当前版本完整存储目录路径
    """
    version_folder = os.path.join(MODEL_ROOT_DIR, version_tag)
    os.makedirs(version_folder, exist_ok=True)
    return version_folder

# 版本目录内固定文件命名规范
VERSION_MODEL_NAME = "xgb_best_model.pkl"          # 最优训练模型
VERSION_SCALER_NAME = "std_scaler.pkl"             # 特征标准化器
VERSION_FEATURE_TXT = "train_feature_columns.txt"  # 训练特征名列表
VERSION_META_JSON = "train_metadata.json"          # 训练元数据：种子、超参、训练时间
VERSION_EVAL_JSON = "eval_metrics.json"            # 模型评估指标

# ===================== API输入校验配置 =====================


# API预测接口必填字段
# 必须和训练阶段最终特征保持一致

REQUIRED_FEATURES = [

    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"

]


# =====================
# 数值字段范围校验
# =====================

FEATURE_RANGE_RULES = {

    # 老年用户标识，只允许0/1
    "SeniorCitizen": (0, 1),

    # Telco数据集实际业务范围：0~72个月
    "tenure": (0, 72),

    # 月消费金额
    # Telco数据集中最大值约118.75
    # 留出一定业务扩展空间
    "MonthlyCharges": (0, 200),

    # 总消费金额
    # Telco数据集中最大值约8684.8
    # 留出一定业务扩展空间
    "TotalCharges": (0, 20000)
}


# =====================
# 枚举字段合法取值
# =====================

FEATURE_ALLOWED_VALUES = {

    "gender": {
        "Male",
        "Female"
    },

    "Partner": {
        "Yes",
        "No"
    },

    "Dependents": {
        "Yes",
        "No"
    },

    "PhoneService": {
        "Yes",
        "No"
    },

    "MultipleLines": {
        "Yes",
        "No",
        "No phone service"
    },

    "InternetService": {
        "DSL",
        "Fiber optic",
        "No"
    },

    "OnlineSecurity": {
        "Yes",
        "No",
        "No internet service"
    },

    "OnlineBackup": {
        "Yes",
        "No",
        "No internet service"
    },

    "DeviceProtection": {
        "Yes",
        "No",
        "No internet service"
    },

    "TechSupport": {
        "Yes",
        "No",
        "No internet service"
    },

    "StreamingTV": {
        "Yes",
        "No",
        "No internet service"
    },

    "StreamingMovies": {
        "Yes",
        "No",
        "No internet service"
    },

    "Contract": {
        "Month-to-month",
        "One year",
        "Two year"
    },

    "PaperlessBilling": {
        "Yes",
        "No"
    },

    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    }
}

