import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import (
    DATA_PATH,
    DEFAULT_SEED,
    TEST_SIZE,
    NUMERIC_COLS,
    BINARY_COLS,
    MULTI_CAT_COLS,
    DROP_COLS,
    TARGET_COL
)
from src.logger import logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"数据处理模块全局随机种子已设置为：{seed}")


def load_raw_data():
    logger.info(f"开始读取原始数据集，文件路径：{DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"原始数据加载完成，数据集shape={df.shape}")
    return df


def clean_total_charges(df):
    df = df.copy()
    logger.info("执行TotalCharges字段清洗，处理空字符串与缺失值")
    # 电信数据集特殊处理：空字符串转为缺失值
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def build_business_features(df):
    df = df.copy()
    logger.info("开始构造衍生业务特征：avg_monthly_charges")
    # 构造平均月消费特征，tenure=0避免除零
    df["avg_monthly_charges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].replace([np.inf, -np.inf], np.nan)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].fillna(df["MonthlyCharges"])
    return df


def encode_binary_features(df):
    df = df.copy()
    logger.info("开始处理二元类别特征编码")

    # --------修复：只有标签列存在的时候，才转换标签，推理时跳过----------
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        logger.info("目标标签Churn完成0/1数值编码")

    for col in BINARY_COLS:
        if col not in df.columns:
            logger.warning(f"二元特征列 {col} 不存在于当前数据集，跳过")
            continue
        if col == "gender":
            df[col] = df[col].map({"Female": 1, "Male": 0})
        else:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_multi_category(df):
    df = df.copy()
    logger.info(f"对多分类特征执行OneHot编码，待编码字段：{MULTI_CAT_COLS}")
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)
    logger.info(f"多分类编码完成，当前总特征数量：{df.shape[1]}")
    return df


def raw_data_transform(df: pd.DataFrame):
    """
    推理接口专用！输入【原始业务DataFrame】，和训练执行完全一样清洗、衍生、编码
    ❗不读取csv、❗不做标准化、❗不分割数据集
    """
    logger.info("进入推理数据预处理流水线，执行与训练一致的数据变换逻辑")
    df = df.copy()
    df = clean_total_charges(df)
    df = build_business_features(df)
    df = encode_binary_features(df)
    df = encode_multi_category(df)

    # 丢弃不需要的原始列
    drop_cols = DROP_COLS.copy()
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    logger.info(f"推理数据预处理结束，处理后特征维度：{df.shape[1]}")
    return df


def standardize_features(X):
    X = X.copy()
    logger.info("开始对数值特征执行标准化处理")
    scaler = StandardScaler()

    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

    # 仅使用训练阶段统计量进行标准化，预测阶段复用scaler
    scaled_array = scaler.fit_transform(X[NUMERIC_COLS])
    scaled_df = pd.DataFrame(scaled_array, columns=NUMERIC_COLS, index=X.index)
    X = pd.concat([X.drop(columns=NUMERIC_COLS), scaled_df], axis=1)
    logger.info("数值特征标准化完成")
    return X, scaler


def split_dataset(X, y, seed):
    logger.info(f"开始划分训练集/测试集，测试集比例={TEST_SIZE}，分层抽样开启")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y
    )
    logger.info(f"数据集划分完成 | 训练集:{X_train.shape[0]}条 | 测试集:{X_test.shape[0]}条")
    return X_train, X_test, y_train, y_test


def get_full_dataset(fix_seed=DEFAULT_SEED):
    logger.info("===== 启动完整数据集加载&预处理流水线 =====")
    set_seed(fix_seed)

    df = load_raw_data()
    df = clean_total_charges(df)
    df = build_business_features(df)
    df = encode_binary_features(df)
    df = encode_multi_category(df)

    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS+[TARGET_COL])
    logger.info(f"分离特征与标签，原始特征矩阵shape={X.shape}")

    X_scaled, scaler = standardize_features(X)
    X_train, X_test, y_train, y_test = split_dataset(X_scaled, y, fix_seed)
    feature_cols = X_scaled.columns.tolist()
    logger.info(f"完整数据处理流程结束，最终特征列表长度：{len(feature_cols)}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


if __name__ == "__main__":
    logger.info("执行data_process自测入口")
    X_train, X_test, y_train, y_test, scaler, features = get_full_dataset()
    logger.info(f"【自测输出】训练集形状：{X_train.shape}")
    logger.info(f"【自测输出】测试集形状：{X_test.shape}")