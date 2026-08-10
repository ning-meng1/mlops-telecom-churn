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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成，shape={df.shape}")
    return df


def clean_total_charges(df):
    df = df.copy()
    # 电信数据集特殊处理：空字符串转为缺失值
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def build_business_features(df):
    df = df.copy()
    # 构造平均月消费特征，tenure=0避免除零
    df["avg_monthly_charges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].replace([np.inf, -np.inf], np.nan)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].fillna(df["MonthlyCharges"])
    return df


def encode_binary_features(df):
    df = df.copy()

    # --------修复：只有标签列存在的时候，才转换标签，推理时跳过----------
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        if col == "gender":
            df[col] = df[col].map({"Female": 1, "Male": 0})
        else:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_multi_category(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)
    return df

def raw_data_transform(df: pd.DataFrame):
    """
    推理接口专用！输入【原始业务DataFrame】，和训练执行完全一样清洗、衍生、编码
    ❗不读取csv、❗不做标准化、❗不分割数据集
    """
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
    return df

def standardize_features(X):
    X = X.copy()
    scaler = StandardScaler()

    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

    # 仅使用训练阶段统计量进行标准化，预测阶段复用scaler
    scaled_array = scaler.fit_transform(X[NUMERIC_COLS])
    scaled_df = pd.DataFrame(scaled_array, columns=NUMERIC_COLS, index=X.index)
    X = pd.concat([X.drop(columns=NUMERIC_COLS), scaled_df], axis=1)
    return X, scaler


def split_dataset(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_full_dataset(fix_seed=DEFAULT_SEED):
    set_seed(fix_seed)

    df = load_raw_data()
    df = clean_total_charges(df)
    df = build_business_features(df)
    df = encode_binary_features(df)
    df = encode_multi_category(df)

    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS+[TARGET_COL])

    X_scaled, scaler = standardize_features(X)
    X_train, X_test, y_train, y_test = split_dataset(X_scaled, y, fix_seed)
    feature_cols = X_scaled.columns.tolist()

    return X_train, X_test, y_train, y_test, scaler, feature_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = get_full_dataset()
    # 自测专用打印，正式训练脚本不会执行该分支
    print(f"训练集形状：{X_train.shape}")
    print(f"测试集形状：{X_test.shape}")