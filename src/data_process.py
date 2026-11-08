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
    TARGET_COL,
    VAL_SEED_OFFSET
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
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def build_business_features(df):
    df = df.copy()
    logger.info("开始构造衍生业务特征：avg_monthly_charges")
    df["avg_monthly_charges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].replace([np.inf, -np.inf], np.nan)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].fillna(df["MonthlyCharges"])
    return df


def encode_binary_features(df):
    df = df.copy()
    logger.info("开始处理二元类别特征编码")
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


def raw_data_transform(df: pd.DataFrame,median_dict: dict):
    """
    推理接口专用！输入【原始业务DataFrame】，和训练执行完全一样清洗、衍生、编码
    ❗不读取csv、❗不做标准化、❗不分割数据集
    """
    logger.info("进入推理数据预处理流水线，执行与训练一致的数据变换逻辑")
    df = df.copy()
    df = clean_total_charges(df)
    df = build_business_features(df)

    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(median_dict)

    df = encode_binary_features(df)
    df = encode_multi_category(df)

    drop_cols = DROP_COLS.copy()
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    logger.info(f"推理数据预处理结束，处理后特征维度：{df.shape[1]}")
    return df


def standardize_features(X_train):
    """只用训练集fit，返回标准化后的训练集、scaler、训练集各数值列中位数"""
    X = X_train.copy()
    logger.info("[TRANSFORM阶段]基于训练集拟合scaler，统计中位数用于缺失填充")
    scaler = StandardScaler()
    median_dict = {}
    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median_dict[col] = X[col].median()
            X[col] = X[col].fillna(median_dict.get(col, 0))

    scaled_array = scaler.fit_transform(X[NUMERIC_COLS])
    scaled_df = pd.DataFrame(scaled_array, columns=NUMERIC_COLS, index=X.index)
    X = pd.concat([X.drop(columns=NUMERIC_COLS), scaled_df], axis=1)
    logger.info("训练集标准化完成")
    return X, scaler, median_dict


def standardize_transform(X, scaler, median_dict):
    """验证集/测试集/推理: 只Transform,不用fit,中位数用训练集的"""
    X = X.copy()
    logger.info("[transform阶段]复用训练集scaler和中位数做标准化")
    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(median_dict.get(col, 0))

    scaled_array = scaler.transform(X[NUMERIC_COLS])
    scaled_df = pd.DataFrame(scaled_array, columns=NUMERIC_COLS, index=X.index)
    X = pd.concat([X.drop(columns=NUMERIC_COLS), scaled_df], axis=1)
    return X


def split_dataset(X, y, seed):
    """
    ✅ 工业级改造：6:2:2 划分 + 固定测试集考卷（防数据漂移）
    """
    logger.info("开始划分数据集 | 目标比例：Train 60% / Val 20% / Test 20%")


    cache_path = "assets/test_indices.npy"

    os.makedirs("assets", exist_ok=True)


    if os.path.exists(cache_path):
        test_idx = np.load(cache_path)
        test_idx = pd.Index(test_idx)  # 转为 pandas 索引以便校验

        # 防止数据集更新导致旧索引越界引发 KeyError
        if not test_idx.isin(X.index).all():
            logger.error("缓存的测试集索引包含当前数据集不存在的行号！")
            raise ValueError("数据集已变动，请手动删除 assets/test_indices.npy 后重新运行代码！")

        logger.info(f"✅ 读取已缓存的测试集名单，共 {len(test_idx)} 条")
    else:

        train_val_idx, test_idx = train_test_split(
            X.index,
            test_size=TEST_SIZE,  # settings 里是 0.2
            random_state=seed,
            stratify=y
        )
        np.save(cache_path, test_idx)
        logger.info(f"✅ 第一次运行，已生成并持久化测试集名单 (assets/test_indices.npy)，共 {len(test_idx)} 条")

    #剩下的 80% 索引，保留Index类型，不转list，保证顺序稳定
    train_val_idx = X.index.difference(test_idx)

    #第二次切分：在剩下的 80% 中切分验证集，test_size=0.25
    # 单独用一个新种子，避免和第一次随机序列耦合
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=VAL_SEED_OFFSET,
        stratify=y.loc[train_val_idx]
    )

    #按名单提取数据
    X_train_raw = X.loc[train_idx].copy()
    X_val_raw = X.loc[val_idx].copy()
    X_test_raw = X.loc[test_idx].copy()

    y_train = y.loc[train_idx].copy()
    y_val = y.loc[val_idx].copy()
    y_test = y.loc[test_idx].copy()

    logger.info(f"数据集划分完成 | 训练集:{len(train_idx)}条 | 验证集:{len(val_idx)}条 | 测试集:{len(test_idx)}条")

    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test


def get_full_dataset(fix_seed=DEFAULT_SEED):
    logger.info("===== 启动完整数据集加载&预处理流水线 =====")
    set_seed(fix_seed)

    df = load_raw_data()
    df = clean_total_charges(df)
    df = build_business_features(df)
    df = encode_binary_features(df)
    df = encode_multi_category(df)

    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    logger.info(f"分离特征与标签，原始特征矩阵shape={X.shape}")

    # ✅ 接收 6 份原始数据
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_dataset(X, y, fix_seed)

    # 💡 面试重点：防泄露铁律！只有训练集配 fit，验证和测试只配 transform
    X_train, scaler, median_dict = standardize_features(X_train_raw)
    X_val = standardize_transform(X_val_raw, scaler, median_dict)
    X_test = standardize_transform(X_test_raw, scaler, median_dict)

    # 排查 1：三份数据特征列是否一致
    if list(X_train.columns) != list(X_test.columns) or list(X_train.columns) != list(X_val.columns):
        logger.error("【严重】训练集、验证集、测试集特征列不一样！")
        logger.error(f"训练独有列：{set(X_train.columns) - set(X_test.columns)}")
    else:
        logger.info("排查通过：训练集、验证集、测试集特征列完全一致")

    # 排查 2：检查空值
    train_null_cnt = X_train.isnull().sum().sum()
    val_null_cnt = X_val.isnull().sum().sum()
    test_null_cnt = X_test.isnull().sum().sum()
    if train_null_cnt > 0 or test_null_cnt > 0 or val_null_cnt > 0:
        logger.error(f"【警告】存在空值！训练集={train_null_cnt}, 验证集={val_null_cnt}, 测试集={test_null_cnt}")
    else:
        logger.info("排查通过：训练集、验证集、测试集没有空值")

    # 排查 3：流失标签比例对比 (验证分层抽样是否成功)
    logger.info(
        f"训练集流失比例：{y_train.mean():.4f}，验证集流失比例：{y_val.mean():.4f}，测试集流失比例：{y_test.mean():.4f}")

    # 排查 4：确认中位数是训练集算出
    logger.info(f"排查校验：中位数字典来自训练集，一共统计了 {len(median_dict)} 个数值列")

    feature_cols = X_train.columns.tolist()
    logger.info(f"完整数据处理流程结束，最终特征列表长度：{len(feature_cols)}")

    # ✅ 返回 9 个值
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, median_dict, feature_cols


if __name__ == "__main__":
    logger.info("执行data_process自测入口")
    # ✅ 接收 9 个值
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, median_dict, features = get_full_dataset()

    logger.info(f"【自测输出】训练集形状：{X_train.shape}")
    logger.info(f"【自测输出】验证集形状：{X_val.shape}")
    logger.info(f"【自测输出】测试集形状：{X_test.shape}")

