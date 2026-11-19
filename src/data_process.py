import pandas as pd
import numpy as np
import random
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

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
    """
    固定全局随机种子
    目的：保证数据集划分、特征变换、模型训练可复现，消除随机波动，满足MLOps实验可追溯要求
    Args:
        seed: int 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"数据处理模块全局随机种子已设置为：{seed}")


def load_raw_data():
    """
    加载原始数据集
    只做读取，不做任何清洗、特征变换，隔离IO与数据处理逻辑
    Returns:
        pd.DataFrame: 原始数据集
    """
    logger.info(f"开始读取原始数据集，文件路径：{DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"原始数据加载完成，数据集shape={df.shape}")
    return df


def clean_total_charges(df):
    """
    清洗TotalCharges字段
    业务背景：电信数据集该字段存在空格字符串，无法直接转为数值
    填充策略：空值用当月月租MonthlyCharges填充，符合业务逻辑
    Args:
        df: pd.DataFrame 原始数据集
    Returns:
        pd.DataFrame: 清洗后数据集
    """
    df = df.copy()
    logger.info("执行TotalCharges字段清洗，处理空字符串与缺失值")
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def build_business_features(df):
    """
    构造业务衍生特征
    1. avg_monthly_charges：平均月消费，总消费/在网时长
    2. tenure_group：在网时长分群，用于业务分层分析
    Args:
        df: pd.DataFrame 经过基础清洗的数据集
    Returns:
        pd.DataFrame: 增加衍生特征后的数据集
    """
    df = df.copy()
    logger.info("开始构造衍生业务特征：avg_monthly_charges")
    df["avg_monthly_charges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].replace([np.inf, -np.inf], np.nan)
    df["avg_monthly_charges"] = df["avg_monthly_charges"].fillna(df["MonthlyCharges"])

    # tenure 在网时长分群：新用户 / 成长用户 / 成熟老用户
    logger.info("开始构造tenure在网时长分群特征：tenure_group")

    def get_tenure_group(month):
        if month <= 6:
            return "new_user"
        elif month <= 24:
            return "growing_user"
        else:
            return "mature_user"

    df["tenure_group"] = df["tenure"].apply(get_tenure_group)
    logger.info("tenure分群特征构造完成，共3个分群：new_user / growing_user / mature_user")

    return df


def encode_binary_features(df):
    """
    二元特征编码：Yes/No、二分类标签转为0/1
    注意：该步骤只处理二元字段，不涉及多分类OneHot，防止特征列提前销毁
    Args:
        df: pd.DataFrame 数据集
    Returns:
        pd.DataFrame: 二元编码完成数据集
    """
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
    """
    多分类特征OneHot编码
    drop_first=True：消除多重共线性，减少冗余特征
    ⚠️ 重要风险：执行后原始分类字段会被删除，转为多列0/1，业务统计分析必须放在本函数之前
    Args:
        df: pd.DataFrame 数据集
    Returns:
        pd.DataFrame: OneHot编码后数据集
    """
    df = df.copy()
    logger.info(f"对多分类特征执行OneHot编码，待编码字段：{MULTI_CAT_COLS}")
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)
    logger.info(f"多分类编码完成，当前总特征数量：{df.shape[1]}")
    return df


def raw_data_transform(df: pd.DataFrame, median_dict: dict):
    """
    推理侧数据预处理流水线
    约束：推理流程必须和训练流程变换逻辑完全对齐，防止训练/推理分布偏移（训练推理Skew）
    ❗不读取csv、❗不做标准化、❗不分割数据集
    Args:
        df: pd.DataFrame 单条/批量原始业务输入数据
        median_dict: dict 训练集统计得到的数值列中位数，用于缺失填充
    Returns:
        pd.DataFrame: 编码完成、待标准化的特征矩阵
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
    """
    训练集标准化拟合
    工业规范：Scaler仅允许在训练集fit，验证/测试集只能transform，杜绝数据泄露
    同时统计数值列中位数，保存用于推理阶段缺失填充
    Args:
        X_train: pd.DataFrame 训练集原始特征矩阵
    Returns:
        X_train_scaled: pd.DataFrame 标准化后的训练集
        scaler: StandardScaler 拟合好的标准化转换器
        median_dict: dict 各数值列中位数
    """
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
    """
    验证集、测试集、推理数据标准化变换
    只transform，不fit；缺失填充复用训练集统计量，保证分布一致
    Args:
        X: pd.DataFrame 待标准化特征矩阵
        scaler: StandardScaler 训练阶段拟合好的转换器
        median_dict: dict 训练集中位数
    Returns:
        pd.DataFrame: 标准化后数据
    """
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
    数据集分层划分：训练60% / 验证20% / 测试20%
    工业设计：测试集索引持久化缓存，固定"考卷"，多次实验使用同一套测试集，保证指标横向可比
    分层抽样 stratify：保证训练/验证/测试流失标签比例和全集一致，避免类别分布偏移
    Args:
        X: pd.DataFrame 全量特征矩阵
        y: pd.Series 全量标签
        seed: int 随机种子
    Returns:
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test
    """
    logger.info("开始划分数据集 | 目标比例：Train 60% / Val 20% / Test 20%")

    cache_path = "assets/test_indices.npy"
    os.makedirs("assets", exist_ok=True)

    if os.path.exists(cache_path):
        test_idx = np.load(cache_path)
        test_idx = pd.Index(test_idx)
        # 校验：数据集变更后旧索引失效，抛出异常，防止下标越界
        if not test_idx.isin(X.index).all():
            logger.error("缓存的测试集索引包含当前数据集不存在的行号！")
            raise ValueError("数据集已变动，请手动删除 assets/test_indices.npy 后重新运行代码！")
        logger.info(f"✅ 读取已缓存的测试集名单，共 {len(test_idx)} 条")
    else:
        train_val_idx, test_idx = train_test_split(
            X.index,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y
        )
        np.save(cache_path, test_idx)
        logger.info(f"✅ 第一次运行，已生成并持久化测试集名单 (assets/test_indices.npy)，共 {len(test_idx)} 条")

    train_val_idx = X.index.difference(test_idx)
    # 剩余80%切分训练、验证，使用独立种子，避免随机序列耦合
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=VAL_SEED_OFFSET,
        stratify=y.loc[train_val_idx]
    )

    X_train_raw = X.loc[train_idx].copy()
    X_val_raw = X.loc[val_idx].copy()
    X_test_raw = X.loc[test_idx].copy()

    y_train = y.loc[train_idx].copy()
    y_val = y.loc[val_idx].copy()
    y_test = y.loc[test_idx].copy()

    logger.info(f"数据集划分完成 | 训练集:{len(train_idx)}条 | 验证集:{len(val_idx)}条 | 测试集:{len(test_idx)}条")
    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test


def get_full_dataset(fix_seed=DEFAULT_SEED):
    """
    【统一入口】完整数据预处理流水线
    执行顺序规范（面试重点）：
        1. 加载原始数据 -> 基础清洗
        2. 构造业务衍生特征（tenure_group 文字分组保留）
        3. 二元编码：标签转为0/1，为业务统计做准备
        4. 业务分层统计：基于tenure_group统计各组流失率（必须在onehot之前）
        5. 多分类OneHot编码（销毁原始分类列，转为模型可用特征）
        6. 分层数据集划分，固定测试集考卷
        7. 训练集拟合标准化，验证/测试集仅transform
        8. 数据合法性校验：特征对齐、空值校验、标签分布校验
    Args:
        fix_seed: int 全局随机种子
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, median_dict, feature_cols
    """
    logger.info("===== 启动完整数据集加载&预处理流水线 =====")
    set_seed(fix_seed)

    df = load_raw_data()
    df = clean_total_charges(df)
    df = build_business_features(df)

    # 二元编码前置：Churn标签转为0/1，满足均值计算条件
    df = encode_binary_features(df)

    # 业务分层分析：基于在网时长分组统计流失率
    # 约束：tenure_group尚未被onehot销毁，标签已经数值化，满足统计条件
    group_churn = df.groupby("tenure_group")[TARGET_COL].agg(["count", "mean"])
    group_churn.columns = ["用户数", "流失率"]
    logger.info("===== tenure分群业务分析 =====")
    for group_name, row in group_churn.iterrows():
        logger.info(f"{group_name:15s} | 用户数：{int(row['用户数']):4d} | 流失率：{row['流失率']:.4f}")
    new_churn = group_churn.loc["new_user", "流失率"]
    mature_churn = group_churn.loc["mature_user", "流失率"]
    logger.info(f"业务结论：新用户流失率是成熟老用户的 {new_churn / mature_churn:.2f} 倍")

    # OneHot编码后置：业务分析完成后，再转换为模型可识别的0/1特征
    df = encode_multi_category(df)

    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    logger.info(f"分离特征与标签，原始特征矩阵shape={X.shape}")

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_dataset(X, y, fix_seed)

    # 标准化：严格遵循训练集fit，其余仅transform，防数据泄露
    X_train, scaler, median_dict = standardize_features(X_train_raw)
    X_val = standardize_transform(X_val_raw, scaler, median_dict)
    X_test = standardize_transform(X_test_raw, scaler, median_dict)

    # 校验1：训练、验证、测试特征列完全对齐，防止推理维度不一致
    if list(X_train.columns) != list(X_test.columns) or list(X_train.columns) != list(X_val.columns):
        logger.error("【严重】训练集、验证集、测试集特征列不一样！")
        logger.error(f"训练独有列：{set(X_train.columns) - set(X_test.columns)}")
    else:
        logger.info("排查通过：训练集、验证集、测试集特征列完全一致")

    # 校验2：全数据集空值校验，提前拦截脏数据
    train_null_cnt = X_train.isnull().sum().sum()
    val_null_cnt = X_val.isnull().sum().sum()
    test_null_cnt = X_test.isnull().sum().sum()
    if train_null_cnt > 0 or test_null_cnt > 0 or val_null_cnt > 0:
        logger.error(f"【警告】存在空值！训练集={train_null_cnt}, 验证集={val_null_cnt}, 测试集={test_null_cnt}")
    else:
        logger.info("排查通过：训练集、验证集、测试集没有空值")

    # 校验3：分层抽样校验，监控标签分布漂移
    logger.info(
        f"训练集流失比例：{y_train.mean():.4f}，验证集流失比例：{y_val.mean():.4f}，测试集流失比例：{y_test.mean():.4f}")

    # 校验4：确认填充统计量来源是训练集，杜绝泄露
    logger.info(f"排查校验：中位数字典来自训练集，一共统计了 {len(median_dict)} 个数值列")

    feature_cols = X_train.columns.tolist()
    logger.info(f"完整数据处理流程结束，最终特征列表长度：{len(feature_cols)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, median_dict, feature_cols


if __name__ == "__main__":
    """
    模块自测入口
    单独运行该脚本，验证整条数据流水线无异常、输出维度符合预期
    """
    logger.info("执行data_process自测入口")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, median_dict, features = get_full_dataset()

    logger.info(f"【自测输出】训练集形状：{X_train.shape}")
    logger.info(f"【自测输出】验证集形状：{X_val.shape}")
    logger.info(f"【自测输出】测试集形状：{X_test.shape}")
