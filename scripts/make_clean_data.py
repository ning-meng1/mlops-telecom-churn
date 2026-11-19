import sys
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config.settings import DATA_PATH, DATA_DIR, TARGET_COL
from src.data_process import (
    load_raw_data,
    clean_total_charges,
    build_business_features,
    encode_binary_features
)
from src.logger import logger


def make_v2_clean_data():
    """
    生成v2版本特征数据集
    处理范围：原始数据加载 → 清洗 → 业务衍生特征 → 二元标签编码
    仅单独把目标列Churn转0/1；其余分类字段保留原始文本（Yes/No）
    不做独热编码、不做标准化、不划分训练测试集、不批量编码其他二元特征
    输出：telco_churn_clean.csv，交由门禁B校验 + DVC版本管理
    """
    # 复用data_process流水线，保证逻辑唯一来源
    df = load_raw_data()
    df = clean_total_charges(df)
    df = build_business_features(df)

    # 只转换目标标签，其他二元字段不动
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    out_path = os.path.join(DATA_DIR, "telco_churn_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(f"v2清洗数据集生成成功：{out_path}")
    logger.info(f"v2 shape: {df.shape}")


if __name__ == "__main__":
    make_v2_clean_data()
