import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config.settings import DATA_PATH, DATA_DIR, NUMERIC_COLS, TARGET_COL

def make_v2_clean_data():
    # 读取原始数据
    df = pd.read_csv(DATA_PATH)

    # 1. TotalCharges清洗，和data_process保持一致
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2. 构造衍生特征
    df["avg_monthly_charges"] = df["TotalCharges"] / df["tenure"]
    df["avg_monthly_charges"] = df["avg_monthly_charges"].fillna(0)

    # 3. 标签编码
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # ✅ 这里停止！不做onehot、不做划分、不做scaler
    # 这一份v2：清洗+衍生特征+标签01编码，保留原始类别字段，不做独热编码
    # 独热编码、标准化属于训练流水线，放到训练脚本里，不打进数据集版本

    out_path = os.path.join(DATA_DIR, "telco_churn_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"v2清洗数据集生成成功：{out_path}")
    print(f"v2 shape: {df.shape}")

if __name__ == "__main__":
    make_v2_clean_data()
