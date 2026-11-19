"""
【门禁B：训练前特征集校验脚本】
用途：清洗、特征工程完成之后，送入模型训练前的门禁
校验对象：telco_churn_clean.csv（v2特征数据集）
输出：校验报告json，可存入MLflow artifact
校验规则：schema、值域、枚举、缺失率、标签分布
"""
import pandas as pd
import json
import os
import numpy as np
from config.settings import CLEAN_DATA_PATH, TARGET_COL
from src.logger import logger

# ===================== 特征集Schema基准（门禁B专用） =====================
# 数值字段：最小值、最大值
NUMERIC_RULES = {
    "tenure": {"min": 0, "max": 72},
    "MonthlyCharges": {"min": 0, "max": 120},
    "TotalCharges": {"min": 0, "max": 9000},
    "avg_monthly_charges": {"min": 0, "max": 122}
}
# 分类字段：允许的枚举值
CAT_RULES = {
    "SeniorCitizen": ["0", "1"],
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    "tenure_group": ["new_user", "growing_user", "mature_user"]
}
# 清洗后标签：0/1 数字
TARGET_ENUM = [0, 1]

# 阈值
MAX_NULL_RATIO = 0.05
BASELINE_CHURN_RATE = 0.265
DRIFT_TOLERANCE = 0.03

REPORT_SAVE_PATH = "assets/data_validation_feature.json"

def convert_numpy_type(obj):
    """把numpy专属类型转成python原生类型，兼容json"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if np.isscalar(obj) and pd.isna(obj):
        return None
    if isinstance(obj, dict):
        return {k: convert_numpy_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_type(item) for item in obj]
    return obj

def validate_schema(df: pd.DataFrame) -> dict:
    """校验字段完整性，不能少字段、不能多出未知字段"""
    report = {"schema_ok": True, "msg": ""}
    std_cols = list(NUMERIC_RULES.keys()) + list(CAT_RULES.keys()) + [TARGET_COL, "customerID"]
    missing = set(std_cols) - set(df.columns)
    extra = set(df.columns) - set(std_cols)
    if missing:
        report["schema_ok"] = False
        report["msg"] = f"缺失字段：{missing}"
    if extra:
        report["schema_ok"] = False
        report["msg"] += f"；存在未知多余字段：{extra}"
    return report

def validate_numeric_range(df: pd.DataFrame) -> dict:
    """数值范围校验，负数、超范围异常值检测，先做类型转换"""
    fail_list = []
    for col, rule in NUMERIC_RULES.items():
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        convert_err_cnt = ser.isna().sum() - df[col].isna().sum()
        if convert_err_cnt > 0:
            fail_list.append({
                "col": col,
                "err_count": int(convert_err_cnt),
                "reason": "字符串无法转为数值"
            })
            continue
        min_violate = ser < rule["min"]
        max_violate = ser > rule["max"]
        cnt = (min_violate | max_violate).sum()
        if cnt > 0:
            fail_list.append({
                "col": col,
                "err_count": int(cnt),
                "rule": rule
            })
    return {"numeric_ok": len(fail_list) == 0, "fail_detail": fail_list}

def validate_categorical_enum(df: pd.DataFrame) -> dict:
    """分类列枚举校验，防止出现脏文本、错别字"""
    fail_list = []
    for col, allow_vals in CAT_RULES.items():
        if col not in df.columns:
            continue
        unique_vals = df[col].astype(str).unique()
        unknown = set(unique_vals) - set(allow_vals)
        if unknown:
            fail_list.append({"col": col, "unknown_values": list(unknown)})
    # 标签校验：标签已经是数字0/1，不再转字符串
    target_unknown = set(df[TARGET_COL].unique()) - set(TARGET_ENUM)
    if target_unknown:
        fail_list.append({"col": TARGET_COL, "unknown_values": list(target_unknown)})
    return {"cat_ok": len(fail_list) == 0, "fail_detail": fail_list}

def validate_null_ratio(df: pd.DataFrame) -> dict:
    """缺失率监控"""
    df_cp = df.copy()
    for col in NUMERIC_RULES:
        if col in df_cp.columns:
            df_cp[col] = pd.to_numeric(df_cp[col], errors="coerce")
    null_ratio = df_cp.isnull().sum() / len(df_cp)
    fail_list = []
    for col, ratio in null_ratio.items():
        if ratio > MAX_NULL_RATIO:
            fail_list.append({"col": col, "null_ratio": round(float(ratio), 4)})
    return {"null_ok": len(fail_list) == 0, "fail_detail": fail_list}

def validate_label_distribution(df: pd.DataFrame) -> dict:
    """标签分布漂移校验，0代表不流失，1代表流失"""
    churn_rate = (df[TARGET_COL] == 1).mean()
    drift = abs(churn_rate - BASELINE_CHURN_RATE)
    dist_ok = drift < DRIFT_TOLERANCE
    return {
        "dist_ok": dist_ok,
        "current_churn_rate": round(float(churn_rate), 4),
        "baseline": BASELINE_CHURN_RATE,
        "drift": round(float(drift), 4)
    }

def run_all_validation() -> tuple[bool, dict]:
    """
    统一校验入口
    return: pass_flag, report_dict
    pass_flag: True=全部校验通过，可以继续训练；False=禁止训练
    """
    logger.info("===== 【门禁B】特征集全量校验启动 =====")
    df = pd.read_csv(CLEAN_DATA_PATH)
    logger.info(f"待校验特征数据集 shape={df.shape}")

    schema_res = validate_schema(df)
    numeric_res = validate_numeric_range(df)
    cat_res = validate_categorical_enum(df)
    null_res = validate_null_ratio(df)
    dist_res = validate_label_distribution(df)

    all_ok = all([
        schema_res["schema_ok"],
        numeric_res["numeric_ok"],
        cat_res["cat_ok"],
        null_res["null_ok"],
        dist_res["dist_ok"]
    ])

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_path": CLEAN_DATA_PATH,
        "sample_count": int(len(df)),
        "schema": schema_res,
        "numeric_range": numeric_res,
        "categorical_enum": cat_res,
        "null_check": null_res,
        "label_dist": dist_res,
        "all_passed": all_ok
    }

    os.makedirs("assets", exist_ok=True)
    report = convert_numpy_type(report)
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"【门禁B】校验报告已写入：{REPORT_SAVE_PATH}")
    if all_ok:
        logger.info("✅ 门禁B校验全部通过，可以进入训练流程！")
    else:
        logger.error("❌ 门禁B校验失败，禁止启动训练！")
    return all_ok, report


if __name__ == "__main__":
    ok, rep = run_all_validation()
    rep = convert_numpy_type(rep)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
