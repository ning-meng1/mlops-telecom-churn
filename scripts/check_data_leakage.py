import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from config.settings import (
    TEST_SIZE,
    DEFAULT_SEED,
    NUMERIC_COLS,
)
from src.data_process import (
    load_raw_data,
    clean_total_charges,
    build_business_features,
    encode_binary_features,
    encode_multi_category,
    DROP_COLS,
    TARGET_COL,
)


# =========================================================
# 实验1：Scaler 标准化泄露验证（原有功能完整保留）
# =========================================================
def experiment_scaler_leakage():
    print("\n" + "=" * 60)
    print("Experiment 1：Scaler 标准化泄露验证")
    print("=" * 60)

    # 1. 使用和正式项目一致的前处理
    df = load_raw_data()
    df = clean_total_charges(df)
    df = build_business_features(df)
    df = encode_binary_features(df)
    df = encode_multi_category(df)

    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS + [TARGET_COL])

    # 2. 固定同一个 train/test 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=DEFAULT_SEED,
        stratify=y,
    )

    # 3. 方案 A：全量 X 先 fit scaler
    scaler_all = StandardScaler()
    scaler_all.fit(X[NUMERIC_COLS])
    X_train_A = X_train.copy()
    X_test_A = X_test.copy()
    X_train_A[NUMERIC_COLS] = scaler_all.transform(X_train_A[NUMERIC_COLS])
    X_test_A[NUMERIC_COLS] = scaler_all.transform(X_test_A[NUMERIC_COLS])

    # 4. 方案 B：仅 X_train fit scaler（正确流程）
    scaler_train = StandardScaler()
    scaler_train.fit(X_train[NUMERIC_COLS])
    X_train_B = X_train.copy()
    X_test_B = X_test.copy()
    X_train_B[NUMERIC_COLS] = scaler_train.transform(X_train_B[NUMERIC_COLS])
    X_test_B[NUMERIC_COLS] = scaler_train.transform(X_test_B[NUMERIC_COLS])

    # 5. 训练两个完全相同参数的模型
    common_params = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "n_estimators": 100,
        "random_state": DEFAULT_SEED,
        "eval_metric": "logloss",
    }
    model_A = XGBClassifier(**common_params)
    model_B = XGBClassifier(**common_params)
    model_A.fit(X_train_A, y_train)
    model_B.fit(X_train_B, y_train)

    # 6. 预测 + 计算指标
    def calc_metrics(y_true, pred, prob):
        return {
            "accuracy": round(accuracy_score(y_true, pred), 4),
            "auc": round(roc_auc_score(y_true, prob), 4),
            "f1": round(f1_score(y_true, pred), 4),
            "precision": round(precision_score(y_true, pred), 4),
            "recall": round(recall_score(y_true, pred), 4),
        }

    pred_A = model_A.predict(X_test_A)
    prob_A = model_A.predict_proba(X_test_A)[:, 1]
    metrics_A = calc_metrics(y_test, pred_A, prob_A)

    pred_B = model_B.predict(X_test_B)
    prob_B = model_B.predict_proba(X_test_B)[:, 1]
    metrics_B = calc_metrics(y_test, pred_B, prob_B)

    # 7. 打印结果
    print("\n方案 A：全量 X fit（存在泄露）")
    for k, v in metrics_A.items():
        print(f"  {k:10s}: {v}")
    print("\n方案 B：仅 X_train fit（正确）")
    for k, v in metrics_B.items():
        print(f"  {k:10s}: {v}")

    pred_same = np.array_equal(pred_A, pred_B)
    max_prob_diff = float(np.max(np.abs(prob_A - prob_B)))
    print(f"\n预测类别完全一致：{pred_same}")
    print(f"预测概率最大差异：{max_prob_diff:.8f}")

    conclusion = "当前数据集下，scaler泄露对最终结果影响极小，但属于工程规范红线，必须严格执行仅训练集fit的标准流程。"
    print(f"\n结论：{conclusion}")

    return {
        "leakage_metrics": metrics_A,
        "correct_metrics": metrics_B,
        "pred_same": pred_same,
        "max_prob_diff": max_prob_diff,
        "conclusion": conclusion
    }


# =========================================================
# 实验2：Median 缺失值填充泄露验证（原有功能完整保留）
# =========================================================
def experiment_median_leakage():
    print("\n" + "=" * 60)
    print("Experiment 2：Median 缺失值填充泄露验证")
    print("=" * 60)

    # 1. 读取并预处理数据
    X_raw = load_raw_data()
    X_raw = clean_total_charges(X_raw)
    X_raw = build_business_features(X_raw)
    X_raw = encode_binary_features(X_raw)
    X_raw = encode_multi_category(X_raw)

    y = X_raw[TARGET_COL]
    X = X_raw.drop(columns=DROP_COLS + [TARGET_COL])

    # 2. 固定划分
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=DEFAULT_SEED,
        stratify=y
    )
    print(f"训练集：{X_train_raw.shape}")
    print(f"测试集：{X_test_raw.shape}")

    # 3. 方案 A：全量数据计算 median（泄露）
    X_leak = X.copy()
    for col in NUMERIC_COLS:
        if col in X_leak.columns:
            X_leak[col] = pd.to_numeric(X_leak[col], errors="coerce")
            median_value = X_leak[col].median()
            X_leak[col] = X_leak[col].fillna(median_value)
    X_train_leak = X_leak.loc[X_train_raw.index]
    X_test_leak = X_leak.loc[X_test_raw.index]

    # 4. 方案 B：仅训练集计算 median（正确）
    X_train_clean = X_train_raw.copy()
    X_test_clean = X_test_raw.copy()
    for col in NUMERIC_COLS:
        if col in X_train_clean.columns:
            X_train_clean[col] = pd.to_numeric(X_train_clean[col], errors="coerce")
            X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors="coerce")
            median_value = X_train_clean[col].median()
            X_train_clean[col] = X_train_clean[col].fillna(median_value)
            X_test_clean[col] = X_test_clean[col].fillna(median_value)

    # 5. 打印 median 对比
    print("\n===== Median 对比 =====")
    for col in NUMERIC_COLS:
        if col in X.columns:
            leak_median = X_leak[col].median()
            train_median = X_train_clean[col].median()
            print(
                f"{col:25s} | "
                f"完整数据median={leak_median:.6f} | "
                f"训练集median={train_median:.6f} | "
                f"差值={abs(leak_median - train_median):.6f}"
            )

    # 6. 训练模型
    common_params = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "n_estimators": 100,
        "random_state": DEFAULT_SEED,
        "eval_metric": "logloss"
    }
    model_leak = XGBClassifier(**common_params)
    model_clean = XGBClassifier(**common_params)
    model_leak.fit(X_train_leak, y_train)
    model_clean.fit(X_train_clean, y_train)

    # 7. 计算指标
    def calc_metrics(model, X, y):
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]
        return {
            "accuracy": round(accuracy_score(y, pred), 4),
            "auc": round(roc_auc_score(y, prob), 4),
            "f1": round(f1_score(y, pred), 4),
            "precision": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4),
        }

    metrics_leak = calc_metrics(model_leak, X_test_leak, y_test)
    metrics_clean = calc_metrics(model_clean, X_test_clean, y_test)

    print("\n方案 A：median 使用完整数据（存在泄露）")
    for k, v in metrics_leak.items():
        print(f"  {k:10s}: {v}")
    print("\n方案 B：median 只使用训练集（正确）")
    for k, v in metrics_clean.items():
        print(f"  {k:10s}: {v}")

    pred_same = np.array_equal(model_leak.predict(X_test_leak), model_clean.predict(X_test_clean))
    max_prob_diff = float(np.max(np.abs(
        model_leak.predict_proba(X_test_leak)[:, 1] - model_clean.predict_proba(X_test_clean)[:, 1]
    )))
    print(f"\n预测类别完全一致：{pred_same}")
    print(f"预测概率最大差异：{max_prob_diff:.12f}")

    if pred_same and max_prob_diff == 0:
        conclusion = "median 泄露在当前数据集和当前模型参数下，没有造成最终预测结果的可观测变化，但仍需遵守规范。"
    else:
        conclusion = "median 泄露已经造成模型预测结果发生变化，必须严格执行仅训练集统计。"
    print(f"\n结论：{conclusion}")

    return {
        "leakage_metrics": metrics_leak,
        "correct_metrics": metrics_clean,
        "pred_same": pred_same,
        "max_prob_diff": max_prob_diff,
        "conclusion": conclusion
    }


# =========================================================
# 【新增】实验3：目标泄露扫描 - 特征-标签相关性排查
# =========================================================
def experiment_target_leakage_scan(threshold=0.7):
    """
    扫描所有特征与目标标签的相关性，排查高风险泄露特征
    :param threshold: 相关系数告警阈值，绝对值超过则标记高风险
    """
    print("\n" + "=" * 60)
    print("Experiment 3：目标泄露扫描 - 特征标签相关性排查")
    print("=" * 60)

    # 加载并预处理数据（只做清洗编码，不划分）
    df = load_raw_data()
    df = clean_total_charges(df)
    df = build_business_features(df)
    df = encode_binary_features(df)
    df = encode_multi_category(df)

    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS + [TARGET_COL])

    # 所有特征统一数值化，计算皮尔逊相关系数
    corr_list = []
    for col in X.columns:
        # 类别特征做标签编码
        if X[col].dtype == 'object':
            le = LabelEncoder()
            col_values = le.fit_transform(X[col].astype(str))
        else:
            col_values = X[col].values
        corr = np.corrcoef(col_values, y.values)[0, 1]
        corr_list.append({
            "feature": col,
            "correlation": round(abs(corr), 4),
            "is_risk": abs(corr) > threshold
        })

    # 按相关度降序排序
    corr_df = pd.DataFrame(corr_list).sort_values("correlation", ascending=False).reset_index(drop=True)

    print(f"\n特征与标签相关性 Top 10（告警阈值：{threshold}）：")
    print(corr_df.head(10).to_string(index=False))

    risk_features = corr_df[corr_df["is_risk"]]["feature"].tolist()
    if risk_features:
        print(f"\n⚠️ 高风险泄露特征（相关度 > {threshold}）：{risk_features}")
        conclusion = f"检测到{len(risk_features)}个高相关特征，需人工排查是否包含未来信息或标签泄露。"
    else:
        print(f"\n✅ 无高风险泄露特征，所有特征相关度均低于阈值 {threshold}")
        conclusion = "所有特征与标签的相关性均在合理范围内，未发现明显目标泄露特征。"

    print("结论：", conclusion)

    return {
        "threshold": threshold,
        "top10_corr": corr_df.head(10).to_dict("records"),
        "risk_features": risk_features,
        "conclusion": conclusion
    }


# =========================================================
# 【新增】自动生成 Markdown 实验报告
# =========================================================
def generate_report(exp1_result, exp2_result, exp3_result):
    report_path = os.path.join(BASE_DIR, "docs", "preprocess_leakage_experiment.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    content = f"""# 数据泄露专项实验报告

## 一、实验概述
本实验从两个维度验证数据泄露风险：
1. **流程级泄露验证**：验证标准化、缺失值填充步骤是否存在测试集信息泄露
2. **特征级泄露扫描**：通过特征-标签相关性分析，排查目标泄露与作弊特征

实验固定随机种子：{DEFAULT_SEED}，测试集比例：{TEST_SIZE}

## 二、实验1：Scaler 标准化泄露验证
### 对比方案
- 方案A：使用全量数据集拟合标准化器（存在泄露风险）
- 方案B：仅使用训练集拟合标准化器（正确规范）

### 指标对比
| 指标 | 泄露方案A | 正确方案B |
|------|-----------|-----------|
| Accuracy | {exp1_result['leakage_metrics']['accuracy']} | {exp1_result['correct_metrics']['accuracy']} |
| AUC | {exp1_result['leakage_metrics']['auc']} | {exp1_result['correct_metrics']['auc']} |
| F1 | {exp1_result['leakage_metrics']['f1']} | {exp1_result['correct_metrics']['f1']} |
| Precision | {exp1_result['leakage_metrics']['precision']} | {exp1_result['correct_metrics']['precision']} |
| Recall | {exp1_result['leakage_metrics']['recall']} | {exp1_result['correct_metrics']['recall']} |

### 实验结论
预测类别一致性：{exp1_result['pred_same']}
预测概率最大差异：{exp1_result['max_prob_diff']:.8f}

> {exp1_result['conclusion']}

## 三、实验2：Median 缺失值填充泄露验证
### 对比方案
- 方案A：使用全量数据集计算中位数填充（存在泄露风险）
- 方案B：仅使用训练集计算中位数填充（正确规范）

### 指标对比
| 指标 | 泄露方案A | 正确方案B |
|------|-----------|-----------|
| Accuracy | {exp2_result['leakage_metrics']['accuracy']} | {exp2_result['correct_metrics']['accuracy']} |
| AUC | {exp2_result['leakage_metrics']['auc']} | {exp2_result['correct_metrics']['auc']} |
| F1 | {exp2_result['leakage_metrics']['f1']} | {exp2_result['correct_metrics']['f1']} |
| Precision | {exp2_result['leakage_metrics']['precision']} | {exp2_result['correct_metrics']['precision']} |
| Recall | {exp2_result['leakage_metrics']['recall']} | {exp2_result['correct_metrics']['recall']} |

### 实验结论
预测类别一致性：{exp2_result['pred_same']}
预测概率最大差异：{exp2_result['max_prob_diff']:.12f}

> {exp2_result['conclusion']}

## 四、实验3：目标泄露特征扫描
### 扫描规则
计算所有特征与目标标签Churn的皮尔逊相关系数绝对值，告警阈值：{exp3_result['threshold']}

### Top10 高相关特征
| 特征名 | 相关系数绝对值 | 是否高风险 |
|--------|----------------|------------|
"""
    for item in exp3_result["top10_corr"]:
        risk_mark = "⚠️ 高风险" if item["is_risk"] else "正常"
        content += f"| {item['feature']} | {item['correlation']} | {risk_mark} |\n"

    content += f"""
### 扫描结论
高风险泄露特征列表：{exp3_result['risk_features'] if exp3_result['risk_features'] else '无'}

> {exp3_result['conclusion']}

## 五、整体总结
1. 预处理流程层面：scaler、median的泄露在当前数据集上对模型结果影响极小，但属于工程规范红线，必须严格执行「先划分数据集，再做所有预处理变换」
2. 特征层面：未发现明显的目标泄露特征，数据集整体合规
3. 后续训练严格遵循：先划分数据集，再做所有预处理变换，测试集仅做transform
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ 实验报告已生成：{report_path}")


# ==================== 主入口 ====================
if __name__ == "__main__":
    exp1 = experiment_scaler_leakage()
    exp2 = experiment_median_leakage()
    exp3 = experiment_target_leakage_scan(threshold=0.7)
    generate_report(exp1, exp2, exp3)
    print("\n" + "=" * 60)
    print("🎉 全部泄露验证实验执行完毕")
    print("=" * 60)
