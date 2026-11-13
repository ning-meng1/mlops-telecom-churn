import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 1. 使用和正式项目一致的前处理
# =========================================================

df = load_raw_data()
df = clean_total_charges(df)
df = build_business_features(df)
df = encode_binary_features(df)
df = encode_multi_category(df)

y = df[TARGET_COL]
X = df.drop(columns=DROP_COLS + [TARGET_COL])


# =========================================================
# 2. 固定同一个 train/test 划分
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=DEFAULT_SEED,
    stratify=y,
)


# =========================================================
# 3. 方案 A：模拟当前代码
#    整个 X 先 fit scaler，再使用
# =========================================================

scaler_all = StandardScaler()

scaler_all.fit(X[NUMERIC_COLS])

X_train_A = X_train.copy()
X_test_A = X_test.copy()

X_train_A[NUMERIC_COLS] = scaler_all.transform(
    X_train_A[NUMERIC_COLS]
)

X_test_A[NUMERIC_COLS] = scaler_all.transform(
    X_test_A[NUMERIC_COLS]
)


# =========================================================
# 4. 方案 B：正确流程
#    只使用 X_train fit scaler
# =========================================================

scaler_train = StandardScaler()

scaler_train.fit(X_train[NUMERIC_COLS])

X_train_B = X_train.copy()
X_test_B = X_test.copy()

X_train_B[NUMERIC_COLS] = scaler_train.transform(
    X_train_B[NUMERIC_COLS]
)

X_test_B[NUMERIC_COLS] = scaler_train.transform(
    X_test_B[NUMERIC_COLS]
)


# =========================================================
# 5. 使用相同模型参数训练两个模型
# =========================================================

model_A = XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=100,
    random_state=DEFAULT_SEED,
    eval_metric="logloss",
)

model_B = XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=100,
    random_state=DEFAULT_SEED,
    eval_metric="logloss",
)


model_A.fit(X_train_A, y_train)
model_B.fit(X_train_B, y_train)


# =========================================================
# 6. 分别预测测试集
# =========================================================

pred_A = model_A.predict(X_test_A)
prob_A = model_A.predict_proba(X_test_A)[:, 1]

pred_B = model_B.predict(X_test_B)
prob_B = model_B.predict_proba(X_test_B)[:, 1]


# =========================================================
# 7. 计算指标
# =========================================================

def evaluate_model(name, y_true, pred, prob):

    print(f"\n========== {name} ==========")

    print(
        "Accuracy :",
        round(accuracy_score(y_true, pred), 4)
    )

    print(
        "AUC      :",
        round(roc_auc_score(y_true, prob), 4)
    )

    print(
        "F1       :",
        round(f1_score(y_true, pred), 4)
    )

    print(
        "Precision:",
        round(precision_score(y_true, pred), 4)
    )

    print(
        "Recall   :",
        round(recall_score(y_true, pred), 4)
    )


evaluate_model(
    "方案 A：全量 X fit（存在泄露）",
    y_test,
    pred_A,
    prob_A,
)

evaluate_model(
    "方案 B：仅 X_train fit（正确）",
    y_test,
    pred_B,
    prob_B,
)


# =========================================================
# 8. 最后直接比较
# =========================================================

print("\n========== 最终对比 ==========")

print(
    "预测结果完全一致：",
    np.array_equal(pred_A, pred_B)
)

print(
    "预测概率最大差异：",
    np.max(np.abs(prob_A - prob_B))
)
# ============================================================
# Experiment 2：验证 median() 数据泄漏是否影响最终模型结果
# 注意：仅用于实验，不修改正式项目代码
# ============================================================

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_model(name, model, X_test, y_test):
    """统一计算模型评估指标"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    print(f"\n===== {name} =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"AUC      : {metrics['auc']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")

    return metrics, y_pred, y_prob


# ------------------------------------------------------------
# 1. 读取经过业务清洗、编码后的 X 和 y
# ------------------------------------------------------------

X_raw = load_raw_data()

X_raw = clean_total_charges(X_raw)
X_raw = build_business_features(X_raw)
X_raw = encode_binary_features(X_raw)
X_raw = encode_multi_category(X_raw)

y = X_raw[TARGET_COL]
X = X_raw.drop(columns=DROP_COLS + [TARGET_COL])

print("\n================================================")
print("Experiment 2：median() 泄漏影响验证")
print("================================================")

# ------------------------------------------------------------
# 2. 先划分数据集
#    两个实验必须使用完全相同的 train/test
# ------------------------------------------------------------

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=DEFAULT_SEED,
    stratify=y
)

print(f"训练集：{X_train_raw.shape}")
print(f"测试集：{X_test_raw.shape}")


# ------------------------------------------------------------
# 3. 方案 A：存在 median 泄漏
#
#    median 在完整 X 上计算
#    相当于：
#
#    X_train + X_test
#           ↓
#       计算 median
#           ↓
#       填充缺失值
# ------------------------------------------------------------

X_leak = X.copy()

for col in NUMERIC_COLS:
    if col in X_leak.columns:
        X_leak[col] = pd.to_numeric(
            X_leak[col],
            errors="coerce"
        )

        # ❌ 泄漏：median 使用了完整数据集
        median_value = X_leak[col].median()

        X_leak[col] = X_leak[col].fillna(median_value)

# 使用相同的 train/test 索引
X_train_leak = X_leak.loc[X_train_raw.index]
X_test_leak = X_leak.loc[X_test_raw.index]


# ------------------------------------------------------------
# 4. 方案 B：正确做法
#
#    只使用 X_train 计算 median
#
#    X_train
#       ↓
#    计算 median
#       ↓
#    X_train / X_test 都使用这个 median
# ------------------------------------------------------------

X_train_clean = X_train_raw.copy()
X_test_clean = X_test_raw.copy()

for col in NUMERIC_COLS:
    if col in X_train_clean.columns:

        X_train_clean[col] = pd.to_numeric(
            X_train_clean[col],
            errors="coerce"
        )

        X_test_clean[col] = pd.to_numeric(
            X_test_clean[col],
            errors="coerce"
        )

        # ✅ 正确：median 只从训练集计算
        median_value = X_train_clean[col].median()

        X_train_clean[col] = X_train_clean[col].fillna(
            median_value
        )

        X_test_clean[col] = X_test_clean[col].fillna(
            median_value
        )


# ------------------------------------------------------------
# 5. 检查两种方案填充值是否真的不同
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 6. 两个方案分别训练完全相同的 XGBoost
# ------------------------------------------------------------

common_params = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "n_estimators": 100,
    "random_state": DEFAULT_SEED,
    "eval_metric": "logloss"
}


# A：median 泄漏
model_leak = XGBClassifier(**common_params)

model_leak.fit(
    X_train_leak,
    y_train
)

metrics_leak, pred_leak, prob_leak = evaluate_model(
    "方案 A：median 使用完整数据（存在泄漏）",
    model_leak,
    X_test_leak,
    y_test
)


# B：正确 median
model_clean = XGBClassifier(**common_params)

model_clean.fit(
    X_train_clean,
    y_train
)

metrics_clean, pred_clean, prob_clean = evaluate_model(
    "方案 B：median 只使用训练集（正确）",
    model_clean,
    X_test_clean,
    y_test
)


# ------------------------------------------------------------
# 7. 最终比较
# ------------------------------------------------------------

print("\n================================================")
print("最终结果比较")
print("================================================")

for metric in metrics_leak:

    diff = metrics_clean[metric] - metrics_leak[metric]

    print(
        f"{metric:10s} | "
        f"泄漏={metrics_leak[metric]:.6f} | "
        f"正确={metrics_clean[metric]:.6f} | "
        f"差值={diff:.6f}"
    )


# ------------------------------------------------------------
# 8. 比较每一条测试数据的预测结果
# ------------------------------------------------------------

prediction_same = np.array_equal(
    pred_leak,
    pred_clean
)

max_probability_diff = np.max(
    np.abs(prob_leak - prob_clean)
)

print("\n===== 预测结果比较 =====")
print(f"预测类别完全一致：{prediction_same}")
print(f"预测概率最大差异：{max_probability_diff:.12f}")


if prediction_same and max_probability_diff == 0:
    print("\n结论：")
    print("median 泄漏在当前数据集和当前模型参数下")
    print("没有造成最终预测结果的可观测变化。")
else:
    print("\n结论：")
    print("median 泄漏已经造成模型预测结果发生变化。")