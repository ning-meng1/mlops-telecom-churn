import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier

from src.data_process import get_full_dataset


# =========================
# 1. 固定实验参数
# =========================
SEED = 42
SCALE_POS_WEIGHT = 2.7683

# 使用和 v2 完全一样的最优超参数
BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "n_estimators": 100
}


# =========================
# 2. 获取与 v2 完全一致的数据划分
# =========================
X_train, X_test, y_train, y_test, scaler, features = get_full_dataset(
    fix_seed=SEED
)


# =========================
# 3. 训练加权实验模型
# =========================
model = XGBClassifier(
    random_state=SEED,
    eval_metric="logloss",
    scale_pos_weight=SCALE_POS_WEIGHT,
    **BEST_PARAMS
)

model.fit(X_train, y_train)


# =========================
# 4. 测试集预测
# =========================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


# =========================
# 5. 计算指标
# =========================
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# =========================
# 6. 输出实验结果
# =========================
print("\n==============================")
print("scale_pos_weight 加权实验结果")
print("==============================")
print(f"scale_pos_weight: {SCALE_POS_WEIGHT}")
print(f"accuracy:  {accuracy:.4f}")
print(f"auc:       {auc:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall:    {recall:.4f}")
print(f"f1:        {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\n==============================")