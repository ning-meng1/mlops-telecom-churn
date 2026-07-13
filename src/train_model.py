import sys
sys.path.append('.')

import joblib
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import ParameterGrid
from src.data_process import load_and_process_data
from config.settings import (
    MODEL_SAVE_PATH, BASE_DIR, TRAIN_CONFIG,
    SEED, PARAM_GRID, SCALER_SAVE_PATH
)
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# 固定所有随机种子，保证训练可复现
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# ---------------------- Day5 调参核心 ----------------------
def train_single_model(params, X_train, X_test, y_train, y_test, features):
    """用一组参数训练模型，传入提前加载好的数据，避免重复读取"""
    model = XGBClassifier(
        random_state=TRAIN_CONFIG["random_state"],
        eval_metric=TRAIN_CONFIG["eval_metric"],
        **params
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "params": params,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred)
    }
    return model, metrics


def hyperparameter_tuning():
    # 只加载一次数据，优化速度
    X_train, X_test, y_train, y_test, scaler, features = load_and_process_data()
    grid = ParameterGrid(PARAM_GRID)
    results = []
    best_auc = 0
    best_model = None
    best_params = None

    print("=== 开始超参数调优 ===")
    for idx, params in enumerate(grid):
        print(f"\n[{idx + 1}/{len(grid)}] 测试参数：{params}")
        model, metrics = train_single_model(params, X_train, X_test, y_train, y_test, features)
        results.append(metrics)
        print(f"  AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_model = model
            best_params = params

    print(f"\n=== 调优完成！最优参数：{best_params} ===")
    print(f"最优AUC: {best_auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"✅ 最优模型已保存：{MODEL_SAVE_PATH}")
    print(f"✅ 标准化器已保存：{SCALER_SAVE_PATH}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(BASE_DIR, "assets/hyperparam_results.csv"), index=False)
    print("✅ 调参记录已保存到 assets/hyperparam_results.csv")

    return best_model, X_test, features


# ---------------------- Day6 可视化核心 ----------------------
def plot_shap_summary(model, X_test, features):
    print("\n🔍 正在生成SHAP特征重要性图...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "assets/shap_importance.png"))
    plt.close()
    print("✅ SHAP特征重要性图已保存到 assets 目录")


def plot_feature_importance(model, features):
    print("\n📊 正在生成XGBoost特征重要性图...")
    plt.figure(figsize=(10, 6))
    plt.barh(features, model.feature_importances_)
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "assets/xgb_feature_importance.png"))
    plt.close()
    print("✅ XGBoost特征重要性图已保存到 assets 目录")


if __name__ == "__main__":
    model, X_test, features = hyperparameter_tuning()
    plot_shap_summary(model, X_test, features)
    plot_feature_importance(model, features)

    print("\n🎉 Day5-Day6 任务全部完成！")
    print("检查以下文件是否生成：")
    print("- assets/hyperparam_results.csv（调参记录）")
    print("- assets/shap_importance.png（SHAP图）")
    print("- assets/xgb_feature_importance.png（特征重要性图）")
    print("- models/model_v1.pkl / scaler.pkl（模型文件）")
