import sys
sys.path.append('.')

# ========== 标准库导入 ==========
import argparse
import joblib
import os
import json
import random
from datetime import datetime

# ========== 第三方数据分析库导入 ==========
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# ========== 项目内部模块导入 ==========
from config.settings import (
    DEFAULT_SEED, DEFAULT_MODEL_VERSION,
    TRAIN_CONFIG, PARAM_GRID,
    ASSETS_DIR,
    get_version_storage_dir,
    VERSION_MODEL_NAME,
    VERSION_SCALER_NAME,
    VERSION_FEATURE_TXT,
    VERSION_META_JSON,
    VERSION_EVAL_JSON
)
from src.data_process import get_full_dataset

# ===========命令行参数解析===============
parser = argparse.ArgumentParser(description="电信用户流失XGBoost训练脚本")
# 默认值从settings读取，杜绝硬编码
parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="全局随机种子，保障实验可复现")
parser.add_argument("--version", type=str, default=DEFAULT_MODEL_VERSION, help="模型版本标记，例：v1/v2")
parser.add_argument("--enable_shap", action="store_true", help="开启SHAP可解释性绘图，关闭节省算力")

args = parser.parse_args()
# 全局运行参数
RUN_SEED = args.seed
RUN_VERSION = args.version
DRAW_SHAP = args.enable_shap

# ===================== 全链路固定随机种子，确保实验可复现 =====================
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)
os.environ['PYTHONHASHSEED'] = str(RUN_SEED)

# 训练启动日志
print("=" * 70)
print(f"【模型训练启动日志】")
print(f"固定随机种子: {RUN_SEED}")
print(f"模型归档版本: {RUN_VERSION}")
print(f"是否开启SHAP绘图: {DRAW_SHAP}")
print("=" * 70)


def train_single_model(params, X_train, X_test, y_train, y_test):
    """
    单组超参训练XGBoost模型
    :param params: XGB超参字典
    :return: 训练完成模型 + 测试集评估指标
    """
    model = XGBClassifier(
        random_state=RUN_SEED,
        eval_metric=TRAIN_CONFIG.get("eval_metric", "logloss"),
        scale_pos_weight=TRAIN_CONFIG.get("scale_pos_weight", 1),
        **params
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "params": params,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_pred_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4)
    }
    return model, metrics


def hyperparameter_tuning():
    """网格搜索遍历超参数，筛选最优模型并归档全部产物"""
    X_train, X_test, y_train, y_test, scaler, features = get_full_dataset(fix_seed=RUN_SEED)
    param_grid = ParameterGrid(PARAM_GRID)

    all_result = []
    best_auc = 0.0
    best_model = None
    best_params = None

    print("===== 开始超参数网格遍历训练 =====")
    total_num = len(param_grid)

    for idx, param in enumerate(param_grid):
        print(f"\n[{idx + 1}/{total_num}] 当前测试超参：{param}")
        try:
            model, metric = train_single_model(param, X_train, X_test, y_train, y_test)
            all_result.append(metric)
            print(f"指标 -> AUC:{metric['auc']} ACC:{metric['accuracy']} F1:{metric['f1']}")

            # 以AUC作为最优模型筛选标准
            if metric["auc"] > best_auc:
                best_auc = metric["auc"]
                best_model = model
                best_params = param
        except Exception as e:
            print(f"⚠️ 该组参数训练失败，自动跳过，错误信息：{str(e)}")
            continue

    print(f"\n===== 网格搜索全部完成 =====")
    print(f"最优超参数：{best_params}")
    print(f"最优验证集AUC：{best_auc:.4f}")

    # 创建当前版本独立归档文件夹
    version_dir = get_version_storage_dir(RUN_VERSION)

    # 可视化图片输出路径
    feat_img_path = os.path.join(ASSETS_DIR, f"xgb_feature_importance_{RUN_VERSION}.png")
    shap_img_path = os.path.join(ASSETS_DIR, f"shap_bar_summary_{RUN_VERSION}.png")
    csv_result_path = os.path.join(ASSETS_DIR, f"hyperparam_search_result_{RUN_VERSION}.csv")

    # 1. 保存最优模型与标准化器
    joblib.dump(best_model, os.path.join(version_dir, VERSION_MODEL_NAME))
    joblib.dump(scaler, os.path.join(version_dir, VERSION_SCALER_NAME))

    # 2. 保存特征名称列表，推理阶段保证特征顺序对齐
    with open(os.path.join(version_dir, VERSION_FEATURE_TXT), "w", encoding="utf-8") as f:
        f.write("\n".join(features))

    # 3. 写入训练元数据，用于实验追溯、复现
    meta_info = {
        "version": RUN_VERSION,
        "seed": RUN_SEED,
        "train_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_split_ratio": TRAIN_CONFIG.get("test_size", 0.2),
        "best_params": best_params,
        "total_search_count": total_num,
        "enable_shap_plot": DRAW_SHAP
    }
    with open(os.path.join(version_dir, VERSION_META_JSON), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    # 4. 保存最优评估指标
    best_metric_record = max(all_result, key=lambda x: x["auc"])
    eval_result_json = {
        "accuracy": best_metric_record["accuracy"],
        "auc": best_metric_record["auc"],
        "f1_score": best_metric_record["f1"]
    }
    with open(os.path.join(version_dir, VERSION_EVAL_JSON), "w", encoding="utf-8") as f:
        json.dump(eval_result_json, f, indent=2)

    # 5. 保存全部超参搜索结果
    pd.DataFrame(all_result).to_csv(csv_result_path, index=False, encoding="utf-8-sig")

    print(f"\n版本{RUN_VERSION}全套模型文件已归档至：{version_dir}")
    print(f"超参遍历记录表已保存：{csv_result_path}")

    return best_model, X_test, features, feat_img_path, shap_img_path, csv_result_path, version_dir, eval_result_json


def plot_shap_bar(model, X_test_data, feature_names, save_path):
    """绘制SHAP特征重要性条形图"""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test_data)
    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_vals, X_test_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ SHAP可视化图已保存：{save_path}")


def plot_xgb_feature_importance(model, feature_names, save_path):
    """绘制XGB原生特征重要性图"""
    plt.figure(figsize=(11, 6))
    plt.barh(feature_names, model.feature_importances_)
    plt.xlabel("Feature Importance Score")
    plt.title("XGBoost Native Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ XGB特征重要性图已保存：{save_path}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    model, X_test, feat_list, feat_img_save, shap_img_save, csv_path, archive_folder, eval_dict = hyperparameter_tuning()

    # 按需生成可视化图片
    if DRAW_SHAP:
        plot_shap_bar(model, X_test, feat_list, shap_img_save)
        plot_xgb_feature_importance(model, feat_list, feat_img_save)
    else:
        print("\nℹ️ 未启用SHAP绘图，跳过可视化生成")

    print("\n" + "=" * 70)
    print("✅ 模型训练+调参+版本归档全流程执行完毕！")
    print(f"归档版本号：{RUN_VERSION}")
    print(f"模型存储目录：{archive_folder}")
    print(f"验证集最终指标：{eval_dict}")
    print("外部产出文件：")
    print(f"1. 超参搜索明细CSV：{csv_path}")
    if DRAW_SHAP:
        print(f"2. SHAP重要性图：{shap_img_save}")
        print(f"3. XGB特征重要性图：{feat_img_save}")
    print("版本文件夹内归档资产清单：")
    print("① 最优xgb模型  ② 数据scaler  ③ 特征列表txt  ④ 训练元信息json  ⑤ 评估指标json")
    print("=" * 70)