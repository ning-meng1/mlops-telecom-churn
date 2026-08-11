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
from src.logger import logger

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

# 训练启动日志（替换print）
logger.info("=" * 70)
logger.info("【模型训练启动】")
logger.info(f"固定随机种子: {RUN_SEED}")
logger.info(f"模型归档版本: {RUN_VERSION}")
logger.info(f"是否开启SHAP绘图: {DRAW_SHAP}")
logger.info("=" * 70)


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
    logger.info("开始加载训练数据集与划分训练/测试集")
    X_train, X_test, y_train, y_test, scaler, features = get_full_dataset(fix_seed=RUN_SEED)
    logger.info(f"数据集加载完成，训练集样本：{X_train.shape[0]}，测试集样本：{X_test.shape[0]}")

    param_grid = ParameterGrid(PARAM_GRID)

    all_result = []
    best_auc = 0.0
    best_model = None
    best_params = None

    logger.info("===== 开始超参数网格遍历训练 =====")
    total_num = len(param_grid)
    logger.info(f"待遍历超参组合总数：{total_num}")

    for idx, param in enumerate(param_grid):
        logger.info(f"\n[{idx + 1}/{total_num}] 当前测试超参：{param}")
        try:
            model, metric = train_single_model(param, X_train, X_test, y_train, y_test)
            all_result.append(metric)
            logger.info(f"本组指标 -> AUC:{metric['auc']} ACC:{metric['accuracy']} F1:{metric['f1']}")

            # 以AUC作为最优模型筛选标准
            if metric["auc"] > best_auc:
                best_auc = metric["auc"]
                best_model = model
                best_params = param
                logger.info(f"更新最优模型，当前最高AUC={best_auc:.4f}")
        except Exception as e:
            logger.error(f"该组参数训练失败，自动跳过，错误信息：{str(e)}", exc_info=True)
            continue

    logger.info(f"\n===== 网格搜索全部完成 =====")
    logger.info(f"最优超参数：{best_params}")
    logger.info(f"最优验证集AUC：{best_auc:.4f}")

    # 创建当前版本独立归档文件夹
    version_dir = get_version_storage_dir(RUN_VERSION)
    logger.info(f"模型版本存储目录：{version_dir}")

    # 可视化图片输出路径
    feat_img_path = os.path.join(ASSETS_DIR, f"xgb_feature_importance_{RUN_VERSION}.png")
    shap_img_path = os.path.join(ASSETS_DIR, f"shap_bar_summary_{RUN_VERSION}.png")
    csv_result_path = os.path.join(ASSETS_DIR, f"hyperparam_search_result_{RUN_VERSION}.csv")

    # 1. 保存最优模型与标准化器
    joblib.dump(best_model, os.path.join(version_dir, VERSION_MODEL_NAME))
    joblib.dump(scaler, os.path.join(version_dir, VERSION_SCALER_NAME))
    logger.info("最优模型、scaler保存完成")

    # 2. 保存特征名称列表，推理阶段保证特征顺序对齐
    with open(os.path.join(version_dir, VERSION_FEATURE_TXT), "w", encoding="utf-8") as f:
        f.write("\n".join(features))
    logger.info("特征列表文件保存完成")

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
    logger.info("训练元信息meta.json保存完成")

    # 4. 保存最优评估指标
    best_metric_record = max(all_result, key=lambda x: x["auc"])
    eval_result_json = {
        "accuracy": best_metric_record["accuracy"],
        "auc": best_metric_record["auc"],
        "f1_score": best_metric_record["f1"]
    }
    with open(os.path.join(version_dir, VERSION_EVAL_JSON), "w", encoding="utf-8") as f:
        json.dump(eval_result_json, f, indent=2)
    logger.info("模型评估指标eval.json保存完成")

    # 5. 保存全部超参搜索结果
    pd.DataFrame(all_result).to_csv(csv_result_path, index=False, encoding="utf-8-sig")
    logger.info(f"超参遍历记录表已保存：{csv_result_path}")

    logger.info(f"\n版本{RUN_VERSION}全套模型文件归档完毕，路径：{version_dir}")

    return best_model, X_test, features, feat_img_path, shap_img_path, csv_result_path, version_dir, eval_result_json


def plot_shap_bar(model, X_test_data, feature_names, save_path):
    """绘制SHAP特征重要性条形图"""
    logger.info("开始生成SHAP特征重要性图")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test_data)
    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_vals, X_test_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"SHAP可视化图已保存：{save_path}")


def plot_xgb_feature_importance(model, feature_names, save_path):
    """绘制XGB原生特征重要性图"""
    logger.info("开始生成XGB原生特征重要性图")
    plt.figure(figsize=(11, 6))
    plt.barh(feature_names, model.feature_importances_)
    plt.xlabel("Feature Importance Score")
    plt.title("XGBoost Native Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"XGB特征重要性图已保存：{save_path}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    logger.info("执行训练主流程 hyperparameter_tuning")
    model, X_test, feat_list, feat_img_save, shap_img_save, csv_path, archive_folder, eval_dict = hyperparameter_tuning()

    # 按需生成可视化图片
    if DRAW_SHAP:
        plot_shap_bar(model, X_test, feat_list, shap_img_save)
        plot_xgb_feature_importance(model, feat_list, feat_img_save)
    else:
        logger.info("未启用SHAP绘图，跳过可视化生成")

    logger.info("\n" + "=" * 70)
    logger.info("✅ 模型训练+调参+版本归档全流程执行完毕！")
    logger.info(f"归档版本号：{RUN_VERSION}")
    logger.info(f"模型存储目录：{archive_folder}")
    logger.info(f"验证集最终指标：{eval_dict}")
    logger.info("外部产出文件：")
    logger.info(f"1. 超参搜索明细CSV：{csv_path}")
    if DRAW_SHAP:
        logger.info(f"2. SHAP重要性图：{shap_img_save}")
        logger.info(f"3. XGB特征重要性图：{feat_img_save}")
    logger.info("版本文件夹内归档资产清单：")
    logger.info("① 最优xgb模型  ② 数据scaler  ③ 特征列表txt  ④ 训练元信息json  ⑤ 评估指标json")
    logger.info("=" * 70)