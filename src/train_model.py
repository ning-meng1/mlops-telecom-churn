import sys
sys.path.append('.')

# ========== 标准库导入 ==========
import argparse
import joblib
import os
import json
import random
import mlflow
import mlflow.xgboost
from datetime import datetime

# ========== 第三方数据分析库导入 ==========
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use('Agg')  # 后台无GUI稳定绘图，工业环境必加
import matplotlib.pyplot as plt

# ========== 项目内部模块导入 ==========
from config.settings import (
    DEFAULT_SEED, DEFAULT_MODEL_VERSION,
    TRAIN_CONFIG, PARAM_GRID,
    ASSETS_DIR, CLEAN_DATA_PATH, TARGET_COL,
    get_version_storage_dir,
    VERSION_MODEL_NAME,
    VERSION_SCALER_NAME,
    VERSION_FEATURE_TXT,
    VERSION_META_JSON,
    VERSION_EVAL_JSON,
    VERSION_MEDIAN_NAME
)
from src.data_process import encode_multi_category, encode_binary_features
from src.logger import logger
from src.validation_feature import run_all_validation

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

# 训练前门禁B校验
ok, report = run_all_validation()
if not ok:
    print("特征集校验失败，终止训练！")
    exit(1)


def load_and_split_clean_data(fix_seed: int, test_index_save_path: str):
    """
    方案B：读取v2版本清洗数据集，在训练脚本内完成编码、标准化、分层划分
    划分规则：
    1. 先切20%测试集，测试集索引持久化保存
    2. 剩余80%，再分层切分为训练集(70%)、验证集(30%)
    对应全局比例：train=0.56, val=0.24, test=0.2
    """
    # 读取版本化的clean数据集（已经清洗、衍生特征、Churn标签0/1，分类字段保留原始文本）
    df = pd.read_csv(CLEAN_DATA_PATH)
    logger.info(f"加载v2清洗数据集 shape={df.shape}")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL, "customerID"])

    # ========== 补上二元编码！这一行是缺失的关键 ==========
    X = encode_binary_features(X)
    # 训练阶段才做多分类独热编码（方案B约定）
    X = encode_multi_category(X)
    feature_cols = list(X.columns)

    # 第一步：拆分出20%测试集，分层抽样，保证标签分布一致
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=fix_seed, stratify=y
    )
    # 保存测试集索引，固定下来，后续复现、离线验证都用同一批测试样本
    test_idx = X_test.index
    np.save(test_index_save_path, test_idx)
    logger.info(f"测试集索引已持久化保存：{test_index_save_path}")

    # 第二步：剩下80%里面，再划分训练、验证
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=fix_seed, stratify=y_temp
    )

    # 标准化：只拿训练集拟合scaler，防止泄露
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 中位数填充字典（和原逻辑保持一致，兼容缺失兜底）
    median_dict = {}
    for col in feature_cols:
        col_idx = feature_cols.index(col)
        median_dict[col] = np.median(X_train[:, col_idx])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, median_dict, feature_cols


def train_single_model(params, X_train, X_val, y_train, y_val):
    """
    单组超参训练XGBoost模型
    :param params: XGB超参字典
    :return: 训练完成模型 + 验证集评估指标
    """
    model = XGBClassifier(
        random_state=RUN_SEED,
        eval_metric=TRAIN_CONFIG.get("eval_metric", "logloss"),
        scale_pos_weight=TRAIN_CONFIG.get("scale_pos_weight", 1),
        early_stopping_rounds=20,
        **params
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "params": params,
        "accuracy_val": round(accuracy_score(y_val, y_val_pred), 4),
        "auc_val": round(roc_auc_score(y_val, y_val_pred_proba), 4),
        "f1_val": round(f1_score(y_val, y_val_pred), 4)
    }
    return model, metrics


def hyperparameter_tuning():
    """网格搜索遍历超参数，筛选最优模型并归档全部产物"""
    logger.info("开始加载训练数据集与划分训练/测试集/验证集")
    # 测试集索引文件路径，跟着模型版本走
    test_idx_file = os.path.join(ASSETS_DIR, f"test_index_{RUN_VERSION}.npy")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, median_dict, feature_cols = load_and_split_clean_data(
        fix_seed=RUN_SEED,
        test_index_save_path=test_idx_file
    )
    logger.info(f"数据集加载完成，训练集样本：{X_train.shape[0]}，测试集样本：{X_test.shape[0]}")

    param_grid = ParameterGrid(PARAM_GRID)

    all_result = []
    best_auc = 0.0
    best_model = None
    best_params = None

    # 生成本轮网格搜索统一批次标记，用来给所有run打tag
    grid_batch_id = f"{RUN_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"本轮网格搜索batch标记：{grid_batch_id}")

    logger.info("===== 开始超参数网格遍历训练 =====")
    total_num = len(param_grid)
    logger.info(f"待遍历超参组合总数：{total_num}")

    for idx, param in enumerate(param_grid):
        logger.info(f"\n[{idx + 1}/{total_num}] 当前测试超参：{param}")
        try:
            # 每一组参数单独一个mlflow run
            with mlflow.start_run():
                # 打上标签，方便后续筛选同一批实验
                mlflow.set_tag("grid_batch_id", grid_batch_id)
                mlflow.set_tag("model_version", RUN_VERSION)
                mlflow.set_tag("seed", RUN_SEED)
                mlflow.set_tag("run_index", idx + 1)

                # 记录当前这一组超参数
                mlflow.log_params(param)

                model, metric = train_single_model(param, X_train, X_val, y_train, y_val)
                all_result.append(metric)

                # 记录验证集指标
                mlflow.log_metric("auc_val", metric["auc_val"])
                mlflow.log_metric("accuracy_val", metric["accuracy_val"])
                mlflow.log_metric("f1_val", metric["f1_val"])

            logger.info(f"本组指标 -> AUC:{metric['auc_val']} ACC:{metric['accuracy_val']} F1:{metric['f1_val']}")

            # 以AUC作为最优模型筛选标准
            if metric["auc_val"] > best_auc:
                best_auc = metric["auc_val"]
                best_model = model
                best_params = param
                logger.info(f"更新最优模型，当前最高AUC={best_auc:.4f}")

        except Exception as e:
            logger.error(f"该组参数训练失败，自动跳过，错误信息：{str(e)}", exc_info=True)
            continue

    logger.info(f"\n===== 网格搜索全部完成 =====")
    logger.info(f"最优超参数：{best_params}")
    #验证集测试完成后开始进入测试集环节
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    test_acc = round(accuracy_score(y_test, y_test_pred), 4)
    test_auc = round(roc_auc_score(y_test, y_test_proba), 4)
    test_f1 = round(f1_score(y_test, y_test_pred), 4)

    logger.info(f"测试集指标 | acc={test_acc}, auc={test_auc}, f1={test_f1}")
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
    joblib.dump(median_dict, os.path.join(version_dir, VERSION_MEDIAN_NAME))
    logger.info("最优模型、scaler保存完成")

    # 2. 保存特征名称列表，推理阶段保证特征顺序对齐
    with open(os.path.join(version_dir, VERSION_FEATURE_TXT), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))
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
    eval_result_json = {
        "accuracy": test_acc,
        "auc": test_auc,
        "f1_score": test_f1
    }
    with open(os.path.join(version_dir, VERSION_EVAL_JSON), "w", encoding="utf-8") as f:
        json.dump(eval_result_json, f, indent=2)
    logger.info("模型评估指标eval.json保存完成")

    # 5. 保存全部超参搜索结果
    pd.DataFrame(all_result).to_csv(csv_result_path, index=False, encoding="utf-8-sig")
    logger.info(f"超参遍历记录表已保存：{csv_result_path}")

    logger.info(f"\n版本{RUN_VERSION}全套模型文件归档完毕，路径：{version_dir}")

    # ========== 单独一条run，记录最终测试集指标（不和调参循环混在一起） ==========
    with mlflow.start_run(run_name="best_model-final-test"):
        mlflow.set_tag("grid_batch_id", grid_batch_id)
        mlflow.set_tag("model_version", RUN_VERSION)
        mlflow.set_tag("seed", RUN_SEED)
        mlflow.set_tag("run_type", "best_final")
        mlflow.set_tag("exp_group","C_weight_5")
        mlflow.log_params(best_params)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_artifact(os.path.join(version_dir, VERSION_MODEL_NAME))
        mlflow.log_artifact(os.path.join(version_dir, VERSION_SCALER_NAME))
        mlflow.log_artifact(os.path.join(version_dir, VERSION_MEDIAN_NAME))
        mlflow.log_artifact(os.path.join(version_dir, VERSION_FEATURE_TXT))
        mlflow.log_artifact(os.path.join(version_dir, VERSION_EVAL_JSON))
        mlflow.log_artifact(os.path.join(version_dir, VERSION_META_JSON))

        # ===== 先画图、保存，再上传=====
        if DRAW_SHAP:
            plot_shap_bar(best_model, X_test, feature_cols, shap_img_path)
            plot_xgb_feature_importance(best_model, feature_cols, feat_img_path)
            mlflow.log_artifact(feat_img_path, artifact_path="plots")
            mlflow.log_artifact(shap_img_path, artifact_path="plots")

    logger.info("✅ 测试集指标已经单独写入MLflow一条独立run")

    return best_model, X_test, feature_cols, feat_img_path, shap_img_path, csv_result_path, version_dir, eval_result_json


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
    # 档案柜：所有实验都放在这个实验名下
    mlflow.set_experiment("电信用户流失预测实验")

    # 直接调用，不再外层包裹mlflow start_run
    model, X_test, feat_list, feat_img_save, shap_img_save, csv_path, archive_folder, eval_dict = hyperparameter_tuning()


    logger.info("\n" + "=" * 70)
    logger.info("✅ 模型训练+调参+版本归档全流程执行完毕！")
    logger.info(f"归档版本号：{RUN_VERSION}")
    logger.info(f"模型存储目录：{archive_folder}")
    logger.info(f"测试集最终指标：{eval_dict}")
    logger.info("外部产出文件：")
    logger.info(f"1. 超参搜索明细CSV：{csv_path}")
    if DRAW_SHAP:
        logger.info(f"2. SHAP重要性图：{shap_img_save}")
        logger.info(f"3. XGB特征重要性图：{feat_img_save}")
    logger.info("版本文件夹内归档资产清单：")
    logger.info("① 最优xgb模型  ② 数据scaler  ③ 特征列表txt  ④ 训练元信息json  ⑤ 评估指标json")
    logger.info("=" * 70)
