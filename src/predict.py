import os
import joblib
import pandas as pd
import gc
import json

from config.settings import (
    get_version_storage_dir,
    DEFAULT_MODEL_VERSION,
    VERSION_MODEL_NAME,
    VERSION_SCALER_NAME,
    VERSION_FEATURE_TXT,
    VERSION_META_JSON,
    NUMERIC_COLS
)
# ✅导入推理转换函数
from src.data_process import raw_data_transform
from src.logger import logger

# Day16 全局缓存字典，同一个版本模型只加载一次
_MODEL_CACHE = {}


def load_model_assets(version: str = DEFAULT_MODEL_VERSION):
    if version in _MODEL_CACHE:
        logger.info(f"版本 {version} 模型已存在全局缓存，直接复用")
        return _MODEL_CACHE[version]

    model_dir = get_version_storage_dir(version)
    model_path = os.path.join(model_dir, VERSION_MODEL_NAME)
    scaler_path = os.path.join(model_dir, VERSION_SCALER_NAME)
    feature_path = os.path.join(model_dir, VERSION_FEATURE_TXT)
    meta_path = os.path.join(model_dir, VERSION_META_JSON)

    logger.info(f"开始加载版本 {version} 模型资产，目录：{model_dir}")

    if not os.path.exists(model_path):
        err_msg = f"模型文件不存在：{model_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    if not os.path.exists(scaler_path):
        err_msg = f"标准化器文件不存在：{scaler_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    if not os.path.exists(feature_path):
        err_msg = f"特征列表文件不存在：{feature_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    if not os.path.exists(meta_path):
        err_msg = f"元数据文件不存在：{meta_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"开始载入模型、Scaler、特征配置文件")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path, "r", encoding="utf-8") as f:
        feature_cols = f.read().splitlines()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _MODEL_CACHE[version] = (model, scaler, feature_cols, meta)
    logger.info(f"✅ 版本 {version} 模型、Scaler已全局一次性加载缓存完成，特征总数：{len(feature_cols)}")
    return model, scaler, feature_cols, meta


def predict(input_data: pd.DataFrame, version: str = DEFAULT_MODEL_VERSION):
    """
    单样本预测
    input_data：【原始业务字段】DataFrame（未做one‑hot编码）
    """
    logger.info(f"收到单样本推理请求，模型版本：{version}")
    model, scaler, feature_cols, _ = load_model_assets(version)

    # --------关键：执行和训练一模一样清洗+独热编码--------
    df_transformed = raw_data_transform(input_data)
    logger.info(f"单样本数据预处理完成，对齐训练特征列表")

    # 对齐模型训练时全部特征列，缺失的独热列自动补0
    df_transformed = df_transformed.reindex(columns=feature_cols, fill_value=0)

    # 标准化数值列
    df_transformed[NUMERIC_COLS] = scaler.transform(df_transformed[NUMERIC_COLS])

    pred_label = model.predict(df_transformed)
    pred_prob = model.predict_proba(df_transformed)[:, 1]
    logger.info(f"单样本推理完成，预测流失概率={round(float(pred_prob[0]),4)}")

    del df_transformed, input_data
    gc.collect()

    return {
        "pred_class": int(pred_label[0]),
        "confidence": round(float(pred_prob[0]), 4),
        "model_version": version,
        "label_desc": "0=不流失，1=流失"
    }


def predict_batch(input_data: pd.DataFrame, version: str = DEFAULT_MODEL_VERSION):
    """
    批量预测
    input_data：【原始业务字段】DataFrame，多行原始样本
    """
    sample_cnt = len(input_data)
    logger.info(f"收到批量推理请求，样本数量：{sample_cnt}，模型版本：{version}")
    model, scaler, feature_cols, _ = load_model_assets(version)

    df_transformed = raw_data_transform(input_data)
    df_transformed = df_transformed.reindex(columns=feature_cols, fill_value=0)
    df_transformed[NUMERIC_COLS] = scaler.transform(df_transformed[NUMERIC_COLS])

    pred_label = model.predict(df_transformed).tolist()
    pred_prob = model.predict_proba(df_transformed)[:,1].tolist()
    logger.info(f"{sample_cnt}条批量样本推理计算完成")

    del df_transformed, input_data
    gc.collect()

    return {
        "pred_label": [int(x) for x in pred_label],
        "pred_proba": [round(float(p),4) for p in pred_prob]
    }

# 本地自测入口
if __name__ == "__main__":
    logger.info("进入predict模块本地自测流程")
    from src.data_process import load_raw_data
    df_raw = load_raw_data()
    test_sample_raw = df_raw.iloc[[5]]
    res = predict(test_sample_raw)
    logger.info("单条原始样本自测结果：%s", res)

    batch_sample_raw = df_raw.iloc[10:13]
    batch_res = predict_batch(batch_sample_raw)
    logger.info("批量原始样本自测结果：%s", batch_res)