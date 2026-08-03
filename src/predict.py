import os
import joblib
import pandas as pd

from config.settings import (
    get_version_storage_dir,
    DEFAULT_MODEL_VERSION,
    VERSION_MODEL_NAME,
    VERSION_SCALER_NAME,
    VERSION_FEATURE_TXT,
    NUMERIC_COLS
)


def load_model_assets(version: str = DEFAULT_MODEL_VERSION):
    """
    根据版本号加载全套模型资产
    :param version: 模型版本标识，如v1/v2
    :return: model, scaler, feature_list
    """
    model_dir = get_version_storage_dir(version)
    model_path = os.path.join(model_dir, VERSION_MODEL_NAME)
    scaler_path = os.path.join(model_dir, VERSION_SCALER_NAME)
    feature_path = os.path.join(model_dir, VERSION_FEATURE_TXT)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在！路径：{model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化器文件不存在！路径：{scaler_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"特征列表文件不存在！路径：{feature_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path, "r", encoding="utf-8") as f:
        features = f.read().splitlines()
    return model, scaler, features


def predict(input_data: pd.DataFrame, version: str = DEFAULT_MODEL_VERSION):
    """
    用户流失预测接口
    注意：传入的DataFrame必须是【经过特征工程编码之后】的特征，不支持原始csv行数据
    :param input_data: DataFrame，待预测样本
    :param version: 指定使用哪个版本模型
    :return: 预测标签、流失概率
    """
    model, scaler, features = load_model_assets(version)

    # 校验输入特征完整性
    input_cols = set(input_data.columns)
    required_cols = set(features)
    missing_cols = required_cols - input_cols
    if missing_cols:
        raise ValueError(f"预测输入缺失必要特征字段：{sorted(missing_cols)}")

    # 强制对齐训练时特征顺序，避免列错位预测出错
    input_data = input_data[features].copy()

    # 数值特征标准化，必须使用训练阶段保存的scaler.transform
    input_data[NUMERIC_COLS] = scaler.transform(input_data[NUMERIC_COLS])

    # 预测类别 & 流失概率
    pred_label = model.predict(input_data)
    pred_proba = model.predict_proba(input_data)[:, 1]

    return {
        "prediction": int(pred_label[0]),
        "churn_probability": float(pred_proba[0])
    }


if __name__ == "__main__":
    # 自测代码：使用测试集样本进行单条预测验证
    from src.data_process import get_full_dataset
    _, X_test, _, _, _, _ = get_full_dataset()

    # 选取第5条样本作为测试
    test_sample = X_test.iloc[[5]]
    pred_result = predict(test_sample)

    print("===== 单样本预测测试 =====")
    print(f"预测结果: {pred_result['prediction']}")
    print(f"客户流失概率: {pred_result['churn_probability']:.4f}")
    print("0=不流失，1=流失")