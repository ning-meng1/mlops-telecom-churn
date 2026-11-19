import pandas as pd

from src.logger import logger

from src.predict import (
    predict,
    predict_batch,
    load_model_assets
)

from src.validator import validate_features

from config.settings import DEFAULT_MODEL_VERSION
# =========================
# 模型服务配置
# =========================

MODEL_VERSION = DEFAULT_MODEL_VERSION
MODEL_LOADED = False
# =========================
# 模型预加载
# =========================
def init_model():
    """
    服务启动时加载模型。
    返回：
        True  -> 模型加载成功
        False -> 模型加载失败
    """

    global MODEL_LOADED

    try:

        load_model_assets(
            MODEL_VERSION
        )

        MODEL_LOADED = True

        logger.info(
            f"模型加载成功: {MODEL_VERSION}"
        )

        return True
    except Exception:

        MODEL_LOADED = False

        logger.exception(
            f"模型加载失败: {MODEL_VERSION}"
        )

        return False
# =========================
# 模型状态
# =========================
def is_model_loaded():
    """
    返回当前模型是否加载成功。
    """
    return MODEL_LOADED
def get_model_version():
    """
    返回当前模型版本。
    """
    return MODEL_VERSION
# =========================
# 单样本预测业务
# =========================

def predict_single(features):
    """
    单样本预测业务服务。

    API层不直接操作validator、DataFrame、predict，
    所有业务流程统一由service层处理。

    :param features: 原始业务字段字典
    :return: 模型预测结果
    """

    # -------------------------
    # 参数校验
    # -------------------------

    error = validate_features(
        features
    )

    if error:

        logger.warning(
            f"单样本参数校验失败: {error}"
        )

        raise ValueError(error)

    # -------------------------
    # 构造DataFrame
    # -------------------------

    raw_df = pd.DataFrame(
        [features]
    )

    # -------------------------
    # 模型预测
    # -------------------------

    result = predict(
        raw_df,
        MODEL_VERSION
    )

    logger.info(
        f"单样本业务预测完成: {result}"
    )

    return result


# =========================
# 批量预测业务
# =========================

def predict_batch_service(features_list):
    """
    批量预测业务服务。

    :param features_list: 原始业务字段字典列表
    :return: 批量预测结果
    """

    # -------------------------
    # 基础校验
    # -------------------------

    if not isinstance(
        features_list,
        list
    ):
        raise ValueError(
            "features_list必须数组"
        )

    if len(features_list) == 0:
        raise ValueError(
            "预测数据不能为空"
        )

    # -------------------------
    # 逐条参数校验
    # -------------------------

    for index, item in enumerate(
        features_list
    ):

        error = validate_features(
            item
        )

        if error:

            logger.warning(
                f"第{index}条数据参数校验失败: {error}"
            )

            raise ValueError(
                f"第{index}条数据错误: {error}"
            )

    # -------------------------
    # 构造DataFrame
    # -------------------------

    raw_df = pd.DataFrame(
        features_list
    )

    # -------------------------
    # 批量预测
    # -------------------------

    result = predict_batch(
        raw_df,
        MODEL_VERSION
    )

    logger.info(
        f"批量业务预测完成，样本量: {len(features_list)}"
    )

    return result

