import sys
import pandas as pd

from flask import Flask, request, jsonify

sys.path.append(".")

from src.logger import logger

from src.predict import (
    predict,
    predict_batch,
    load_model_assets
)

from src.validator import validate_features

from config.settings import (
    DEFAULT_MODEL_VERSION
)


app = Flask(__name__)


MODEL_VERSION = DEFAULT_MODEL_VERSION


def make_response(code, msg, data=None):
    return jsonify({
        "code": code,
        "msg": msg,
        "data": data if data is not None else {}
    })


# =========================
# 模型预加载
# =========================

MODEL_LOADED = False

try:

    load_model_assets(
        MODEL_VERSION
    )

    MODEL_LOADED = True

    logger.info(
        f"模型加载成功: {MODEL_VERSION}"
    )

except Exception:

    logger.exception(
        f"模型加载失败: {MODEL_VERSION}"
    )


# =========================
# 单预测接口
# =========================

@app.route(
    "/predict",
    methods=["POST"]
)
def api_predict():

    if not MODEL_LOADED:

        return make_response(
            503,
            "模型未加载"
        )

    try:

        req_json = request.get_json(
            silent=True
        )

        logger.info(
            f"收到预测请求: {req_json}"
        )

        # -------------------------
        # 请求体校验
        # -------------------------

        if not isinstance(
            req_json,
            dict
        ):

            return make_response(
                400,
                "请求体必须JSON对象"
            )

        # -------------------------
        # features字段校验
        # -------------------------

        if "features" not in req_json:

            return make_response(
                400,
                "缺少features字段"
            )

        features = req_json["features"]

        # -------------------------
        # 参数校验
        # -------------------------

        error = validate_features(
            features
        )

        if error:

            logger.warning(
                f"参数校验失败: {error}"
            )

            return make_response(
                400,
                error
            )

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

        return make_response(
            0,
            "success",
            result
        )

    except ValueError as e:

        return make_response(
            400,
            str(e)
        )

    except Exception:

        logger.exception(
            "预测过程中发生未预期异常"
        )

        return make_response(
            500,
            "服务器内部异常"
        )


# =========================
# 批量预测接口
# =========================

@app.route(
    "/batch_predict",
    methods=["POST"]
)
def api_batch_predict():

    if not MODEL_LOADED:

        return make_response(
            503,
            "模型未加载"
        )

    try:

        req_json = request.get_json(
            silent=True
        )

        # -------------------------
        # 请求体校验
        # -------------------------

        if not isinstance(
            req_json,
            dict
        ):

            return make_response(
                400,
                "请求体必须JSON对象"
            )

        # -------------------------
        # features_list字段校验
        # -------------------------

        if "features_list" not in req_json:

            return make_response(
                400,
                "缺少features_list字段"
            )

        features_list = req_json[
            "features_list"
        ]

        # -------------------------
        # 类型校验
        # -------------------------

        if not isinstance(
            features_list,
            list
        ):

            return make_response(
                400,
                "features_list必须数组"
            )

        # -------------------------
        # 空数组校验
        # -------------------------

        if len(features_list) == 0:

            return make_response(
                400,
                "预测数据不能为空"
            )

        # -------------------------
        # 每条数据校验
        # -------------------------

        for index, item in enumerate(
            features_list
        ):

            error = validate_features(
                item
            )

            if error:

                return make_response(
                    400,
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

        return make_response(
            0,
            "success",
            result
        )

    except Exception:

        logger.exception(
            "批量预测过程中发生未预期异常"
        )

        return make_response(
            500,
            "批量预测异常"
        )


# =========================
# 健康检查
# =========================

@app.route(
    "/health",
    methods=["GET"]
)
def health_check():

    return make_response(
        0,
        "service ok",
        {
            "service_status": "running",
            "model_version": MODEL_VERSION,
            "model_loaded": MODEL_LOADED
        }
    )


# =========================
# 全局异常处理
# =========================

@app.errorhandler(Exception)
def handle_global_exception(e):

    logger.exception(
        "未捕获的全局异常"
    )

    return make_response(
        500,
        "服务器内部异常"
    )


# =========================
# 启动服务
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )