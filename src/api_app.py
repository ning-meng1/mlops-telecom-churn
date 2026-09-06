from flask import Flask, request, jsonify

from src.logger import logger

from src.service import (
    predict_single,
    predict_batch_service,
    init_model,
    is_model_loaded,
    get_model_version
)


# =========================
# Flask应用初始化
# =========================

app = Flask(__name__)


# =========================
# 统一响应
# =========================

def make_response(code, msg, data=None):
    """
    统一API响应格式。
    """

    return jsonify({
        "code": code,
        "msg": msg,
        "data": data if data is not None else {}
    })


# =========================
# 单样本预测接口
# =========================

@app.route(
    "/predict",
    methods=["POST"]
)
def api_predict():
    """
    单样本用户流失预测接口。

    API层只负责：
    1. 接收HTTP请求
    2. 检查请求结构
    3. 调用service层
    4. 返回HTTP响应
    """

    # -------------------------
    # 模型状态检查
    # -------------------------

    if not is_model_loaded():

        return make_response(
            503,
            "模型未加载"
        )

    try:

        # -------------------------
        # JSON解析
        # -------------------------

        req_json = request.get_json(
            silent=True
        )

        logger.info(
            f"收到单样本预测请求: {req_json}"
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
        # 调用Service层
        # -------------------------

        result = predict_single(
            features
        )

        return make_response(
            0,
            "success",
            result
        )

    # -------------------------
    # 参数错误
    # -------------------------

    except ValueError as e:

        logger.warning(
            f"单样本请求参数错误: {str(e)}"
        )

        return make_response(
            400,
            str(e)
        )

    # -------------------------
    # 未预期异常
    # -------------------------

    except Exception:

        logger.exception(
            "单样本预测过程中发生未预期异常"
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
    """
    批量用户流失预测接口。
    """

    # -------------------------
    # 模型状态检查
    # -------------------------

    if not is_model_loaded():

        return make_response(
            503,
            "模型未加载"
        )

    try:

        # -------------------------
        # JSON解析
        # -------------------------

        req_json = request.get_json(
            silent=True
        )

        logger.info(
            f"收到批量预测请求: {req_json}"
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
        # 调用Service层
        # -------------------------

        result = predict_batch_service(
            features_list
        )

        return make_response(
            0,
            "success",
            result
        )

    # -------------------------
    # 参数错误
    # -------------------------

    except ValueError as e:

        logger.warning(
            f"批量预测请求参数错误: {str(e)}"
        )

        return make_response(
            400,
            str(e)
        )

    # -------------------------
    # 未预期异常
    # -------------------------

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
    """
    服务健康检查。
    """

    return make_response(
        0,
        "service ok",
        {
            "service_status": "running",
            "model_version": get_model_version(),
            "model_loaded": is_model_loaded()
        }
    )


# =========================
# Flask全局异常处理
# =========================

@app.errorhandler(Exception)
def handle_global_exception(e):
    """
    全局异常兜底。

    对外隐藏详细异常信息，
    完整堆栈写入日志。
    """

    logger.exception(
        "未捕获的全局异常"
    )

    return make_response(
        500,
        "服务器内部异常"
    )

@app.route("/test_error", methods=["GET"])
def test_error():
    raise Exception("Day24测试异常")

# =========================
# 本地开发启动
# =========================

if __name__ == "__main__":

    # 如果模型没有成功加载，
    # 尝试重新初始化一次
    if not is_model_loaded():

        logger.warning(
            "检测到模型未加载，尝试重新加载"
        )

        init_model()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )

