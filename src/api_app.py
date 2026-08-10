import sys
import logging

# 将当前目录加入python模块搜索路径，保证config、src模块可以正常导入
sys.path.append(".")

from flask import Flask, request, jsonify
import pandas as pd

# 导入预测业务函数：单样本预测、批量预测、加载模型全套资产
from src.predict import (
    predict,
    predict_batch,
    load_model_assets
)
# 导入配置：默认使用的模型版本
from config.settings import DEFAULT_MODEL_VERSION


# ==========================
# 日志模块配置
# 设计目的：统一收集服务运行信息与异常堆栈，方便线上问题排查，后续可扩展落地写入磁盘文件
# ==========================
logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO及以上级别日志都会输出
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 日志输出格式：时间、日志器名、日志级别、日志内容
    datefmt="%Y-%m-%d %H:%M:%S"  # 时间字段格式化
)
# 创建本推理服务专属日志对象
logger = logging.getLogger("churn_infer_service")


# ==========================
# Flask Web服务初始化
# ==========================
app = Flask(__name__)
# 指定当前推理服务使用的模型版本
MODEL_VERSION = DEFAULT_MODEL_VERSION


# ==========================
# 业务必填特征列表
# 设计目的：API层做输入schema校验，提前拦截字段缺失，避免脏数据流入模型，防止出现500服务内部错误
# 必须和训练数据集特征集合保持完全一致
# ==========================
REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# ==========================
# 统一接口返回封装函数
# 设计目的：所有接口输出统一JSON结构，前端调用方可以用同一套逻辑解析成功/失败响应
# :param code: 业务状态码，0代表业务成功，非0代表业务失败
# :param msg: 可读文字描述信息
# :param data: 返回业务数据，无业务数据返回空字典
# :return: flask json响应对象
# ==========================
def make_response(code, msg, data=None):
    return jsonify({
        "code": code,
        "msg": msg,
        "data": data if data is not None else {}
    })


# ==========================
# 服务启动时全局预加载模型资产（触发内部缓存）
# 设计目的：服务启动阶段一次性加载模型至内存，利用predict模块内部的_MODEL_CACHE缓存
# 后续接口请求直接读取内存缓存，避免每次请求重复读取磁盘文件，降低接口耗时
# ==========================
try:
    # 仅调用函数触发缓存预热，返回变量在本文件无使用，不做接收
    load_model_assets(MODEL_VERSION)
    MODEL_LOADED = True
    logger.info(f"模型加载成功:{MODEL_VERSION}")

except Exception as e:
    MODEL_LOADED = False
    # critical级别日志，打印完整异常堆栈，方便定位启动失败原因
    logger.critical(f"模型加载失败:{str(e)}", exc_info=True)


# ==========================
# 单样本流失预测接口 POST /predict
# 请求方式：POST
# 请求体示例：
# {
#     "features": {
#         "gender":"Female",
#         "SeniorCitizen":0,
#         "...其余全部REQUIRED_FEATURES字段"
#     }
# }
# ==========================
@app.route("/predict", methods=["POST"])
def api_predict():
    """
    单样本用户流失预测接口
    """
    # 如果启动阶段模型加载失败，直接返回服务不可用，不再执行推理逻辑
    if not MODEL_LOADED:
        return make_response(503, "模型未加载")

    try:
        # 使用silent=True，解析JSON失败返回None，不直接抛出Flask原生异常
        # 目的：所有非法请求统一进入自定义校验逻辑，输出项目统一格式JSON错误，屏蔽框架原生报错
        req_json = request.get_json(silent=True)

        # 请求不是合法json
        if req_json is None:
            return make_response(400, "请求必须为JSON格式")

        # 严格校验请求顶层必须是JSON对象，拒绝JSON数组，防止调用方传错外层结构
        if not isinstance(req_json, dict):
            return make_response(400, "请求体必须是JSON对象")

        # 判断请求体内是否存在features字段
        if "features" not in req_json:
            return make_response(400, "缺少features字段")

        features = req_json["features"]

        # features必须为字典对象，拒绝数组等其他类型
        if not isinstance(features, dict):
            return make_response(400, "features必须是对象")

        # 必填字段完整性校验，接口层提前拦截缺失字段，不下传给模型推理
        missing = [x for x in REQUIRED_FEATURES if x not in features]
        if missing:
            return make_response(400, f"缺少字段:{missing}")

        # 将特征字典转为DataFrame，适配预测函数入参格式
        raw_df = pd.DataFrame([features])
        # 调用业务预测函数，内部读取全局模型缓存
        result = predict(raw_df, MODEL_VERSION)

        return make_response(0, "success", result)

    except ValueError as e:
        # 参数数值类错误，返回客户端400错误码
        return make_response(400, str(e))

    except Exception as e:
        # 捕获未知异常，记录完整错误堆栈日志，对外屏蔽内部细节
        logger.error(str(e), exc_info=True)
        return make_response(500, "服务器内部异常")


# ==========================
# 批量样本流失预测接口 POST /batch_predict
# 请求方式：POST
# 请求体示例：
# {
#     "features_list": [
#         {"gender":"Female","...":"..."},
#         {"gender":"Male","...":"..."}
#     ]
# }
# ==========================
@app.route("/batch_predict", methods=["POST"])
def api_batch_predict():
    """
    批量用户流失预测接口
    """
    # 模型加载失败直接返回服务不可用
    if not MODEL_LOADED:
        return make_response(503, "模型未加载")

    try:
        # 使用silent=True，解析JSON失败返回None，不直接抛出Flask原生异常
        # 目的：所有非法请求统一进入自定义校验逻辑，输出项目统一格式JSON错误，屏蔽框架原生报错
        req_json = request.get_json(silent=True)

        if req_json is None:
            return make_response(400, "请求必须JSON格式")

        # 严格校验请求顶层必须是JSON对象，拒绝JSON数组
        if not isinstance(req_json, dict):
            return make_response(400, "请求体必须JSON对象")

        if "features_list" not in req_json:
            return make_response(400, "缺少features_list字段")

        features_list = req_json["features_list"]

        # 校验批量输入必须为数组类型
        if not isinstance(features_list, list):
            return make_response(400, "features_list必须数组")

        # 拒绝空批量请求，无样本不需要执行推理
        if len(features_list) == 0:
            return make_response(400, "预测数据不能为空")

        # 逐条校验批量样本：校验单条格式、必填字段；报错带上样本下标，方便调用方定位哪一条数据出错
        for index, item in enumerate(features_list):
            if not isinstance(item, dict):
                return make_response(400, f"第{index}条数据必须对象格式")

            missing = [x for x in REQUIRED_FEATURES if x not in item]
            if missing:
                return make_response(400, f"第{index}条缺少字段:{missing}")

        # 样本字典数组转为DataFrame，适配预测函数入参
        raw_df = pd.DataFrame(features_list)
        # 调用批量预测业务函数，内部读取全局模型缓存
        result = predict_batch(raw_df, MODEL_VERSION)

        return make_response(0, "success", result)

    except ValueError as e:
        return make_response(400, str(e))

    except Exception as e:
        logger.error(str(e), exc_info=True)
        return make_response(500, "批量预测异常")


# ==========================
# 健康检查接口 GET /health
# 设计目的：用于服务监控、k8s存活探针，快速确认服务运行状态、模型加载状态
# ==========================
@app.route("/health", methods=["GET"])
def health_check():
    """
    服务健康检查接口
    返回服务运行状态、使用的模型版本、模型是否加载成功
    """
    return make_response(
        0,
        "service ok",
        {
            "service_status": "running",
            "model_version": MODEL_VERSION,
            "model_loaded": MODEL_LOADED
        }
    )


# ==========================
# 开发环境运行入口
# 注意：Gunicorn生产部署不会执行本块代码，仅直接导入app对象
# ==========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",  # 监听全部网卡，允许局域网其他机器访问
        port=5000,
        debug=False,  # 关闭debug模式，禁止线上开启debug，存在安全风险
        threaded=True  # 开启多线程，支持并发处理多个请求
    )