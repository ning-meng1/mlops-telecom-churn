import requests


BASE_URL = "http://127.0.0.1:5000"

# 测试统计
PASS_COUNT = 0
FAIL_COUNT = 0


# ============================================================
# 通用测试结果判断
# ============================================================

def check_result(test_name, response, expected_code, expected_msg=None):
    """
    统一检查接口测试结果。

    :param test_name: 测试名称
    :param response: requests返回对象
    :param expected_code: 期望的业务code
    :param expected_msg: 可选，期望的msg内容
    """

    global PASS_COUNT, FAIL_COUNT

    try:
        result = response.json()
    except ValueError:
        print(f"[FAIL] {test_name}")
        print("原因：接口返回内容不是合法JSON")
        FAIL_COUNT += 1
        return False

    actual_code = result.get("code")
    actual_msg = result.get("msg", "")

    # 判断业务code
    code_ok = actual_code == expected_code

    # 判断msg
    msg_ok = True
    if expected_msg is not None:
        msg_ok = expected_msg in actual_msg

    if code_ok and msg_ok:
        print(f"[PASS] {test_name}")
        PASS_COUNT += 1
        return True

    print(f"[FAIL] {test_name}")
    print(f"期望 code：{expected_code}")
    print(f"实际 code：{actual_code}")
    if expected_msg is not None:
        print(f"期望 msg：{expected_msg}")
        print(f"实际 msg：{actual_msg}")

    print("完整响应：", result)

    FAIL_COUNT += 1
    return False


def print_response(response):
    """统一打印HTTP状态码和响应结果。"""
    print("HTTP状态码：", response.status_code)

    try:
        print("响应结果：", response.json())
    except ValueError:
        print("响应结果：", response.text)


# ============================================================
# 1. 健康检查
# ============================================================

def test_health():
    print("\n" + "=" * 60)
    print("测试1：健康检查 /health")
    print("=" * 60)

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=10
        )

        print_response(response)

        result = check_result(
            "健康检查 /health",
            response,
            expected_code=0,
            expected_msg="service ok"
        )

        # 进一步检查健康检查返回的数据
        if result:
            data = response.json().get("data", {})

            if (
                data.get("model_loaded") is True
                and data.get("model_version") == "v2"
                and data.get("service_status") == "running"
            ):
                print("[PASS] 健康检查数据字段正确")
                global PASS_COUNT
                PASS_COUNT += 1
            else:
                print("[FAIL] 健康检查数据字段异常")
                print("实际data：", data)
                global FAIL_COUNT
                FAIL_COUNT += 1

        return result

    except Exception as e:
        print("[FAIL] 健康检查接口请求异常：", e)
        FAIL_COUNT += 1
        return False


# ============================================================
# 2. 单样本正常预测
# ============================================================

def test_predict():
    print("\n" + "=" * 60)
    print("测试2：单样本正常预测 /predict")
    print("=" * 60)

    payload = {
        "features": {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 92.5,
            "TotalCharges": 277.5
        }
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=10
        )

        print_response(response)

        result = check_result(
            "单样本正常预测 /predict",
            response,
            expected_code=0,
            expected_msg="success"
        )

        if result:
            data = response.json().get("data", {})

            if (
                "pred_class" in data
                and "confidence" in data
                and "model_version" in data
            ):
                print("[PASS] 单样本预测返回字段完整")
                global PASS_COUNT
                PASS_COUNT += 1
            else:
                print("[FAIL] 单样本预测返回字段不完整")
                print("实际data：", data)
                global FAIL_COUNT
                FAIL_COUNT += 1

        return result

    except Exception as e:
        print("[FAIL] 单样本预测请求异常：", e)
        FAIL_COUNT += 1
        return False


# ============================================================
# 3. 批量正常预测
# ============================================================

def test_batch_predict():
    print("\n" + "=" * 60)
    print("测试3：批量正常预测 /batch_predict")
    print("=" * 60)

    payload = {
        "features_list": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.2,
                "TotalCharges": 1804.8
            },
            {
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 92.5,
                "TotalCharges": 277.5
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            timeout=10
        )

        print_response(response)

        result = check_result(
            "批量正常预测 /batch_predict",
            response,
            expected_code=0,
            expected_msg="success"
        )

        if result:
            data = response.json().get("data", {})

            if (
                "pred_label" in data
                and "pred_proba" in data
                and len(data["pred_label"]) == 2
                and len(data["pred_proba"]) == 2
            ):
                print("[PASS] 批量预测返回字段及数量正确")
                global PASS_COUNT
                PASS_COUNT += 1
            else:
                print("[FAIL] 批量预测返回字段或数量异常")
                print("实际data：", data)
                global FAIL_COUNT
                FAIL_COUNT += 1

        return result

    except Exception as e:
        print("[FAIL] 批量预测请求异常：", e)
        FAIL_COUNT += 1
        return False


# ============================================================
# 4. Day23：参数校验测试
# ============================================================

def test_missing_field():
    print("\n" + "=" * 60)
    print("Day23-测试1：缺少必填字段")
    print("=" * 60)

    payload = {
        "features_list": [
            {
                # 故意删除 gender
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.2,
                "TotalCharges": 1804.8
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            timeout=10
        )

        print_response(response)

        return check_result(
            "缺少必填字段",
            response,
            expected_code=400,
            expected_msg="缺少字段"
        )

    except Exception as e:
        print("[FAIL] 缺少字段测试请求异常：", e)
        FAIL_COUNT += 1
        return False


def test_wrong_type():
    print("\n" + "=" * 60)
    print("Day23-测试2：字段数据类型错误")
    print("=" * 60)

    payload = {
        "features_list": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",

                # 故意传字符串
                "tenure": "abc",

                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.2,
                "TotalCharges": 1804.8
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            timeout=10
        )

        print_response(response)

        return check_result(
            "字段数据类型错误",
            response,
            expected_code=400,
            expected_msg="类型错误"
        )

    except Exception as e:
        print("[FAIL] 类型错误测试请求异常：", e)
        FAIL_COUNT += 1
        return False


def test_out_of_range():
    print("\n" + "=" * 60)
    print("Day23-测试3：数值越界")
    print("=" * 60)

    payload = {
        "features_list": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",

                # 故意超过允许范围
                "tenure": 222,

                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.2,
                "TotalCharges": 1804.8
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            timeout=10
        )

        print_response(response)

        return check_result(
            "数值越界",
            response,
            expected_code=400,
            expected_msg="数值异常"
        )

    except Exception as e:
        print("[FAIL] 数值越界测试请求异常：", e)
        FAIL_COUNT += 1
        return False


def test_empty_list():
    print("\n" + "=" * 60)
    print("Day23-测试4：features_list为空")
    print("=" * 60)

    payload = {
        "features_list": []
    }

    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            timeout=10
        )

        print_response(response)

        return check_result(
            "features_list为空",
            response,
            expected_code=400,
            expected_msg="预测数据不能为空"
        )

    except Exception as e:
        print("[FAIL] 空批量测试请求异常：", e)
        FAIL_COUNT += 1
        return False


def test_wrong_list_type():
    print("\n" + "=" * 60)
    print("Day23-测试5：features_list类型错误")
    print("=" * 60)

    payload = {
        "features_list": {
            "gender": "Male",
            "SeniorCitizen": 1
        }
    }

    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            timeout=10
        )

        print_response(response)

        return check_result(
            "features_list类型错误",
            response,
            expected_code=400,
            expected_msg="features_list必须数组"
        )

    except Exception as e:
        print("[FAIL] features_list类型错误测试请求异常：", e)
        FAIL_COUNT += 1
        return False


# ============================================================
# 5. Day24：异常处理测试
# ============================================================

def test_internal_exception():
    print("\n" + "=" * 60)
    print("Day24-测试1：故意触发内部异常")
    print("=" * 60)

    try:
        response = requests.get(
            f"{BASE_URL}/test_error",
            timeout=10
        )

        print_response(response)

        result = check_result(
            "内部异常处理",
            response,
            expected_code=500,
            expected_msg="服务器内部异常"
        )

        # 检查客户端响应不能暴露Traceback
        if "Traceback" in response.text:
            print("[FAIL] 客户端响应暴露了Traceback")
            global FAIL_COUNT
            FAIL_COUNT += 1
            return False

        print("[PASS] 客户端未暴露Traceback")

        global PASS_COUNT
        PASS_COUNT += 1

        return result

    except Exception as e:
        print("[FAIL] 内部异常测试请求异常：", e)
        FAIL_COUNT += 1
        return False


def test_health_after_exception():
    print("\n" + "=" * 60)
    print("Day24-测试2：异常后检查服务是否仍然正常")
    print("=" * 60)

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=10
        )

        print_response(response)

        return check_result(
            "异常后服务健康检查",
            response,
            expected_code=0,
            expected_msg="service ok"
        )

    except Exception as e:
        print("[FAIL] 异常后健康检查请求异常：", e)
        FAIL_COUNT += 1
        return False


# ============================================================
# 6. Day26：完整自动化回归测试
# ============================================================

def test_day26_regression():

    print("\n")
    print("#" * 70)
    print("# Day26：API接口自动化回归测试")
    print("#" * 70)

    # --------------------------------------------------------
    # 正常场景
    # --------------------------------------------------------

    test_health()

    test_predict()

    test_batch_predict()

    # --------------------------------------------------------
    # 异常场景
    # --------------------------------------------------------

    test_missing_field()

    test_wrong_type()

    test_out_of_range()

    test_empty_list()

    test_wrong_list_type()

    # --------------------------------------------------------
    # 内部异常场景
    # --------------------------------------------------------

    test_internal_exception()

    test_health_after_exception()

    # --------------------------------------------------------
    # 最终统计
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Day26 自动化测试结果")
    print("=" * 70)

    print(f"PASS：{PASS_COUNT}")
    print(f"FAIL：{FAIL_COUNT}")
    print(f"TOTAL：{PASS_COUNT + FAIL_COUNT}")

    print("=" * 70)

    if FAIL_COUNT == 0:
        print("✅ 全部接口自动化测试通过")
        print("✅ 正常请求场景通过")
        print("✅ 异常请求场景通过")
        print("✅ 内部异常处理通过")
        print("✅ 异常后服务仍保持正常")
        print("✅ Day26 API自动化测试验收通过")
    else:
        print("❌ 存在测试失败，请检查上面的 [FAIL] 项")

        # 让命令行返回失败状态
        raise SystemExit(1)

    print("=" * 70)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    test_day26_regression()