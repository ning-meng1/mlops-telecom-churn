import math

from config.settings import (
    REQUIRED_FEATURES,
    FEATURE_RANGE_RULES,
    FEATURE_ALLOWED_VALUES
)


def validate_features(features):
    """
    API输入参数统一校验。

    校验内容：
    1. features是否为JSON对象
    2. 必填字段是否缺失
    3. 数值字段类型是否合法
    4. 数值字段是否为有限数字
    5. 数值字段是否越界
    6. 类别字段是否属于合法业务枚举

    返回：
        None:
            校验通过

        str:
            校验失败原因
    """

    # =========================
    # 1. 基础类型校验
    # =========================

    if not isinstance(features, dict):
        return "features必须是JSON对象"

    # =========================
    # 2. 字段完整性校验
    # =========================

    missing = [
        field
        for field in REQUIRED_FEATURES
        if field not in features
    ]

    if missing:
        return f"缺少字段: {missing}"

    # =========================
    # 3. 数值字段类型校验
    # =========================

    for field in FEATURE_RANGE_RULES:

        value = features[field]

        # bool虽然属于int子类，但业务上不是合法数字
        if isinstance(value, bool):
            return f"{field}类型错误，必须为数字，不能为布尔值"

        if not isinstance(value, (int, float)):
            return f"{field}类型错误，必须为数字"

        # 防止NaN / Infinity进入模型
        if not math.isfinite(float(value)):
            return f"{field}数值异常，不能为NaN或Infinity"

    # =========================
    # 4. 数值范围校验
    # =========================

    for field, (min_value, max_value) in FEATURE_RANGE_RULES.items():

        value = features[field]

        if value < min_value or value > max_value:
            return (
                f"{field}数值异常，"
                f"允许范围: {min_value}-{max_value}，"
                f"实际值: {value}"
            )

    # =========================
    # 5. 类别字段业务枚举校验
    # =========================

    for field, allowed_values in FEATURE_ALLOWED_VALUES.items():

        value = features[field]

        if not isinstance(value, str):
            return (
                f"{field}类型错误，"
                f"必须为字符串"
            )

        if value not in allowed_values:
            return (
                f"{field}业务值非法，"
                f"允许值: {sorted(allowed_values)}，"
                f"实际值: {value}"
            )

    return None